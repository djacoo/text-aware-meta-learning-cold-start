from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from coldstart.model import MeLU

RATINGS = torch.arange(1.0, 6.0)


class ProtoNet(nn.Module):
    # Prototypical Networks (Snell et al., 2017), the five ratings are the classes
    def __init__(self, data, emb_dim=32, hidden=64, text="none", id_dropout=0.0):
        super().__init__()
        # the user is the same for every item of a task, so only items are embedded
        self.encoder = MeLU(replace(data, user_features={}), emb_dim, hidden, text, id_dropout)

    def log_probs(self, task):
        device = self.encoder.fc1.weight.device
        support = self.encoder.hidden(None, task.support_items.to(device))
        query = self.encoder.hidden(None, task.query_items.to(device))
        classes = task.support_ratings.to(device).long() - 1
        present = classes.unique()
        prototypes = torch.stack([support[classes == c].mean(dim=0) for c in present])
        distances = ((query[:, None] - prototypes[None]) ** 2).sum(dim=2)
        # ratings the user never gave in the support set get no probability
        logits = torch.full((len(query), len(RATINGS)), float("-inf"), device=device)
        logits[:, present] = -distances
        return F.log_softmax(logits, dim=1)

    def predict(self, task):
        # expected rating, so it can be scored with MAE like MeLU
        return self.log_probs(task).exp() @ RATINGS.to(self.encoder.fc1.weight.device)


def proto_step(model, tasks, optimizer):
    losses = []
    for task in tasks:
        log_probs = model.log_probs(task)
        classes = task.query_ratings.to(log_probs.device).long() - 1
        # a query rating missing from the support set has no prototype, it is left out
        seen = torch.isfinite(log_probs.gather(1, classes[:, None]).squeeze(1))
        if seen.any():
            losses.append(F.nll_loss(log_probs[seen], classes[seen]))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def protonet_predictor(model):
    model.eval()

    def protonet(task):
        with torch.no_grad():
            return model.predict(task).cpu()

    return protonet

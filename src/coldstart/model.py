from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from coldstart.dataset import Task
from coldstart.prototype import add_prototype

# only the decision layers adapt to each user, as in MeLU
LOCAL_PARAMS = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "out.weight", "out.bias")
ID_FIELDS = ("director", "actors")
RATINGS = torch.arange(1.0, 6.0)


class MeLU(nn.Module):
    def __init__(self, data, emb_dim=32, hidden=64, text="none", id_dropout=0.0, adapt_text=False):
        super().__init__()
        # text is "none", "only" (text replaces the item fields) or "concat" (text added to them)
        self.text = text
        self.id_dropout = id_dropout
        self.local_params = LOCAL_PARAMS
        if adapt_text:
            self.local_params += ("text_proj.weight", "text_proj.bias")
        item_features = {} if text == "only" else data.item_features
        self.item_fields = list(item_features)
        self.user_fields = list(data.user_features)
        self.emb = nn.ModuleDict()
        for prefix, features in (("item", item_features), ("user", data.user_features)):
            for name, table in features.items():
                self.register_buffer(f"{prefix}_{name}", table)
                size = data.vocab_sizes[name]
                # multi-valued fields (genre, director, actors) are padded index lists
                if table.dim() == 2:
                    self.emb[name] = nn.EmbeddingBag(size, emb_dim, mode="mean", padding_idx=0)
                else:
                    self.emb[name] = nn.Embedding(size, emb_dim, padding_idx=0)

        n_fields = len(self.item_fields) + len(self.user_fields)
        if text != "none":
            # text embeddings are fixed, only the projection is learned
            self.register_buffer("item_text", data.text_embeddings)
            self.text_proj = nn.Linear(data.text_embeddings.shape[1], emb_dim)
            n_fields += 1
        self.fc1 = nn.Linear(n_fields * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def hidden(self, users, items):
        x = []
        for name in self.item_fields:
            v = self.emb[name](getattr(self, f"item_{name}")[items])
            if self.training and name in ID_FIELDS and self.id_dropout:
                # hide directors and actors of some items, so the model learns to use the text
                keep = torch.rand(len(items), 1, device=v.device) >= self.id_dropout
                v = v * keep
            x.append(v)
        if self.text != "none":
            x.append(self.text_proj(self.item_text[items]))
        x += [self.emb[name](getattr(self, f"user_{name}")[users]) for name in self.user_fields]
        x = F.relu(self.fc1(torch.cat(x, dim=1)))
        return F.relu(self.fc2(x))

    def forward(self, users, items):
        return self.out(self.hidden(users, items)).squeeze(1)


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


def build_model(data, cfg):
    text, id_dropout = cfg.get("text", "none"), cfg.get("id_dropout", 0.0)
    if cfg.get("model") == "protonet":
        return ProtoNet(data, cfg["emb_dim"], cfg["hidden"], text, id_dropout)
    model = MeLU(
        data, cfg["emb_dim"], cfg["hidden"], text, id_dropout, cfg.get("adapt_text", False)
    )
    if cfg.get("proto", False):
        add_prototype(model, data, cfg.get("proto_dim", 32))
    return model


def task_inputs(task, device, query=False):
    items = task.query_items if query else task.support_items
    ratings = task.query_ratings if query else task.support_ratings
    items = items.to(device)
    return torch.full_like(items, task.user), items, ratings.to(device)


def predict(model, params, users, items):
    return functional_call(model, params, (users, items))


def adapt(model, task, steps, lr, first_order=False):
    # fast weights stay in the graph, so the outer update sees through the inner one;
    # the original code used load_state_dict here, which cut the graph (MeLU issue #9)
    device = next(model.parameters()).device
    users, items, ratings = task_inputs(task, device)
    params = dict(model.named_parameters())
    if hasattr(model, "proto_net"):
        # the text prototype of the support ratings shifts the output layer before the inner loop
        prototype = model.proto_net(items, ratings)
        for name, shift in model.proto_projector(prototype, params).items():
            params[name] = params[name] + shift
    for _ in range(steps):
        loss = F.mse_loss(predict(model, params, users, items), ratings)
        local = [params[name] for name in model.local_params]
        grads = torch.autograd.grad(loss, local, create_graph=not first_order)
        for name, grad in zip(model.local_params, grads, strict=True):
            params[name] = params[name] - lr * grad
    return params


def meta_step(model, tasks, optimizer, steps, lr):
    device = next(model.parameters()).device
    losses = []
    for task in tasks:
        params = adapt(model, task, steps, lr)
        users, items, ratings = task_inputs(task, device, query=True)
        losses.append(F.mse_loss(predict(model, params, users, items), ratings))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def start_params(model):
    # the model a user has before any rating: with a prototype, shifted by an empty support
    empty = Task(0, torch.empty(0, dtype=torch.long), torch.empty(0), None, None)
    return adapt(model, empty, 0, 0.0)

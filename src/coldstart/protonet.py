import torch
import torch.nn.functional as F


def proto_step(model, tasks, optimizer):
    losses = []
    for task in tasks:
        log_probs = model.log_probs(task)
        classes = task.query_ratings.to(log_probs.device).long() - 1
        # a query rating missing from the support set has no prototype, it is left out
        seen = torch.isfinite(log_probs.gather(1, classes[:, None]).squeeze(1))
        if seen.any():
            losses.append(F.nll_loss(log_probs[seen], classes[seen]))
    if not losses:
        # no query rating of the batch has a prototype, nothing to learn from
        return float("nan")
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

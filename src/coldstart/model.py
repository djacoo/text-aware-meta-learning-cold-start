import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

# Only the decision layers adapt to each user, as in MeLU
LOCAL_PARAMS = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "out.weight", "out.bias")


class MeLU(nn.Module):
    def __init__(self, data, emb_dim=32, hidden=64):
        super().__init__()
        self.item_fields = list(data.item_features)
        self.user_fields = list(data.user_features)
        self.emb = nn.ModuleDict()
        for prefix, features in (("item", data.item_features), ("user", data.user_features)):
            for name, table in features.items():
                self.register_buffer(f"{prefix}_{name}", table)
                size = data.vocab_sizes[name]
                # multi-valued fields (genre, director, actors) are padded index lists
                if table.dim() == 2:
                    self.emb[name] = nn.EmbeddingBag(size, emb_dim, mode="mean", padding_idx=0)
                else:
                    self.emb[name] = nn.Embedding(size, emb_dim, padding_idx=0)

        n_fields = len(self.item_fields) + len(self.user_fields)
        self.fc1 = nn.Linear(n_fields * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, users, items):
        x = [self.emb[name](getattr(self, f"item_{name}")[items]) for name in self.item_fields]
        x += [self.emb[name](getattr(self, f"user_{name}")[users]) for name in self.user_fields]
        x = F.relu(self.fc1(torch.cat(x, dim=1)))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(1)


def task_inputs(task, device, query=False):
    items = task.query_items if query else task.support_items
    ratings = task.query_ratings if query else task.support_ratings
    items = items.to(device)
    return torch.full_like(items, task.user), items, ratings.to(device)


def predict(model, params, users, items):
    return functional_call(model, params, (users, items))


def adapt(model, task, steps, lr, first_order=False):
    # Fast weights stay in the graph, so the outer update sees through the inner one.
    # The original code used load_state_dict here, which cut the graph (MeLU issue #9).
    users, items, ratings = task_inputs(task, next(model.parameters()).device)
    params = dict(model.named_parameters())
    for _ in range(steps):
        loss = F.mse_loss(predict(model, params, users, items), ratings)
        local = [params[name] for name in LOCAL_PARAMS]
        grads = torch.autograd.grad(loss, local, create_graph=not first_order)
        for name, grad in zip(LOCAL_PARAMS, grads, strict=True):
            params[name] = params[name] - lr * grad
    return params


def meta_step(model, tasks, optimizer, steps, lr, first_order=False):
    device = next(model.parameters()).device
    losses = []
    for task in tasks:
        params = adapt(model, task, steps, lr, first_order)
        users, items, ratings = task_inputs(task, device, query=True)
        losses.append(F.mse_loss(predict(model, params, users, items), ratings))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

# Only the decision layers adapt to each user, as in MeLU
LOCAL_PARAMS = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "out.weight", "out.bias")
ID_FIELDS = ("director", "actors")


class MeLU(nn.Module):
    def __init__(self, data, emb_dim=32, hidden=64, text="none", id_dropout=0.0):
        super().__init__()
        # text is "none", "only" (plot replaces the item fields) or "concat" (plot added to them)
        self.text = text
        self.id_dropout = id_dropout
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
            # plot embeddings are fixed, only the projection is learned
            self.register_buffer("item_text", data.text_embeddings)
            self.text_proj = nn.Linear(data.text_embeddings.shape[1], emb_dim)
            n_fields += 1
        self.fc1 = nn.Linear(n_fields * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, users, items):
        x = []
        for name in self.item_fields:
            v = self.emb[name](getattr(self, f"item_{name}")[items])
            if self.training and name in ID_FIELDS and self.id_dropout:
                # hide directors and actors of some items, so the model learns to use the plot
                keep = torch.rand(len(items), 1, device=v.device) >= self.id_dropout
                v = v * keep
            x.append(v)
        if self.text != "none":
            x.append(self.text_proj(self.item_text[items]))
        x += [self.emb[name](getattr(self, f"user_{name}")[users]) for name in self.user_fields]
        x = F.relu(self.fc1(torch.cat(x, dim=1)))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(1)


def build_model(data, cfg):
    return MeLU(
        data,
        cfg["emb_dim"],
        cfg["hidden"],
        cfg.get("text", "none"),
        cfg.get("id_dropout", 0.0),
    )


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

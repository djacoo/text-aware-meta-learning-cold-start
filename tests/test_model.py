from types import SimpleNamespace

import torch

from coldstart.model import LOCAL_PARAMS, MeLU, adapt, meta_step, predict, task_inputs

N_ITEMS, N_USERS = 30, 10


def fake_data():
    torch.manual_seed(0)
    item_features = {
        "rate": torch.randint(1, 4, (N_ITEMS,)),
        "genre": torch.randint(0, 6, (N_ITEMS, 3)),
        "director": torch.randint(0, 5, (N_ITEMS, 2)),
        "actors": torch.randint(0, 8, (N_ITEMS, 4)),
    }
    user_features = {
        "gender": torch.randint(1, 3, (N_USERS,)),
        "age": torch.randint(1, 8, (N_USERS,)),
        "occupation": torch.randint(1, 22, (N_USERS,)),
        "zip": torch.randint(1, 20, (N_USERS,)),
    }
    sizes = {"rate": 4, "genre": 6, "director": 5, "actors": 8}
    sizes |= {"gender": 3, "age": 8, "occupation": 22, "zip": 20}
    return SimpleNamespace(
        item_features=item_features, user_features=user_features, vocab_sizes=sizes
    )


def fake_task():
    items = torch.randperm(N_ITEMS)[:20]
    ratings = torch.randint(1, 6, (20,)).float()
    return SimpleNamespace(
        user=3,
        support_items=items[:10],
        support_ratings=ratings[:10],
        query_items=items[10:],
        query_ratings=ratings[10:],
    )


def query_loss(model, params, task):
    users, items, ratings = task_inputs(task, "cpu", query=True)
    return torch.nn.functional.mse_loss(predict(model, params, users, items), ratings)


def test_output_shape():
    model = MeLU(fake_data(), emb_dim=8, hidden=16)
    items = torch.arange(5)
    assert model(torch.zeros_like(items), items).shape == (5,)


def test_adapt_changes_only_local_params():
    model = MeLU(fake_data(), emb_dim=8, hidden=16)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    params = adapt(model, fake_task(), steps=2, lr=0.1)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])
        if name in LOCAL_PARAMS:
            assert not torch.equal(params[name], p)
        else:
            assert params[name] is p


def test_second_order_reaches_embeddings():
    model = MeLU(fake_data(), emb_dim=8, hidden=16)
    task = fake_task()
    grads = []
    for first_order in (True, False):
        model.zero_grad()
        query_loss(model, adapt(model, task, 1, 0.1, first_order), task).backward()
        grads.append(model.emb["rate"].weight.grad.clone())
    assert not torch.allclose(grads[0], grads[1])


def test_meta_step_lowers_query_loss():
    model = MeLU(fake_data(), emb_dim=8, hidden=16)
    task = fake_task()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start = query_loss(model, adapt(model, task, 1, 0.01), task).item()
    for _ in range(20):
        meta_step(model, [task], optimizer, steps=1, lr=0.01)
    assert query_loss(model, adapt(model, task, 1, 0.01), task).item() < start

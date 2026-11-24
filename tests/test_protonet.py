import pytest
import torch

from coldstart import compare
from coldstart.dataset import Task
from coldstart.evaluate import load_model, predictor, score
from coldstart.model import ProtoNet
from coldstart.preprocess import load_tasks
from coldstart.protonet import proto_step
from coldstart.train import train


def make_task(support_ratings):
    items = torch.tensor([0, 1, 2, 0, 1, 2])[: len(support_ratings)]
    return Task(0, items, torch.tensor(support_ratings), torch.tensor([0, 1, 2]), torch.ones(3))


def test_prototypes_only_for_given_ratings(data):
    torch.manual_seed(0)
    model = ProtoNet(data, emb_dim=8, hidden=16)
    probs = model.log_probs(make_task([2.0, 4.0, 4.0])).exp()
    assert torch.allclose(probs.sum(dim=1), torch.ones(3))
    # ratings 1, 3 and 5 are not in the support set
    assert torch.all(probs[:, [0, 2, 4]] == 0)
    # a single rating in the support set is the only possible answer
    assert torch.allclose(model.predict(make_task([4.0, 4.0, 4.0])), torch.full((3,), 4.0))


def test_proto_step_learns(data):
    torch.manual_seed(0)
    model = ProtoNet(data, emb_dim=8, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    support = torch.tensor([0, 1, 2]), torch.tensor([1.0, 3.0, 5.0])
    task = Task(0, *support, torch.tensor([0, 2]), torch.tensor([1.0, 5.0]))
    losses = [proto_step(model, [task], optimizer) for _ in range(30)]
    assert losses[-1] < losses[0]


def test_train_evaluate_compare(ml_root, data, tmp_path):
    cfg = {
        "data_root": str(ml_root),
        "emb_dim": 8,
        "hidden": 16,
        "outer_lr": 0.01,
        "meta_batch": 4,
        "epochs": 2,
        "model": "protonet",
    }
    train(cfg, seed=0, out_dir=tmp_path)
    path = str(tmp_path / "protonet_seed0.pt")
    model, saved_cfg, seed = load_model(path, data)
    assert saved_cfg == cfg and not model.training

    scores = score(load_tasks(data, "user_cold", seed), predictor(model, saved_cfg))
    assert 0 <= scores["mae"] <= 4
    errors, mean_errors = compare.user_errors([path])["user_cold"]
    assert errors.mean() == pytest.approx(scores["mae"])

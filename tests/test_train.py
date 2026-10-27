import torch

from coldstart.evaluate import evaluate, load_model, user_mean
from coldstart.preprocess import load_tasks
from coldstart.train import train


def test_train_then_evaluate(ml_root, data, tmp_path):
    cfg = {
        "data_root": str(ml_root),
        "emb_dim": 8,
        "hidden": 16,
        "inner_lr": 0.01,
        "outer_lr": 0.01,
        "inner_steps": 1,
        "meta_batch": 4,
        "epochs": 2,
    }
    train(cfg, seed=0, out_dir=tmp_path / "checkpoints")

    model, saved_cfg, seed = load_model(tmp_path / "checkpoints" / "melu_seed0.pt", data)
    assert saved_cfg == cfg
    assert seed == 0

    scores = evaluate(model, load_tasks(data, "user_cold", seed), steps=1, lr=0.01)
    assert scores["mae"] >= 0
    assert 0 <= scores["ndcg@1"] <= 1
    assert 0 <= scores["ndcg@3"] <= 1


def test_user_mean_baseline(data):
    task = load_tasks(data, "user_cold", seed=0)[0]
    pred = user_mean(task)
    assert pred.shape == (10,)
    assert torch.allclose(pred, task.support_ratings.mean().expand(10))

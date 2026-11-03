import pytest
import torch

from coldstart.compare import user_errors
from coldstart.dataset import Task
from coldstart.evaluate import evaluate, load_model, text_knn, user_mean
from coldstart.preprocess import load_movielens, load_tasks
from coldstart.train import train


@pytest.mark.parametrize("text", [{}, {"text": "concat", "text_encoder": "someone/fake-encoder"}])
def test_train_then_evaluate(ml_root, tmp_path, text):
    torch.save(torch.rand(3, 4), ml_root / "plots_fake-encoder.pt")
    cfg = {
        "data_root": str(ml_root),
        "emb_dim": 8,
        "hidden": 16,
        "inner_lr": 0.01,
        "outer_lr": 0.01,
        "inner_steps": 1,
        "meta_batch": 4,
        "epochs": 2,
    } | text
    train(cfg, seed=0, out_dir=tmp_path / "checkpoints")

    data = load_movielens(ml_root, cfg.get("text_encoder"))
    model, saved_cfg, seed = load_model(tmp_path / "checkpoints" / "melu_seed0.pt", data)
    assert saved_cfg == cfg
    assert seed == 0

    scores = evaluate(model, load_tasks(data, "user_cold", seed), steps=1, lr=0.01)
    assert scores["mae"] >= 0
    assert 0 <= scores["ndcg@1"] <= 1
    assert 0 <= scores["ndcg@3"] <= 1

    model_errors, mean_errors = user_errors(
        [tmp_path / "checkpoints" / "melu_seed0.pt"], "user_cold"
    )
    assert model_errors.shape == mean_errors.shape == (len(load_tasks(data, "user_cold", 0)),)


def test_user_mean_baseline(data):
    task = load_tasks(data, "user_cold", seed=0)[0]
    pred = user_mean(task)
    assert pred.shape == (10,)
    assert torch.allclose(pred, task.support_ratings.mean().expand(10))


def test_text_knn():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    task = Task(0, torch.tensor([0, 1]), torch.tensor([5.0, 1.0]), torch.tensor([2, 3]), None)
    # item 2 matches item 0 only, item 3 has no text and gets the user mean
    assert text_knn(task, embeddings).tolist() == [5.0, 3.0]

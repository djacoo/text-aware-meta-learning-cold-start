import random

import numpy as np
import pytest
import torch

from coldstart import active_learning
from coldstart.active_learning import (
    compute_rating_frequencies,
    select_ensemble_uncertainty,
    select_expected_model_change,
    select_hybrid,
    select_random,
    select_text_diversity,
)
from coldstart.evaluate import load_model, melu_predictor, task_scores
from coldstart.model import MeLU, start_params
from coldstart.preprocess import load_data, load_tasks
from coldstart.run_active_learning import choose_support, evaluate_active_learning
from coldstart.run_active_learning import main as run_main
from coldstart.train import train


@pytest.fixture
def setup(data):
    torch.manual_seed(0)
    models = [MeLU(data, emb_dim=8, hidden=16) for _ in range(2)]
    tasks = load_tasks(data, "warm_train", seed=0)
    kwargs = {
        "ensemble_models": models,
        "melu_model": models[0],
        "rating_freqs": compute_rating_frequencies(tasks),
        "text_embeddings": torch.rand(3, 4),
    }
    return tasks, kwargs


@pytest.mark.parametrize(
    "strategy",
    [
        select_random,
        select_ensemble_uncertainty,
        select_expected_model_change,
        select_hybrid,
        select_text_diversity,
    ],
)
def test_strategy_picks_k_distinct_items(setup, strategy):
    tasks, kwargs = setup
    task = max(tasks, key=lambda t: len(t.support_items))
    idxs = strategy(task.support_items, 3, user_id=task.user, **kwargs)
    assert len(set(idxs.tolist())) == 3
    assert all(0 <= i < len(task.support_items) for i in idxs.tolist())


def test_rating_frequencies_sum_to_one(setup):
    _, kwargs = setup
    assert sum(kwargs["rating_freqs"].values()) == pytest.approx(1.0)


def test_choose_support_keeps_query(setup):
    tasks, kwargs = setup
    task = max(tasks, key=lambda t: len(t.support_items))
    chosen = choose_support(task, 2, select_random)
    assert len(chosen.support_items) == 2
    assert torch.equal(chosen.query_items, task.query_items)
    assert torch.equal(chosen.query_ratings, task.query_ratings)
    # a support smaller than k is kept whole
    assert len(choose_support(task, 100, select_random).support_items) == len(task.support_items)


def test_run_active_learning(ml_root, tmp_path, capsys):
    torch.save(torch.rand(3, 4), ml_root / "plots_fake-encoder.pt")
    cfg = {
        "data_root": str(ml_root),
        "emb_dim": 8,
        "hidden": 16,
        "inner_lr": 0.01,
        "outer_lr": 0.01,
        "inner_steps": 1,
        "meta_batch": 4,
        "epochs": 1,
        "text_encoder": "someone/fake-encoder",
    }
    paths = []
    for seed in (0, 1):
        train(cfg, seed, tmp_path / str(seed))
        paths.append(str(tmp_path / str(seed) / f"melu_seed{seed}.pt"))

    capsys.readouterr()
    run_main(paths + ["--states", "user_cold", "--k", "1", "2"])
    lines = [line for line in capsys.readouterr().out.split("\n") if line.strip()]
    # header, then one line per strategy and k
    assert len(lines) == 1 + 5 * 2
    assert lines[0].split()[-1] == "choose"
    tasks = load_tasks(load_data(ml_root, cfg["text_encoder"]), "user_cold", 0)
    for line in lines[1:]:
        k = int(line.split()[-5])
        share = np.mean([len(task.support_items) > k for task in tasks])
        assert line.split()[-1] == f"{share:.2f}"


def test_active_learning_uses_prototype(ml_root, tmp_path, monkeypatch):
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
        "text_encoder": "someone/fake-encoder",
        "proto": True,
        "proto_dim": 4,
    }
    train(cfg, seed=0, out_dir=tmp_path)
    data = load_data(ml_root, cfg["text_encoder"])
    model, _, _ = load_model(tmp_path / "proto_melu_seed0.pt", data)
    tasks = load_tasks(data, "user_cold", 0)

    # with k above every support size, active learning must score like plain evaluation
    expected = task_scores(tasks, melu_predictor(model, 1, 0.01, data=data))["mae"]
    result = evaluate_active_learning(model, tasks, 1, 0.01, 100, 0, data, select_random)
    assert result["mae"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        melu_predictor(model, 1, 0.01)(tasks[0])

    # before any rating a proto model is already shifted by the empty prototype
    start, shared = start_params(model, data), dict(model.named_parameters())
    assert not torch.equal(start["out.weight"], shared["out.weight"])
    assert torch.equal(start["fc1.weight"], shared["fc1.weight"])
    task = max(tasks, key=lambda t: len(t.support_items))
    kwargs = {"ensemble_models": [model, model], "melu_model": model, "data": data}
    kwargs["rating_freqs"] = compute_rating_frequencies(tasks)
    used = []
    monkeypatch.setattr(
        active_learning, "start_params", lambda m, d: used.append(d) or start_params(m, d)
    )
    for strategy in (select_ensemble_uncertainty, select_expected_model_change):
        assert len(strategy(task.support_items, 2, user_id=task.user, **kwargs)) == 2
    # both strategies score with the zero-rating model, given the text data
    assert len(used) == 3 and all(d is data for d in used)


def test_text_diversity_picks_films_without_text_last():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.0]])
    items = torch.arange(4)
    for seed in range(10):
        random.seed(seed)
        picked = select_text_diversity(items, 3, embeddings).tolist()
        # after the random first pick, the film without text is never preferred
        assert 3 not in picked[1:]


def test_hybrid_starts_with_model_change(setup):
    tasks, kwargs = setup
    task = max(tasks, key=lambda t: len(t.support_items))
    first = select_expected_model_change(task.support_items, 1, user_id=task.user, **kwargs)
    picked = select_hybrid(task.support_items, 3, user_id=task.user, **kwargs)
    assert picked[0] == first[0] and len(set(picked.tolist())) == 3

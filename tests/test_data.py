import torch

from coldstart.preprocess import load_tasks


def test_item_features(data):
    items = data.item_features
    assert items["rate"].tolist() == [1, 2, 1]
    assert items["genre"].tolist() == [[2, 1], [2, 0], [0, 0]]
    assert items["director"].tolist() == [[1, 0], [1, 2], [0, 0]]
    assert items["actors"].tolist() == [[1, 2], [1, 0], [0, 0]]
    assert data.vocab_sizes["actors"] == 3
    assert data.plots[0] == "A plot."


def test_user_features(data):
    assert data.user_features["gender"][:2].tolist() == [1, 2]
    assert data.user_features["zip"].unique().tolist() == [1]


def test_tasks(data):
    tasks = load_tasks(data, "user_cold", seed=0)
    assert len(tasks) == 18
    assert all(len(task.query_items) == 10 for task in tasks)
    assert all(len(task.support_items) == len(task.support_ratings) for task in tasks)

    again = load_tasks(data, "user_cold", seed=0)
    assert all(
        torch.equal(a.query_ratings, b.query_ratings) for a, b in zip(tasks, again, strict=True)
    )


def test_warm_split(data):
    parts = [
        {task.user for task in load_tasks(data, s, seed=0)}
        for s in ("warm_train", "warm_val", "warm_test")
    ]
    assert [len(p) for p in parts] == [14, 1, 3]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert {task.user for task in load_tasks(data, "warm_val", seed=7)} == parts[1]

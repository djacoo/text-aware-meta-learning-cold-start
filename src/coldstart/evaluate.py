import argparse

import numpy as np
import torch

from coldstart.metrics import mae, ndcg_at_k
from coldstart.model import MeLU, adapt, predict
from coldstart.preprocess import load_movielens, load_tasks

TEST_STATES = ("warm_test", "user_cold", "item_cold", "user_item_cold")
METRICS = ("mae", "ndcg@1", "ndcg@3")


def evaluate(model, tasks, steps, lr):
    device = next(model.parameters()).device
    scores = {name: [] for name in METRICS}
    for task in tasks:
        # first order is enough at test time, no meta-gradient is needed
        params = adapt(model, task, steps, lr, first_order=True)
        items = task.query_items.to(device)
        users = torch.full_like(items, task.user)
        with torch.no_grad():
            pred = predict(model, params, users, items).cpu()
        scores["mae"].append(mae(pred, task.query_ratings))
        scores["ndcg@1"].append(ndcg_at_k(pred, task.query_ratings, 1))
        scores["ndcg@3"].append(ndcg_at_k(pred, task.query_ratings, 3))
    return {name: float(np.mean(values)) for name, values in scores.items()}


def load_model(path, data):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = checkpoint["config"]
    model = MeLU(data, cfg["emb_dim"], cfg["hidden"])
    model.load_state_dict(checkpoint["state"])
    return model, cfg, checkpoint["seed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one checkpoint per seed")
    args = parser.parse_args()

    cfg = torch.load(args.checkpoints[0], map_location="cpu")["config"]
    data = load_movielens(cfg["data_root"])
    results = {state: {name: [] for name in METRICS} for state in TEST_STATES}
    for path in args.checkpoints:
        model, cfg, seed = load_model(path, data)
        for state in TEST_STATES:
            tasks = load_tasks(data, state, seed)
            scores = evaluate(model, tasks, cfg["inner_steps"], cfg["inner_lr"])
            for name in METRICS:
                results[state][name].append(scores[name])

    print(f"{'state':<16}" + "".join(f"{name:>19}" for name in METRICS))
    for state in TEST_STATES:
        row = "".join(f"{np.mean(v):>10.4f} ± {np.std(v):.4f}" for v in results[state].values())
        print(f"{state:<16}{row}")


if __name__ == "__main__":
    main()

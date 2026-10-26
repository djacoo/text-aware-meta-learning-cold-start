import argparse
from functools import partial

import numpy as np
import torch

from coldstart.metrics import mae, ndcg_at_k
from coldstart.model import MeLU, adapt, predict
from coldstart.preprocess import load_movielens, load_tasks

TEST_STATES = ("warm_test", "user_cold", "item_cold", "user_item_cold")
METRICS = ("mae", "ndcg@1", "ndcg@3")


def score(tasks, predict_query):
    scores = {name: [] for name in METRICS}
    for task in tasks:
        pred = predict_query(task)
        scores["mae"].append(mae(pred, task.query_ratings))
        scores["ndcg@1"].append(ndcg_at_k(pred, task.query_ratings, 1))
        scores["ndcg@3"].append(ndcg_at_k(pred, task.query_ratings, 3))
    return {name: float(np.mean(values)) for name, values in scores.items()}


def evaluate(model, tasks, steps, lr):
    device = next(model.parameters()).device

    def melu(task):
        # first order is enough at test time, no meta-gradient is needed
        params = adapt(model, task, steps, lr, first_order=True)
        items = task.query_items.to(device)
        with torch.no_grad():
            return predict(model, params, torch.full_like(items, task.user), items).cpu()

    return score(tasks, melu)


def user_mean(task):
    return task.support_ratings.mean().expand(len(task.query_items))


def random_ratings(task, generator):
    return 1 + 4 * torch.rand(len(task.query_items), generator=generator)


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
    methods = ("melu", "user mean", "random")
    results = {m: {s: {name: [] for name in METRICS} for s in TEST_STATES} for m in methods}
    for path in args.checkpoints:
        model, cfg, seed = load_model(path, data)
        generator = torch.Generator().manual_seed(seed)
        for state in TEST_STATES:
            tasks = load_tasks(data, state, seed)
            scores = {
                "melu": evaluate(model, tasks, cfg["inner_steps"], cfg["inner_lr"]),
                "user mean": score(tasks, user_mean),
                "random": score(tasks, partial(random_ratings, generator=generator)),
            }
            for method in methods:
                for name in METRICS:
                    results[method][state][name].append(scores[method][name])

    for method in methods:
        print(f"\n{method:<16}" + "".join(f"{name:>19}" for name in METRICS))
        for state in TEST_STATES:
            values = results[method][state].values()
            row = "".join(f"{np.mean(v):>10.4f} ± {np.std(v):.4f}" for v in values)
            print(f"{state:<16}{row}")


if __name__ == "__main__":
    main()

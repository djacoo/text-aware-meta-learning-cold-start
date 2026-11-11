import argparse
from functools import partial

import numpy as np
import torch

from coldstart.metrics import mae, ndcg_at_k
from coldstart.model import adapt, build_model, predict
from coldstart.preprocess import load_movielens, load_tasks
from coldstart.prototype import PROTO_TARGET_PARAMS, ProtoProjector, UserTextPrototype

TEST_STATES = ("warm_test", "user_cold", "item_cold", "user_item_cold")
METRICS = ("mae", "ndcg@1", "ndcg@3")


def task_scores(tasks, predict_query):
    scores = {name: [] for name in METRICS}
    for task in tasks:
        pred = predict_query(task)
        scores["mae"].append(mae(pred, task.query_ratings))
        scores["ndcg@1"].append(ndcg_at_k(pred, task.query_ratings, 1))
        scores["ndcg@3"].append(ndcg_at_k(pred, task.query_ratings, 3))
    return scores


def score(tasks, predict_query):
    return {name: float(np.mean(v)) for name, v in task_scores(tasks, predict_query).items()}


def melu_predictor(model, steps, lr, proto_net=None, proto_projector=None, data=None):
    device = next(model.parameters()).device
    model.eval()

    if proto_net is None:
        proto_net = getattr(model, "proto_net", None)
    if proto_projector is None:
        proto_projector = getattr(model, "proto_projector", None)

    def melu(task):
        params = adapt(
            model,
            task,
            steps,
            lr,
            first_order=True,
            proto_net=proto_net,
            proto_projector=proto_projector,
            data=data,
        )
        items = task.query_items.to(device)
        with torch.no_grad():
            return predict(model, params, torch.full_like(items, task.user), items).cpu()

    return melu


def evaluate(model, tasks, steps, lr, proto_net=None, proto_projector=None, data=None):
    return score(tasks, melu_predictor(model, steps, lr, proto_net, proto_projector, data))


def user_mean(task):
    return task.support_ratings.mean().expand(len(task.query_items))


def random_ratings(task, generator):
    return 1 + 4 * torch.rand(len(task.query_items), generator=generator)


def text_knn(task, embeddings):
    weights = (embeddings[task.query_items] @ embeddings[task.support_items].T).clamp(min=0)
    total = weights.sum(dim=1)
    pred = weights @ task.support_ratings / total.clamp(min=1e-8)
    return torch.where(total > 0, pred, task.support_ratings.mean())


def load_model(path, data, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    model = build_model(data, cfg).to(device)
    model.load_state_dict(checkpoint["state"])

    if cfg.get("proto", False) and "proto_net" in checkpoint:
        text_dim = data.text_embeddings.shape[1]
        proto_dim = cfg.get("proto_dim", 32)
        proto_net = UserTextPrototype(text_dim=text_dim, proto_dim=proto_dim).to(device)
        proto_projector = ProtoProjector(
            proto_dim=proto_dim, model=model, local_param_names=PROTO_TARGET_PARAMS
        ).to(device)

        proto_net.load_state_dict(checkpoint["proto_net"])
        proto_projector.load_state_dict(checkpoint["proto_projector"])

        model.proto_net = proto_net
        model.proto_projector = proto_projector
    else:
        model.proto_net = None
        model.proto_projector = None

    return model, cfg, checkpoint["seed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one checkpoint per seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = torch.load(args.checkpoints[0], map_location="cpu")["config"]
    data = load_movielens(cfg["data_root"], cfg.get("text_encoder"), cfg.get("text_source", "plot"))

    methods = ("melu", "user mean", "random")
    if data.text_embeddings is not None:
        methods += ("text knn",)
    results = {m: {s: {name: [] for name in METRICS} for s in TEST_STATES} for m in methods}

    for path in args.checkpoints:
        model, cfg, seed = load_model(path, data, device=device)
        generator = torch.Generator().manual_seed(seed)
        for state in TEST_STATES:
            tasks = load_tasks(data, state, seed)
            scores = {
                "melu": evaluate(
                    model,
                    tasks,
                    cfg["inner_steps"],
                    cfg["inner_lr"],
                    data=data,
                ),
                "user mean": score(tasks, user_mean),
                "random": score(tasks, partial(random_ratings, generator=generator)),
            }
            if data.text_embeddings is not None:
                scores["text knn"] = score(
                    tasks, partial(text_knn, embeddings=data.text_embeddings)
                )
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

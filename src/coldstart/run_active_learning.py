import argparse
import json
import random

import numpy as np
import torch
from scipy.stats import wilcoxon

from coldstart.active_learning import (
    rating_frequencies,
    select_expected_model_change,
    select_hybrid,
    select_random,
    select_text_diversity,
    select_uncertainty,
)
from coldstart.dataset import Task
from coldstart.evaluate import TEST_STATES, expand, load_model, melu_predictor, task_scores
from coldstart.preprocess import load_data, load_tasks
from coldstart.text import DEFAULT_ENCODER


def choose_support(task, k_items, strategy, options):
    # the k items come from the support set, the query stays the same for every strategy
    if len(task.support_items) <= k_items:
        idxs = torch.arange(len(task.support_items))
    else:
        idxs = strategy(task.support_items, k_items, user=task.user, **options)
    return Task(
        user=task.user,
        support_items=task.support_items[idxs],
        support_ratings=task.support_ratings[idxs],
        query_items=task.query_items,
        query_ratings=task.query_ratings,
    )


def evaluate_active_learning(model, tasks, steps, lr, k_items, seed, strategy, options):
    # same seed for every strategy, so random choices can be repeated
    random.seed(seed)
    torch.manual_seed(seed)
    al_tasks = [choose_support(task, k_items, strategy, options) for task in tasks]
    return task_scores(al_tasks, melu_predictor(model, steps, lr))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one checkpoint per seed")
    parser.add_argument("--states", nargs="+", default=TEST_STATES)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--json", help="also save the numbers to this file")
    args = parser.parse_args(argv)
    args.checkpoints = expand(args.checkpoints)

    cfg = torch.load(args.checkpoints[0], map_location="cpu")["config"]
    # text diversity needs item texts also when the model does not use them
    encoder = cfg.get("text_encoder") or DEFAULT_ENCODER
    data = load_data(cfg["data_root"], encoder, cfg.get("text_source", "plot"))
    runs = [load_model(path, data) for path in args.checkpoints]
    ensemble = [model for model, _, _ in runs]
    rating_freqs = rating_frequencies(load_tasks(data, "warm_train", runs[0][2]))

    rows = {}
    for state in args.states:
        print(f"\n{state:<16}{'k':>3}{'mae':>8}{'ndcg@3':>8}{'p':>10}{'choose':>8}")
        for k_items in args.k:
            # per user scores of each strategy, averaged over seeds
            scores = {}
            for model, cfg, seed in runs:
                change = {"model": model, "rating_freqs": rating_freqs}
                strategies = {
                    "random": (select_random, {}),
                    "uncertainty": (select_uncertainty, {"ensemble": ensemble}),
                    "model change": (select_expected_model_change, change),
                    "hybrid": (select_hybrid, change),
                    "text diversity": (
                        select_text_diversity,
                        {"text_embeddings": data.text_embeddings},
                    ),
                }
                tasks = load_tasks(data, state, seed)
                for name, (strategy, options) in strategies.items():
                    result = evaluate_active_learning(
                        model,
                        tasks,
                        cfg["inner_steps"],
                        cfg["inner_lr"],
                        k_items,
                        seed,
                        strategy,
                        options,
                    )
                    scores.setdefault(name, []).append(result)
            # share of users with more than k support items, the others give the same answer
            choose = np.mean([len(task.support_items) > k_items for task in tasks])
            mae = {
                name: np.mean([r["mae"] for r in results], axis=0)
                for name, results in scores.items()
            }
            for name, results in scores.items():
                ndcg = np.mean([r["ndcg@3"] for r in results])
                # two-sided Wilcoxon against random, paired over users
                p = wilcoxon(mae[name], mae["random"]).pvalue if name != "random" else float("nan")
                print(
                    f"{name:<16}{k_items:>3}{mae[name].mean():>8.4f}{ndcg:>8.4f}{p:>10.2g}{choose:>8.2f}"
                )
                row = {"mae": mae[name].mean(), "ndcg@3": ndcg, "p": p, "choose": choose}
                rows.setdefault(state, {}).setdefault(str(k_items), {})[name] = row
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

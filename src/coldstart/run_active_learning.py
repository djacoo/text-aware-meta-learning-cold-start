import argparse
import random

import numpy as np
import torch
from scipy.stats import wilcoxon

from coldstart.active_learning import (
    compute_rating_frequencies,
    select_ensemble_uncertainty,
    select_expected_model_change,
    select_random,
    select_text_diversity,
)
from coldstart.dataset import Task
from coldstart.evaluate import TEST_STATES, expand, load_model, melu_predictor, task_scores
from coldstart.preprocess import load_movielens, load_tasks
from coldstart.text import DEFAULT_ENCODER


def choose_support(task, k_items, strategy_fn, **strategy_kwargs):
    # the k items come from the support set, the query stays the same for every strategy
    if task.support_items.size(0) <= k_items:
        idxs = torch.arange(task.support_items.size(0))
    else:
        idxs = strategy_fn(task.support_items, k_items, user_id=task.user, **strategy_kwargs)
    return Task(
        user=task.user,
        support_items=task.support_items[idxs],
        support_ratings=task.support_ratings[idxs],
        query_items=task.query_items,
        query_ratings=task.query_ratings,
    )


def evaluate_active_learning(model, tasks, steps, lr, k_items, seed, data, strategy_fn, **kwargs):
    # same seed for every strategy, so random choices can be repeated
    random.seed(seed)
    torch.manual_seed(seed)
    al_tasks = [choose_support(task, k_items, strategy_fn, data=data, **kwargs) for task in tasks]
    # a proto model adapts with its user prototype
    return task_scores(al_tasks, melu_predictor(model, steps, lr, data=data))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one checkpoint per seed")
    parser.add_argument("--states", nargs="+", default=TEST_STATES)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args(argv)
    args.checkpoints = expand(args.checkpoints)

    cfg = torch.load(args.checkpoints[0], map_location="cpu")["config"]
    # text diversity needs item texts also when the model does not use them
    encoder = cfg.get("text_encoder") or DEFAULT_ENCODER
    data = load_movielens(cfg["data_root"], encoder, cfg.get("text_source", "plot"))
    runs = [load_model(path, data) for path in args.checkpoints]
    ensemble_models = [model for model, _, _ in runs]
    rating_freqs = compute_rating_frequencies(load_tasks(data, "warm_train", runs[0][2]))

    for state in args.states:
        print(f"\n{state:<16}{'k':>3}{'mae':>8}{'ndcg@3':>8}{'p':>10}{'choose':>8}")
        for k_items in args.k:
            # per user scores of each strategy, averaged over seeds
            scores = {}
            for model, cfg, seed in runs:
                strategies = {
                    "random": (select_random, {}),
                    "uncertainty": (
                        select_ensemble_uncertainty,
                        {"ensemble_models": ensemble_models},
                    ),
                    "model change": (
                        select_expected_model_change,
                        {"melu_model": model, "rating_freqs": rating_freqs},
                    ),
                    "text diversity": (
                        select_text_diversity,
                        {"text_embeddings": data.text_embeddings},
                    ),
                }
                tasks = load_tasks(data, state, seed)
                for name, (strategy_fn, kwargs) in strategies.items():
                    result = evaluate_active_learning(
                        model,
                        tasks,
                        cfg["inner_steps"],
                        cfg["inner_lr"],
                        k_items,
                        seed,
                        data,
                        strategy_fn,
                        **kwargs,
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


if __name__ == "__main__":
    main()

import argparse

import numpy as np
import torch
from scipy.stats import wilcoxon

from coldstart.evaluate import TEST_STATES, load_model, melu_predictor, task_scores, user_mean
from coldstart.preprocess import load_movielens, load_tasks


def user_errors(paths):
    # MAE of each test user per state, averaged over seeds (the users are the same in every seed)
    cfg = torch.load(paths[0], map_location="cpu")["config"]
    data = load_movielens(cfg["data_root"], cfg.get("text_encoder"), cfg.get("text_source", "plot"))
    melu = {state: [] for state in TEST_STATES}
    mean = {state: [] for state in TEST_STATES}
    for path in paths:
        model, cfg, seed = load_model(path, data)
        predictor = melu_predictor(model, cfg["inner_steps"], cfg["inner_lr"])
        for state in TEST_STATES:
            tasks = load_tasks(data, state, seed)
            melu[state].append(task_scores(tasks, predictor)["mae"])
            mean[state].append(task_scores(tasks, user_mean)["mae"])
    return {state: (np.mean(melu[state], axis=0), np.mean(mean[state], axis=0)) for state in melu}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one checkpoint per seed")
    parser.add_argument("--against", nargs="+", required=True, help="checkpoints to compare with")
    args = parser.parse_args(argv)

    errors, against = user_errors(args.checkpoints), user_errors(args.against)
    print(f"{'':<16}{'model':>8}{'against':>9}{'p':>10}{'user mean':>11}{'p':>10}")
    for state in TEST_STATES:
        model, mean = errors[state]
        other = against[state][0]
        # two-sided Wilcoxon, paired over users
        p_other = wilcoxon(model, other).pvalue
        p_mean = wilcoxon(model, mean).pvalue
        print(
            f"{state:<16}{model.mean():>8.4f}{other.mean():>9.4f}{p_other:>10.2g}"
            f"{mean.mean():>11.4f}{p_mean:>10.2g}"
        )


if __name__ == "__main__":
    main()

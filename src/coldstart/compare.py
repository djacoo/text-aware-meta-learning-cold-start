import argparse

import numpy as np
import torch
from scipy.stats import wilcoxon

from coldstart.evaluate import TEST_STATES, load_model, melu_predictor, task_scores, user_mean
from coldstart.preprocess import load_movielens, load_tasks


def user_errors(paths, state):
    # MAE of each test user, averaged over seeds (the users are the same in every seed)
    cfg = torch.load(paths[0], map_location="cpu")["config"]
    data = load_movielens(cfg["data_root"], cfg.get("text_encoder"), cfg.get("text_source", "plot"))
    melu, mean = [], []
    for path in paths:
        model, cfg, seed = load_model(path, data)
        tasks = load_tasks(data, state, seed)
        predictor = melu_predictor(model, cfg["inner_steps"], cfg["inner_lr"])
        melu.append(task_scores(tasks, predictor)["mae"])
        mean.append(task_scores(tasks, user_mean)["mae"])
    return np.mean(melu, axis=0), np.mean(mean, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="one checkpoint per seed")
    parser.add_argument("--against", nargs="+", required=True, help="checkpoints to compare with")
    args = parser.parse_args()

    print(f"{'':<16}{'model':>8}{'against':>9}{'p':>10}{'user mean':>11}{'p':>10}")
    for state in TEST_STATES:
        model, mean = user_errors(args.checkpoints, state)
        other, _ = user_errors(args.against, state)
        # paired over users: is the model better than the other run and than the user mean
        p_other = wilcoxon(model, other).pvalue
        p_mean = wilcoxon(model, mean).pvalue
        print(
            f"{state:<16}{model.mean():>8.4f}{other.mean():>9.4f}{p_other:>10.2g}"
            f"{mean.mean():>11.4f}{p_mean:>10.2g}"
        )


if __name__ == "__main__":
    main()

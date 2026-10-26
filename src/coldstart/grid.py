import argparse
import contextlib
import itertools
import os
from multiprocessing import Pool
from pathlib import Path

import torch
import yaml

from coldstart.train import train

# Selection uses warm_val only, the test states are never touched here
GRID = {
    "inner_lr": [1.0e-4, 1.0e-3, 1.0e-2],
    "outer_lr": [1.0e-4, 5.0e-4, 1.0e-3],
    "inner_steps": [1, 5],
}


def run(cfg):
    torch.set_num_threads(2)
    name = "_".join(f"{key}{cfg[key]:g}" for key in GRID)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        best = train(cfg, seed=0, out_dir=Path("checkpoints/grid") / name)
    return best, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/melu.yaml")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    configs = [
        base | dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())
    ]
    with Pool(args.jobs) as pool:
        results = pool.map(run, configs)

    for best, cfg in sorted(results, key=lambda result: result[0]):
        print(f"val mae {best:.4f}  " + "  ".join(f"{key} {cfg[key]:g}" for key in GRID))


if __name__ == "__main__":
    main()

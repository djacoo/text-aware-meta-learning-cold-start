import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from coldstart.evaluate import evaluate
from coldstart.model import MeLU, meta_step
from coldstart.preprocess import load_movielens, load_tasks


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg, seed, out_dir):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_movielens(cfg["data_root"])
    train_tasks = load_tasks(data, "warm_train", seed)
    val_tasks = load_tasks(data, "warm_val", seed)

    model = MeLU(data, cfg["emb_dim"], cfg["hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["outer_lr"])
    steps, lr, batch = cfg["inner_steps"], cfg["inner_lr"], cfg["meta_batch"]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"melu_seed{seed}.pt"
    best = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        random.shuffle(train_tasks)
        losses = [
            meta_step(model, train_tasks[i : i + batch], optimizer, steps, lr)
            for i in range(0, len(train_tasks), batch)
        ]
        val_mae = evaluate(model, val_tasks, steps, lr)["mae"]
        print(f"epoch {epoch:>3}  loss {np.mean(losses):.4f}  val mae {val_mae:.4f}")

        # keep the epoch with the lowest validation MAE
        if val_mae < best:
            best = val_mae
            torch.save({"config": cfg, "seed": seed, "state": model.state_dict()}, path)
    print(f"best val mae {best:.4f}, saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/melu.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.seed, Path(args.out))


if __name__ == "__main__":
    main()

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from coldstart.evaluate import predictor, score
from coldstart.model import build_model, meta_step
from coldstart.preprocess import load_data, load_tasks
from coldstart.protonet import proto_step


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg, seed, out_dir):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(cfg["data_root"], cfg.get("text_encoder"), cfg.get("text_source", "plot"))
    train_tasks = load_tasks(data, "warm_train", seed)
    val_tasks = load_tasks(data, "warm_val", seed)

    model = build_model(data, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["outer_lr"])
    protonet = cfg.get("model") == "protonet"
    name = "protonet" if protonet else "proto_melu" if cfg.get("proto", False) else "melu"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_seed{seed}.pt"
    best = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        random.shuffle(train_tasks)
        batches = [
            train_tasks[i : i + cfg["meta_batch"]]
            for i in range(0, len(train_tasks), cfg["meta_batch"])
        ]
        if protonet:
            losses = [proto_step(model, batch, optimizer) for batch in batches]
        else:
            steps, lr = cfg["inner_steps"], cfg["inner_lr"]
            losses = [meta_step(model, batch, optimizer, steps, lr) for batch in batches]
        val_mae = score(val_tasks, predictor(model, cfg))["mae"]
        print(f"epoch {epoch:>3}  loss {np.nanmean(losses):.4f}  val mae {val_mae:.4f}")

        # keep the epoch with the lowest validation MAE
        if val_mae < best:
            best = val_mae
            torch.save({"config": cfg, "seed": seed, "state": model.state_dict()}, path)

    print(f"best val mae {best:.4f}, saved to {path}")
    return best


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/melu.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="change config values"
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    train(cfg, args.seed, Path(args.out))


if __name__ == "__main__":
    main()

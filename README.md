# Text-Aware Meta-Learning for Cold-Start Recommendation

APAI project, University of Verona, 2026/27.

MeLU (Lee et al., KDD 2019) extended with item text: text encoders, text-based user initialization, active learning for new users. Data: MovieLens-1M, Amazon Reviews 2023 (Movies & TV).

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Run

One run per seed (we use 0 to 4), with the learning rates of the paper or the ones tuned by `python -m coldstart.grid` on warm validation users:

```bash
python -m coldstart.train --seed 0 --out checkpoints/paper
python -m coldstart.train --config configs/melu_tuned.yaml --seed 0 --out checkpoints/tuned
python -m coldstart.evaluate checkpoints/tuned/melu_seed*.pt
```

Text runs encode the item texts once with a sentence encoder (BGE in our configs) and keep them in `data/ml-1m/`. The full model uses the whole description of each movie and hides directors and actors at random during training:

```bash
python -m coldstart.train --config configs/text_full.yaml --seed 0 --out checkpoints/text_full
python -m coldstart.evaluate checkpoints/text_full/melu_seed*.pt
python -m coldstart.compare checkpoints/text_full/melu_seed*.pt --against checkpoints/tuned/melu_seed*.pt
```

`compare` runs a paired Wilcoxon test over test users. Any config value can be changed with `--set`, for example `--set text_encoder=random` for the control with random vectors, or `--set text_encoder=Qwen/Qwen3-Embedding-0.6B` for another encoder.

The text-based user initialization averages the texts of the movies a user liked and disliked in the support set, and from them shifts the output layer before adaptation, an initialization inspired by Proto-MAML. It starts from zero, so at the beginning of training the model is plain MeLU:

```bash
python -m coldstart.train --config configs/proto_melu.yaml --seed 0 --out checkpoints/proto
python -m coldstart.compare checkpoints/proto/proto_melu_seed*.pt --against checkpoints/tuned/melu_seed*.pt
```

`--set proto=true proto_dim=32` adds it to any text config. Prototypical Networks are the baseline without gradient steps: each rating has a prototype from the support set and the prediction is the expected rating (`configs/protonet.yaml`).

Active learning chooses the k ratings a new user is asked for. Every strategy picks from the support set of the user and is scored on the same query items. The checkpoints of all seeds also form the committee for the uncertainty strategy:

```bash
python -m coldstart.run_active_learning checkpoints/tuned/melu_seed*.pt
```

It prints MAE and nDCG@3 for each state and k, with a paired Wilcoxon test against random choices.

The second dataset is Movies & TV from Amazon Reviews 2023. The first run on `data/amazon/` downloads the ratings and the item metadata (about 1.7 GB) and builds the four states as MeLU does: items need a title and a description, editions of the same film are reduced to the most rated one, the 5000 most rated films are kept, the last 40% to appear are the new items and a random 20% of the users are the new users. Amazon has no user profile, so users are only modelled by adaptation:

```bash
python -m coldstart.amazon
python -m coldstart.train --config configs/amazon_tuned.yaml --seed 0 --out checkpoints/amazon/tuned
```

Other configs train on Amazon with `--set data_root=data/amazon`; evaluation reads the dataset from the checkpoint.

`evaluate`, `compare` and `run_active_learning` also save their numbers with `--json`, and `coldstart.report` turns these files into the tables and figures of the paper:

```bash
python -m coldstart.evaluate checkpoints/tuned/melu_seed*.pt --json tuned.json
python -m coldstart.report table table.tex "MeLU=tuned.json" "User mean=tuned.json:user mean"
python -m coldstart.run_active_learning checkpoints/tuned/melu_seed*.pt --json al.json
python -m coldstart.report curves al.json al.pdf
```

Data is downloaded to `data/ml-1m/` on the first run, from the [MeLU repository](https://github.com/hoyeoplee/MeLU), which extends MovieLens-1M with plots, directors, actors and the cold-start splits. The model is our reimplementation of MeLU, not the original code. The Amazon files come from the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) release by McAuley Lab.

## Authors

Jacopo Parretti, Stefano Giampaoli, Alessio Brighenti

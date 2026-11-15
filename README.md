# Text-Aware Meta-Learning for Cold-Start Recommendation

APAI project, University of Verona, 2026/27.

MeLU (Lee et al., KDD 2019) extended with item text: text encoders, text-based user initialization, active learning for new users. Data: MovieLens-1M, Amazon Reviews 2023 (Movies & TV).

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Run

One run per seed (we use 0, 1 and 2), with the learning rates of the paper or the ones tuned by `python -m coldstart.grid` on warm validation users:

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

Active learning chooses the k ratings a new user is asked for. Every strategy picks from the support set of the user and is scored on the same query items. The checkpoints of all seeds also form the committee for the uncertainty strategy:

```bash
python -m coldstart.run_active_learning checkpoints/tuned/melu_seed*.pt
```

It prints MAE and nDCG@3 for each state and k, with a paired Wilcoxon test against random choices.

Data is downloaded to `data/ml-1m/` on the first run, from the [MeLU repository](https://github.com/hoyeoplee/MeLU), which extends MovieLens-1M with plots, directors, actors and the cold-start splits. The model is our reimplementation of MeLU, not the original code.

## Authors

Jacopo Parretti, Stefano Giampaoli, Alessio Brighenti

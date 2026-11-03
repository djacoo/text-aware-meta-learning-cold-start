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

Data is downloaded to `data/ml-1m/` on the first run, from the [MeLU repository](https://github.com/hoyeoplee/MeLU), which extends MovieLens-1M with plots, directors, actors and the cold-start splits. The model is our reimplementation of MeLU, not the original code.

## Authors

Jacopo Parretti, Stefano Giampaoli, Alessio Brighenti

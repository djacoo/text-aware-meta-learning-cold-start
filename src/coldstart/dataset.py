from dataclasses import dataclass
from pathlib import Path

from torch import Tensor


@dataclass
class Task:
    user: int
    support_items: Tensor
    support_ratings: Tensor
    query_items: Tensor
    query_ratings: Tensor


@dataclass
class MovieLens:
    root: Path
    item_features: dict[str, Tensor]
    user_features: dict[str, Tensor]
    vocab_sizes: dict[str, int]
    item_index: dict[int, int]
    user_index: dict[int, int]
    plots: list[str]

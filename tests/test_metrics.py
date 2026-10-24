import math

import pytest
import torch

from coldstart.metrics import mae, ndcg_at_k

target = torch.tensor([5.0, 3.0, 1.0])


def test_mae():
    pred = torch.tensor([4.0, 3.0, 3.0])
    assert mae(pred, target) == pytest.approx(1.0)


def test_ndcg_perfect_ranking():
    pred = torch.tensor([4.8, 3.1, 0.9])
    assert ndcg_at_k(pred, target, 3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking():
    pred = torch.tensor([1.0, 2.0, 3.0])
    # gains are 31, 7, 1 and the model ranks them 1, 7, 31
    dcg = 1 + 7 / math.log2(3) + 31 / 2
    idcg = 31 + 7 / math.log2(3) + 1 / 2
    assert ndcg_at_k(pred, target, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(pred, target, 1) == pytest.approx(1 / 31)

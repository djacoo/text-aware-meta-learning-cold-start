import torch


def mae(pred, target):
    return (pred - target).abs().mean().item()


def ndcg_at_k(pred, target, k):
    # Query items ranked by predicted rating, gain 2^r - 1 as in MeLU
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float))
    gains = 2.0**target - 1
    ranked = gains[pred.sort(descending=True, stable=True).indices[:k]]
    ideal = gains.sort(descending=True).values[:k]
    return ((ranked / discounts).sum() / (ideal / discounts).sum()).item()

import random

import torch
import torch.nn.functional as F

from coldstart.model import predict


def get_candidate_pool(task):
    """
    Retrieves every user's known items to build the pool where the Active Learning can pick from.
    """
    all_items = torch.cat([task.support_items, task.query_items], dim=0)
    all_ratings = torch.cat([task.support_ratings, task.query_ratings], dim=0)
    return all_items, all_ratings


def select_random(all_items, k, **kwargs):
    """
    Baseline strategy: selects k items completely at random.
    It serves as the lower-bound baseline to prove that Active Learning actually provides a
    statistically significant advantage.
    """
    n = all_items.size(0)
    if k >= n:
        return torch.arange(n)
    return torch.randperm(n)[:k]


def select_uncertainty(all_items, k, melu_model, user_id, **kwargs):
    """
    Selects items where the model is most uncertain.
    For a 1-5 rating scale, maximum uncertainty is assumed around the middle score (3.0).
    """
    n = all_items.size(0)
    if k >= n:
        return torch.arange(n)

    device = next(melu_model.parameters()).device
    items = all_items.to(device)
    users = torch.full_like(items, user_id).to(device)

    params = dict(melu_model.named_parameters())

    with torch.no_grad():
        melu_model.eval()
        predictions = predict(melu_model, params, users, items).view(-1)
        # Distance from neutral value
        uncertainty = torch.abs(predictions - 3.0)

        # The first numbers have a lower distance from 3.0
        top_k_indices = torch.argsort(uncertainty)[:k]

    return top_k_indices.cpu()


def select_text_diversity(all_items, k, text_embeddings, **kwargs):
    """
    Selects k items maximizing semantic diversity using a Greedy Max-Min distance approach.
    """
    n = all_items.size(0)
    if k >= n:
        return torch.arange(n)

    if text_embeddings is None:
        raise ValueError("Text embeddings are missing. Load MovieLens with text_encoder!")

    candidates_emb = text_embeddings[all_items]

    selected_idx = [random.randint(0, n - 1)]
    unselected_idx = list(range(n))
    unselected_idx.remove(selected_idx[0])

    for _ in range(1, k):
        max_min_dist = -1.0
        best_candidate = -1
        sel_tensor = candidates_emb[selected_idx]

        for cand in unselected_idx:
            cand_tensor = candidates_emb[cand].unsqueeze(0)

            # Cosine distance
            sims = F.cosine_similarity(cand_tensor, sel_tensor)
            min_dist = torch.min(1.0 - sims).item()

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = cand

        selected_idx.append(best_candidate)
        unselected_idx.remove(best_candidate)

    return torch.tensor(selected_idx)

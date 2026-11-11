import random

import torch
import torch.nn.functional as F

from coldstart.model import predict

LOCAL_PARAMS = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "out.weight", "out.bias")


def compute_rating_frequencies(tasks):
    """
    Computes the exact normalized distribution of ratings from a list of Tasks.
    """
    all_ratings = []
    for task in tasks:
        all_ratings.append(task.support_ratings)
        all_ratings.append(task.query_ratings)

    all_ratings = torch.cat(all_ratings)
    unique_vals, counts = torch.unique(all_ratings, return_counts=True)

    total = all_ratings.size(0)
    # Map each unique rating to its frequency
    return {u.item(): (c.item() / total * 100) for u, c in zip(unique_vals, counts, strict=False)}


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


def select_ensemble_uncertainty(c_items, k, ensemble_models, user_id, **kwargs):
    """
    Selects k items where the variance among 5 trained models is the highest.
    """
    n = c_items.size(0)
    if k >= n:
        return torch.arange(n)

    device = next(ensemble_models[0].parameters()).device
    items_t = c_items.to(device)
    users_t = torch.full_like(items_t, user_id).to(device)

    preds = []
    with torch.no_grad():
        for model in ensemble_models:
            model.eval()
            params = dict(model.named_parameters())
            preds.append(predict(model, params, users_t, items_t))

    preds_tensor = torch.stack(preds)
    variance = torch.var(preds_tensor, dim=0)

    top_k_indices = torch.argsort(variance, descending=True)[:k]
    return top_k_indices.cpu()


def select_expected_model_change(c_items, k, melu_model, user_id, **kwargs):
    """
    Calculates the Expected Gradient Length (EGL) for each candidate.
    """
    n = c_items.size(0)
    if k >= n:
        return torch.arange(n)

    # Uses the dynamically computed frequencies
    rating_freqs = kwargs.get("rating_freqs")
    if rating_freqs is None:
        raise ValueError("Expected Model Change requires 'rating_freqs' kwargs.")

    device = next(melu_model.parameters()).device
    params = dict(melu_model.named_parameters())
    local_params = [params[name] for name in LOCAL_PARAMS]

    emc_scores = []

    for i in range(n):
        item = c_items[i : i + 1].to(device)
        user = torch.full((1,), user_id, dtype=torch.long, device=device)

        pred = predict(melu_model, params, user, item)
        grads = torch.autograd.grad(pred, local_params, retain_graph=False)
        grad_norm = torch.sqrt(sum(torch.norm(g) ** 2 for g in grads)).item()

        expected_norm = 0.0
        # Iterate dynamically over the provided training frequencies
        for y_val, prob in rating_freqs.items():
            scaling_factor = 2.0 * abs(pred.item() - y_val)
            expected_norm += prob * (scaling_factor * grad_norm)

        emc_scores.append(expected_norm)

    emc_tensor = torch.tensor(emc_scores)
    top_k_indices = torch.argsort(emc_tensor, descending=True)[:k]
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

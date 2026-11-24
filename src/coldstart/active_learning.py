import random

import torch
import torch.nn.functional as F

from coldstart.model import predict, start_params


def rating_frequencies(tasks):
    # how often each rating value is given, used to weight the unknown rating of a candidate
    ratings = torch.cat([torch.cat([t.support_ratings, t.query_ratings]) for t in tasks])
    values, counts = torch.unique(ratings, return_counts=True)
    return {v.item(): c.item() / len(ratings) for v, c in zip(values, counts, strict=True)}


def select_random(items, k, **kwargs):
    return torch.randperm(len(items))[:k]


def select_uncertainty(items, k, ensemble, user, **kwargs):
    # disagreement between the models trained with different seeds, before any rating
    users = torch.full_like(items, user)
    with torch.no_grad():
        preds = torch.stack([predict(m, start_params(m), users, items) for m in ensemble])
    return torch.argsort(torch.var(preds, dim=0), descending=True)[:k]


def select_expected_model_change(items, k, model, user, rating_freqs, **kwargs):
    # expected gradient length: for a rating y the squared error has gradient 2 |f - y| grad f,
    # averaged over how often each rating is given
    params = start_params(model)
    local = [params[name] for name in model.local_params]
    scores = []
    for item in items:
        pred = predict(model, params, torch.tensor([user]), item[None])
        grads = torch.autograd.grad(pred, local)
        norm = torch.sqrt(sum(torch.norm(g) ** 2 for g in grads)).item()
        scores.append(sum(p * (2.0 * abs(pred.item() - y) * norm) for y, p in rating_freqs.items()))
    return torch.argsort(torch.tensor(scores), descending=True)[:k]


def select_hybrid(items, k, **kwargs):
    # expected model change for the first item, the others at random
    first = select_expected_model_change(items, 1, **kwargs).item()
    rest = [i for i in torch.randperm(len(items)).tolist() if i != first][: k - 1]
    return torch.tensor([first, *rest])


def select_text_diversity(items, k, text_embeddings, **kwargs):
    # greedy max-min cosine distance, starting from a random item
    vectors = text_embeddings[items]
    has_text = vectors.norm(dim=1) > 0
    selected = [random.randint(0, len(items) - 1)]
    remaining = [i for i in range(len(items)) if i != selected[0]]
    for _ in range(1, k):
        best, best_distance = -1, -1.0
        for i in remaining:
            sims = F.cosine_similarity(vectors[i][None], vectors[selected])
            # a film without text has a zero vector, which would look farthest from everything
            distance = torch.min(1.0 - sims).item() if has_text[i] else 0.0
            if distance > best_distance:
                best, best_distance = i, distance
        selected.append(best)
        remaining.remove(best)
    return torch.tensor(selected)

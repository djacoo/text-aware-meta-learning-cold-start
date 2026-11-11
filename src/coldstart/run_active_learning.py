import numpy as np
import torch

from coldstart.active_learning import (
    compute_rating_frequencies,
    select_random,
    select_ensemble_uncertainty,
    select_expected_model_change,
    select_text_diversity,
)
from coldstart.dataset import Task
from coldstart.evaluate import load_model
from coldstart.metrics import mae
from coldstart.model import adapt, predict
from coldstart.preprocess import load_movielens, load_tasks


def evaluate_active_learning(model, tasks, steps, lr, k_items, strategy_fn, **strategy_kwargs):
    device = next(model.parameters()).device
    model.eval()

    mae_list, mse_list = [], []

    for task in tasks:
        user_items = task.support_items
        user_ratings = task.support_ratings

        # Fixed test set
        q_items = task.query_items
        q_ratings = task.query_ratings

        if user_items.size(0) <= k_items:
            idxs = torch.arange(user_items.size(0))
        else:
            idxs = strategy_fn(user_items, k_items, **strategy_kwargs, user_id=task.user)

        # Creates a new fake Task with k-size support
        al_task = Task(
            user=task.user,
            support_items=user_items[idxs],
            support_ratings=user_ratings[idxs],
            query_items=q_items,
            query_ratings=q_ratings,
        )

        # Adapts the model
        params = adapt(model, al_task, steps, lr, first_order=True)

        # Predicts and compute the error
        with torch.no_grad():
            q_users = torch.full_like(q_items, task.user).to(device)
            q_items = q_items.to(device)
            q_pred = predict(model, params, q_users, q_items).cpu()

        mae_list.append(mae(q_pred, q_ratings))
        mse_list.append(torch.nn.functional.mse_loss(q_pred, q_ratings).item())

    return float(np.mean(mae_list)), float(np.sqrt(np.mean(mse_list)))


def main():
    # Load dataset with textual embeddings
    data = load_movielens("data/ml-1m", text_encoder="sentence-transformers/all-MiniLM-L6-v2")

    # Move the model to the gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_models = []
    for i in range(5):
        # Load the trained models
        checkpoint_path = f"checkpoints/tuned/melu_seed{i}.pt"
        model, current_cfg, current_seed = load_model(checkpoint_path, data)
        ensemble_models.append(model.to(device))

        if i == 0:
            cfg = current_cfg
            seed = current_seed

    # Arbitrary choice for the "main" model
    model = ensemble_models[0]

    # Load the training set to compute real probabilities
    train_tasks = load_tasks(data, "warm_train", seed=seed)
    dynamic_rating_freqs = compute_rating_frequencies(train_tasks)

    # Load the test set
    tasks = load_tasks(data, "user_cold", seed=seed)

    k_values = [1, 3, 5, 10]
    inner_steps = cfg["inner_steps"]
    inner_lr = cfg["inner_lr"]

    print(f"--- Active Learning on User Cold (Seed: {seed}) ---")

    for k_items in k_values:
        print(f"\n=== Active Learning (k={k_items}) su User Cold ===")

        # Random test
        mae_rand, rmse_rand = evaluate_active_learning(
            model, 
            tasks, 
            inner_steps, 
            inner_lr, 
            k_items, 
            select_random
        )
        print(f"Random      -> MAE: {mae_rand:.4f}, RMSE: {rmse_rand:.4f}")

        # Uncertainty test
        mae_unc, rmse_unc = evaluate_active_learning(
            model, 
            tasks, 
            inner_steps, 
            inner_lr, 
            k_items, 
            select_ensemble_uncertainty, 
            ensemble_models=ensemble_models
        )
        print(f"Ensemble Uncertainty -> MAE: {mae_unc:.4f}, RMSE: {rmse_unc:.4f}")

        # Expected Model Change Test
        mae_emc, rmse_emc = evaluate_active_learning(
            model, tasks, inner_steps, inner_lr, k_items, 
            select_expected_model_change, 
            melu_model=model, 
            rating_freqs=dynamic_rating_freqs
        )
        print(f"Exp. Model Change -> MAE: {mae_emc:.4f}, RMSE: {rmse_emc:.4f}")

        # Text Diversity test
        mae_div, rmse_div = evaluate_active_learning(
            model,
            tasks,
            inner_steps,
            inner_lr,
            k_items,
            select_text_diversity,
            text_embeddings=data.text_embeddings,
        )
        print(f"Text Diversity   -> MAE: {mae_div:.4f}, RMSE: {rmse_div:.4f}")


if __name__ == "__main__":
    main()

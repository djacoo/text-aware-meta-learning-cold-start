import torch
import numpy as np

from coldstart.preprocess import load_movielens, load_tasks
from coldstart.model import MeLU, adapt, predict
from coldstart.metrics import mae
from coldstart.dataset import Task
from coldstart.active_learning import get_candidate_pool, select_random, select_uncertainty, select_text_diversity
from coldstart.evaluate import load_model

def evaluate_active_learning(model, tasks, steps, lr, k_items, strategy_fn, **strategy_kwargs):
    device = next(model.parameters()).device
    model.eval()
    
    mae_list, mse_list = [], []
    
    for task in tasks:
        all_items, all_ratings = get_candidate_pool(task)
        if all_items.size(0) <= k_items:
            continue
            
        # Selects k item through the chosen strategy
        idxs = strategy_fn(all_items, k_items, **strategy_kwargs, user_id=task.user)
        
        # Boolean mask to divided between chosen (support) and not chosen (query)
        mask = torch.zeros(all_items.size(0), dtype=torch.bool)
        mask[idxs] = True
        
        # Creates a new fake Task with k-size support
        al_task = Task(
            user=task.user,
            support_items=all_items[mask],
            support_ratings=all_ratings[mask],
            query_items=all_items[~mask],
            query_ratings=all_ratings[~mask]
        )
        
        # Adapts the model
        params = adapt(model, al_task, steps, lr, first_order=True)
        
        # Predicts and compute the error
        q_items = al_task.query_items.to(device)
        q_users = torch.full_like(q_items, task.user).to(device)
        
        with torch.no_grad():
            q_pred = predict(model, params, q_users, q_items).cpu()
            
        mae_list.append(mae(q_pred, al_task.query_ratings))
        mse_list.append(torch.nn.functional.mse_loss(q_pred, al_task.query_ratings).item())
        
    return float(np.mean(mae_list)), float(np.sqrt(np.mean(mse_list)))


def main():
    # Load dataset with textual embeddings
    data = load_movielens("data/ml-1m", text_encoder="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the trained model
    checkpoint_path = "checkpoints/tuned/melu_seed0.pt"
    model, cfg, seed = load_model(checkpoint_path, data)
    
    # Move the model to the gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load the test set
    tasks = load_tasks(data, "user_cold", seed=42)
    
    k_values = [1, 3, 5, 10]
    inner_steps = cfg["inner_steps"] 
    inner_lr = cfg["inner_lr"]

    print(f"--- Active Learning on User Cold (Seed: {seed}) ---")
    
    for k_items in k_values:
        print(f"\n--- k = {k_items} ---")

        print(f"\n=== Active Learning (k={k_items}) su User Cold ===")
        
        # Random test
        mae_rand, rmse_rand = evaluate_active_learning(
            model, tasks, inner_steps, inner_lr, k_items, select_random
        )
        print(f"Random      -> MAE: {mae_rand:.4f}, RMSE: {rmse_rand:.4f}")
        
        # Uncertainty test
        mae_unc, rmse_unc = evaluate_active_learning(
            model, tasks, inner_steps, inner_lr, k_items, select_uncertainty, melu_model=model
        )
        print(f"Uncertainty -> MAE: {mae_unc:.4f}, RMSE: {rmse_unc:.4f}")
        
        # Text Diversity test
        mae_div, rmse_div = evaluate_active_learning(
            model, tasks, inner_steps, inner_lr, k_items, select_text_diversity, text_embeddings=data.text_embeddings
        )
        print(f"Text Div.   -> MAE: {mae_div:.4f}, RMSE: {rmse_div:.4f}")

if __name__ == "__main__":
    main()
import os
import torch
import pickle

from MeLU import MeLU
from options import config
from model_training import training, evaluate
from data_generation import generate
from evidence_candidate import selection


def load_dataset_from_state(master_path, state_name):
    """Load tuples (supp_x, supp_y, query_x, query_y) from a specific folder"""
    state_dir = f"{master_path}/{state_name}"
    dataset_size = int(len(os.listdir(state_dir)) / 4)
    supp_xs_s, supp_ys_s, query_xs_s, query_ys_s = [], [], [], []
    
    for idx in range(dataset_size):
        with open(f"{state_dir}/supp_x_{idx}.pkl", "rb") as f:
            supp_xs_s.append(pickle.load(f))
        with open(f"{state_dir}/supp_y_{idx}.pkl", "rb") as f:
            supp_ys_s.append(pickle.load(f))
        with open(f"{state_dir}/query_x_{idx}.pkl", "rb") as f:
            query_xs_s.append(pickle.load(f))
        with open(f"{state_dir}/query_y_{idx}.pkl", "rb") as f:
            query_ys_s.append(pickle.load(f))
            
    return list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))


if __name__ == "__main__":
    master_path = "./ml"
    if not os.path.exists("{}/".format(master_path)):
        os.mkdir("{}/".format(master_path))
        # preparing dataset. It needs about 22GB of your hard disk space.
        dataset_path = generate(master_path)

    # Training or loading the model
    melu = MeLU(config)
    #  model_filename = "{}/models.pkl".format(master_path)
    #  if not os.path.exists(model_filename):
    #      total_dataset = load_dataset_from_state(master_path, "warm_state")
    #      training(
    #          melu, 
    #          total_dataset, 
    #          batch_size=config['batch_size'], 
    #          num_epoch=config['num_epoch'], 
    #          model_save=True, 
    #          model_filename=model_filename
    #      )
    #  else:
    #      trained_state_dict = torch.load(model_filename)
    #      melu.load_state_dict(trained_state_dict)
    #      if config['use_cuda']:
    #          melu.cuda()
#  
    #  # Evaluating on 3 cold-start scenarios
    #  test_states = ["user_cold_state", "item_cold_state", "user_and_item_cold_state"]
    #  print("\n--- Evaluation Results ---")
    #  for state in test_states:
    #      test_dataset = load_dataset_from_state(master_path, state)
    #      mae, rmse = evaluate(melu, test_dataset, num_local_update=config['inner'])
    #      print(f"[{state}] MAE: {mae:.4f} | RMSE: {rmse:.4f}")
#  
    #  print("\n")
#  
    #  # 3. Selecting evidence candidates
    #  evidence_candidate_list = selection(melu, master_path, config['num_candidate'], dataset_path)
    #  for movie, score in evidence_candidate_list:
    #      print(movie, score)
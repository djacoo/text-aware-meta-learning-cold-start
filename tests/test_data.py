import sys
from pathlib import Path

# Add the project root to sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coldstart.download import download_movielens_1m
from src.coldstart.preprocess import (
    load_and_clean_data, 
    extract_user_features, 
    extract_item_features, 
    prepare_user_tasks
)
from src.coldstart.dataset import MetaDataTaskDataset

def test_data_pipeline():
    """
    Integration test to verify raw data download, preprocessing, 
    and PyTorch MetaDataTaskDataset initialization.
    """
    print("\n--- Starting Data Pipeline Test ---")
    
    # 1. Download data
    download_movielens_1m()
    
    # 2. Preprocess features and tasks
    users, movies, ratings = load_and_clean_data()
    user_feats = extract_user_features(users)
    item_feats = extract_item_features(movies)
    user_tasks = prepare_user_tasks(ratings, k_shot=5)
    
    # 3. Instantiate Dataset
    dataset = MetaDataTaskDataset(user_tasks, user_feats, item_feats)
    
    # Assertions
    assert len(dataset) > 0, "Dataset length should be greater than 0."
    
    supp_x, supp_y, query_x, query_y = dataset[0]
    assert supp_x.shape[0] == 5, f"Support set size should be 5, got {supp_x.shape[0]}."
    assert supp_x.shape[1] == (user_feats.shape[1] + item_feats.shape[1]), "Feature dimension mismatch."
    
    print("[Test Result] All assertions passed successfully!\n")

if __name__ == "__main__":
    test_data_pipeline()
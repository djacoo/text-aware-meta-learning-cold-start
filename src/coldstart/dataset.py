import torch
from torch.utils.data import Dataset

class MetaDataTaskDataset(Dataset):
    """
    PyTorch Dataset wrapper for MeLU Meta-Learning tasks.
    Serves (Support Set, Query Set) tensor tuples per user task.
    """
    def __init__(self, user_tasks, user_features, item_features):
        """
        Args:
            user_tasks (dict): Dictionary containing support/query item IDs and ratings per user.
            user_features (DataFrame): Encoded user feature vectors indexed by user_id.
            item_features (DataFrame): Encoded item feature vectors indexed by movie_id.
        """
        self.user_ids = list(user_tasks.keys())
        self.user_tasks = user_tasks
        self.user_features = user_features
        self.item_features = item_features

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        task = self.user_tasks[user_id]
        
        # Build support set input vectors and targets
        supp_items = task['support_items']
        supp_x = self._build_input_vectors(user_id, supp_items)
        supp_y = torch.tensor(task['support_ratings'], dtype=torch.float32)

        # Build query set input vectors and targets
        query_items = task['query_items']
        query_x = self._build_input_vectors(user_id, query_items)
        query_y = torch.tensor(task['query_ratings'], dtype=torch.float32)

        return supp_x, supp_y, query_x, query_y

    def _build_input_vectors(self, user_id, item_ids):
        """
        Concatenates user feature vector with corresponding item feature vectors.
        
        Returns:
            torch.Tensor: Combined input representation tensor of shape (num_items, feature_dim).
        """
        user_vector = self.user_features.loc[user_id].values
        input_vectors = []
        
        for item_id in item_ids:
            if item_id in self.item_features.index:
                item_vector = self.item_features.loc[item_id].values
                combined = torch.cat([
                    torch.tensor(user_vector, dtype=torch.float32), 
                    torch.tensor(item_vector, dtype=torch.float32)
                ])
                input_vectors.append(combined)
            else:
                # Fallback for missing item metadata: pad with zeros
                item_dim = self.item_features.shape[1]
                combined = torch.cat([
                    torch.tensor(user_vector, dtype=torch.float32), 
                    torch.zeros(item_dim, dtype=torch.float32)
                ])
                input_vectors.append(combined)
                
        return torch.stack(input_vectors)
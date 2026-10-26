import os
import pandas as pd
import torch

def load_and_clean_data(data_dir="data/raw/ml-1m"):
    """
    Loads MovieLens-1M raw files (.dat) with proper delimiters and encodings.
    
    Returns:
        tuple: DataFrames for users, movies, and ratings.
    """
    print("[Data Preprocessing] Loading raw MovieLens-1M data files...")
    
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"),
        sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zipcode"]
    )
    
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::", engine="python", encoding="latin-1",
        names=["movie_id", "title", "genres"]
    )
    
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    
    return users, movies, ratings

def extract_user_features(users_df):
    """
    Converts categorical user metadata (gender, age, occupation) into one-hot vectors.
    """
    users_encoded = pd.get_dummies(users_df, columns=["gender", "age", "occupation"])
    users_encoded = users_encoded.drop(columns=["zipcode"])
    return users_encoded.set_index("user_id")

def extract_item_features(movies_df):
    """
    Extracts traditional metadata (movie genres) as binary features.
    
    Note: Future NLP text embeddings (e.g., plot descriptions, user reviews)
          will be concatenated to this DataFrame in Phase 2.
    """
    genres_list = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    
    # One-hot encode movie genres
    for genre in genres_list:
        movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)
    
    feature_cols = genres_list  # TODO: Append text embedding columns here when NLP features are added
    return movies_df.set_index('movie_id')[feature_cols]

def prepare_user_tasks(ratings_df, k_shot=13):
    """
    Groups user interactions into meta-learning tasks.
    Each task corresponds to a single user and is split into Support Set and Query Set.
    
    Args:
        ratings_df (DataFrame): User ratings dataset.
        k_shot (int): Number of items in the support set for meta-testing adaptation.
        
    Returns:
        dict: User task dictionary with support and query items/ratings.
    """
    print(f"[Data Preprocessing] Building meta-tasks (k_shot={k_shot})...")
    tasks = {}
    grouped = ratings_df.groupby('user_id')
    
    for user_id, group in grouped:
        # Filter out users with insufficient interaction history for meta-learning
        if len(group) < k_shot + 5:
            continue
        
        # Chronological or index-based split into Support and Query sets
        support = group.iloc[:k_shot]
        query = group.iloc[k_shot:]
        
        tasks[user_id] = {
            'support_items': support['movie_id'].values,
            'support_ratings': support['rating'].values,
            'query_items': query['movie_id'].values,
            'query_ratings': query['rating'].values,
        }
        
    print(f"[Data Preprocessing] Generated {len(tasks)} valid meta-learning user tasks.")
    return tasks
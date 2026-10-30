import json
import random
import re
from pathlib import Path

import torch

from coldstart.dataset import MovieLens, Task
from coldstart.download import download_movielens_1m
from coldstart.text import load_text_embeddings

VOCAB_FILES = {
    "rate": "m_rate.txt",
    "genre": "m_genre.txt",
    "director": "m_director.txt",
    "actors": "m_actor.txt",
    "gender": "m_gender.txt",
    "age": "m_age.txt",
    "occupation": "m_occupation.txt",
    "zip": "m_zipcode.txt",
}
STATES = ("warm_train", "warm_val", "warm_test", "user_cold", "item_cold", "user_item_cold")
STATE_FILES = {
    "user_cold": "user_cold_state",
    "item_cold": "item_cold_state",
    "user_item_cold": "user_and_item_cold_state",
}


def read_rows(path):
    return [line.split("::") for line in path.read_text(encoding="utf-8").splitlines()]


def load_vocab(path):
    # index 0 is left for padding and missing values
    lines = path.read_text(encoding="utf-8").splitlines()
    return {value.strip(): i for i, value in enumerate(lines, start=1)}


def split_values(text, field):
    if text == "N/A":
        return []
    values = text.split(", ")
    if field == "director":
        # drop notes like "(co-director)", as the MeLU vocabulary does
        values = [re.sub(r"\([^()]*\)", "", value) for value in values]
    return values


def pad(rows):
    out = torch.zeros(len(rows), max(len(row) for row in rows), dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def extract_item_features(root, vocab):
    # movie_id, title, year, rate, released, genre, director, writer, actors, plot, poster
    rows = read_rows(root / "movies_extrainfos.dat")
    features = {"rate": torch.tensor([vocab["rate"][row[3]] for row in rows])}
    for field, col in (("genre", 5), ("director", 6), ("actors", 8)):
        values = [split_values(row[col], field) for row in rows]
        features[field] = pad([[vocab[field][v] for v in row] for row in values])
    item_index = {int(row[0]): i for i, row in enumerate(rows)}
    plots = [row[9] for row in rows]
    return features, item_index, plots


def extract_user_features(root, vocab):
    # user_id, gender, age, occupation, zip
    rows = read_rows(root / "users.dat")
    columns = {"gender": 1, "age": 2, "occupation": 3, "zip": 4}
    features = {
        field: torch.tensor([vocab[field][row[col][:5]] for row in rows])
        for field, col in columns.items()
    }
    user_index = {int(row[0]): i for i, row in enumerate(rows)}
    return features, user_index


def load_movielens(root="data/ml-1m", text_encoder=None):
    root = Path(root)
    download_movielens_1m(root)
    vocab = {field: load_vocab(root / name) for field, name in VOCAB_FILES.items()}
    item_features, item_index, plots = extract_item_features(root, vocab)
    user_features, user_index = extract_user_features(root, vocab)
    vocab_sizes = {field: len(values) + 1 for field, values in vocab.items()}
    data = MovieLens(root, item_features, user_features, vocab_sizes, item_index, user_index, plots)
    if text_encoder:
        data.text_embeddings = load_text_embeddings(root, plots, text_encoder)
    return data


def prepare_user_tasks(data, state_file, seed):
    movies = json.loads((data.root / f"{state_file}.json").read_text())
    ratings = json.loads((data.root / f"{state_file}_y.json").read_text())
    rng = random.Random(seed)
    tasks = []
    for user_id, movie_ids in movies.items():
        # same filter as MeLU: 13 to 100 ratings, the last 10 after shuffling are the query
        if not 13 <= len(movie_ids) <= 100:
            continue
        order = list(range(len(movie_ids)))
        rng.shuffle(order)
        items = torch.tensor([data.item_index[int(movie_ids[i])] for i in order])
        y = torch.tensor([ratings[user_id][i] for i in order], dtype=torch.float)
        user = data.user_index[int(user_id)]
        tasks.append(Task(user, items[:-10], y[:-10], items[-10:], y[-10:]))
    return tasks


def load_tasks(data, state, seed):
    if not state.startswith("warm"):
        return prepare_user_tasks(data, STATE_FILES[state], seed)

    tasks = prepare_user_tasks(data, "warm_state", seed)
    # warm users are split 80/10/10 with a fixed seed, so every run uses the same users
    users = sorted(task.user for task in tasks)
    random.Random(0).shuffle(users)
    n_train, n_val = int(0.8 * len(users)), int(0.1 * len(users))
    parts = {
        "warm_train": users[:n_train],
        "warm_val": users[n_train : n_train + n_val],
        "warm_test": users[n_train + n_val :],
    }
    keep = set(parts[state])
    return [task for task in tasks if task.user in keep]

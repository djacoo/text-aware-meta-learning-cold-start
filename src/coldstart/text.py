import argparse
from pathlib import Path

import torch

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def encode_plots(plots, encoder, batch_size=64):
    vectors = encoder.encode(
        plots, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
    )
    vectors = vectors.cpu().float()
    # movies without a plot get a zero vector, not the embedding of the string "N/A"
    vectors[torch.tensor([plot == "N/A" for plot in plots])] = 0
    return vectors


def load_text_embeddings(root, plots, encoder_name=DEFAULT_ENCODER):
    path = Path(root) / f"plots_{encoder_name.split('/')[-1]}.pt"
    if path.exists():
        return torch.load(path)

    # imported here so runs without text don't load transformers
    from sentence_transformers import SentenceTransformer

    vectors = encode_plots(plots, SentenceTransformer(encoder_name))
    torch.save(vectors, path)
    return vectors


def main():
    from coldstart.preprocess import load_movielens

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--root", default="data/ml-1m")
    args = parser.parse_args()

    vectors = load_movielens(args.root, text_encoder=args.encoder).text_embeddings
    print(f"{len(vectors)} plots, {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()

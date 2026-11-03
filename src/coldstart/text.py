import argparse
from pathlib import Path

import torch

DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"


def missing(texts):
    return torch.tensor([text == "N/A" for text in texts])


def encode_texts(texts, encoder, batch_size=64):
    vectors = encoder.encode(
        texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True
    )
    vectors = vectors.cpu().float()
    # movies without a plot get a zero vector, not the embedding of the string "N/A"
    vectors[missing(texts)] = 0
    return vectors


def load_text_embeddings(root, texts, encoder_name=DEFAULT_ENCODER, name="plots"):
    if encoder_name == "random":
        # control run: unit vectors with no meaning, same size as BGE, zero where text is missing
        vectors = torch.randn(len(texts), 768, generator=torch.Generator().manual_seed(0))
        vectors = torch.nn.functional.normalize(vectors, dim=1)
        vectors[missing(texts)] = 0
        return vectors

    path = Path(root) / f"{name}_{encoder_name.split('/')[-1]}.pt"
    if path.exists():
        return torch.load(path)

    # imported here so runs without text don't load transformers
    from sentence_transformers import SentenceTransformer

    vectors = encode_texts(texts, SentenceTransformer(encoder_name))
    torch.save(vectors, path)
    return vectors


def main():
    from coldstart.preprocess import load_movielens

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--root", default="data/ml-1m")
    parser.add_argument("--source", default="plot", choices=["plot", "full"])
    args = parser.parse_args()

    vectors = load_movielens(args.root, args.encoder, args.source).text_embeddings
    print(f"{len(vectors)} texts, {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()

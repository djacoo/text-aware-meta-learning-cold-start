from pathlib import Path

import torch

DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"


def missing(texts):
    return torch.tensor([text == "N/A" for text in texts])


def encode_texts(texts, encoder):
    vectors = encoder.encode(
        texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True
    )
    # encode returns an inference tensor, which cannot be changed in place
    vectors = vectors.cpu().float().clone()
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
        vectors = torch.load(path)
        if len(vectors) != len(texts):
            raise ValueError(f"{path} was built for other items, delete it to encode again")
        return vectors

    # imported here so runs without text don't load transformers
    from sentence_transformers import SentenceTransformer

    vectors = encode_texts(texts, SentenceTransformer(encoder_name))
    torch.save(vectors, path)
    return vectors

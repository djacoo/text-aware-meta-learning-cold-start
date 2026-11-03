import torch

from coldstart.preprocess import load_movielens
from coldstart.text import encode_plots, load_text_embeddings


class FakeEncoder:
    def encode(self, plots, batch_size, convert_to_tensor, normalize_embeddings):
        return torch.nn.functional.normalize(torch.rand(len(plots), 4), dim=1)


def test_missing_plot_is_zero():
    vectors = encode_plots(["A plot.", "N/A", "Another plot."], FakeEncoder())
    assert vectors.shape == (3, 4)
    assert torch.equal(vectors[1], torch.zeros(4))
    assert torch.allclose(vectors[[0, 2]].norm(dim=1), torch.ones(2))


def test_cached_embeddings_are_loaded(ml_root):
    cached = torch.rand(3, 4)
    torch.save(cached, ml_root / "plots_fake-encoder.pt")
    data = load_movielens(ml_root, text_encoder="someone/fake-encoder")
    assert torch.equal(data.text_embeddings, cached)


def test_full_descriptions_have_their_own_cache(ml_root):
    cached = torch.rand(3, 4)
    torch.save(cached, ml_root / "full_fake-encoder.pt")
    data = load_movielens(ml_root, "someone/fake-encoder", text_source="full")
    assert torch.equal(data.text_embeddings, cached)


def test_no_text_by_default(data):
    assert data.text_embeddings is None


def test_random_control(tmp_path):
    vectors = load_text_embeddings(tmp_path, ["a", "b"], "random")
    assert vectors.shape == (2, 768)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(2))
    assert torch.equal(vectors, load_text_embeddings(tmp_path, ["a", "b"], "random"))

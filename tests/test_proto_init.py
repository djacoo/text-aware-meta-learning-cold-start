import torch

from coldstart.model import adapt, build_model, meta_step
from coldstart.preprocess import load_tasks

CFG = {"emb_dim": 8, "hidden": 16, "proto_dim": 4}


def models(data):
    data.text_embeddings = torch.rand(3, 4)
    torch.manual_seed(0)
    plain = build_model(data, CFG)
    torch.manual_seed(0)
    proto = build_model(data, CFG | {"proto": True})
    return plain, proto


def test_untrained_prototype_changes_nothing(data):
    # the projection starts at zero, so before training the model is plain MeLU
    plain, proto = models(data)
    task = load_tasks(data, "warm_train", 0)[0]
    for name, value in adapt(plain, task, 2, 0.01).items():
        assert torch.equal(adapt(proto, task, 2, 0.01)[name], value)


def test_prototype_is_trained(data):
    _, model = models(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    tasks = load_tasks(data, "warm_train", 0)[:4]
    meta_step(model, tasks, optimizer, 1, 0.01)
    # with a zero projection only the projection gets a gradient at the first step
    assert model.proto_projector.projectors["out_weight"].weight.grad.abs().sum() > 0
    meta_step(model, tasks, optimizer, 1, 0.01)
    assert model.proto_net.proto_encoder[0].weight.grad.abs().sum() > 0

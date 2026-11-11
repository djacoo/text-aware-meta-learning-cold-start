from pathlib import Path

import torch

from coldstart.dataset import MovieLens, Task
from coldstart.model import PROTO_TARGET_PARAMS, MeLU, adapt, meta_step
from coldstart.prototype import ProtoProjector, UserTextPrototype


def test_proto_melu_integration():
    num_items = 50
    num_users = 10
    text_dim = 128
    proto_dim = 32
    emb_dim = 16
    hidden_dim = 32

    mock_dataset = MovieLens(
        root=Path("data/raw"),
        item_features={"genre": torch.randint(0, 5, (num_items, 3))},
        user_features={"gender": torch.randint(0, 2, (num_users,))},
        vocab_sizes={"genre": 5, "gender": 2},
        item_index={i: i for i in range(num_items)},
        user_index={i: i for i in range(num_users)},
        plots=["Mock plot description"] * num_items,
        descriptions=["Mock movie description"] * num_items,
        text_embeddings=torch.randn(num_items, text_dim),
    )

    mock_tasks = [
        Task(
            user=0,
            support_items=torch.tensor([0, 1, 2, 3, 4]),
            support_ratings=torch.tensor([5.0, 4.0, 1.0, 2.0, 5.0]),
            query_items=torch.tensor([5, 6, 7]),
            query_ratings=torch.tensor([4.0, 1.0, 3.0]),
        ),
        Task(
            user=1,
            support_items=torch.tensor([10, 11, 12, 13, 14]),
            support_ratings=torch.tensor([2.0, 1.0, 5.0, 4.0, 3.0]),
            query_items=torch.tensor([15, 16, 17]),
            query_ratings=torch.tensor([2.0, 5.0, 4.0]),
        ),
    ]

    melu_model = MeLU(mock_dataset, emb_dim=emb_dim, hidden=hidden_dim)
    proto_net = UserTextPrototype(text_dim=text_dim, proto_dim=proto_dim)
    proto_projector = ProtoProjector(
        proto_dim=proto_dim, model=melu_model, local_param_names=PROTO_TARGET_PARAMS
    )

    adapted_params = adapt(
        model=melu_model,
        task=mock_tasks[0],
        steps=2,
        lr=0.01,
        first_order=False,
        proto_net=proto_net,
        proto_projector=proto_projector,
        data=mock_dataset,
    )

    for name in PROTO_TARGET_PARAMS:
        assert name in adapted_params
        expected_shape = dict(melu_model.named_parameters())[name].shape
        assert adapted_params[name].shape == expected_shape

    optimizer = torch.optim.Adam(
        list(melu_model.parameters())
        + list(proto_net.parameters())
        + list(proto_projector.parameters()),
        lr=1e-3,
    )

    loss = meta_step(
        model=melu_model,
        tasks=mock_tasks,
        optimizer=optimizer,
        steps=2,
        lr=0.01,
        first_order=False,
        proto_net=proto_net,
        proto_projector=proto_projector,
        data=mock_dataset,
    )

    assert not torch.isnan(torch.tensor(loss))
    proto_encoder_grad = proto_net.proto_encoder[0].weight.grad
    projector_grad = list(proto_projector.projectors.values())[0].weight.grad

    assert proto_encoder_grad is not None
    assert projector_grad is not None

import sys
from pathlib import Path

import torch

from src.coldstart.dataset import MovieLens, Task
from src.coldstart.model import LOCAL_PARAMS, MeLU, adapt, meta_step
from src.coldstart.prototype import ProtoProjector, UserTextPrototype

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_integration_test():
    print("--- 1. Setting up Mock Data & Architecture ---")
    num_items = 50
    num_users = 10
    text_dim = 128
    proto_dim = 32
    emb_dim = 16
    hidden_dim = 32

    # 1. Mock MovieLens dataset
    mock_dataset = MovieLens(
        root=Path("data/raw"),
        item_features={"genre": torch.randint(0, 5, (num_items, 3))},
        user_features={"gender": torch.randint(0, 2, (num_users,))},
        vocab_sizes={"genre": 5, "gender": 2},
        item_index={i: i for i in range(num_items)},
        user_index={i: i for i in range(num_users)},
        plots=[],
        text_embeddings=torch.randn(num_items, text_dim),  # Mock text embeddings
    )

    # 2. Mock Tasks (Support Set = 5 items, Query Set = 3 items)
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

    # 3. Instantiate Models
    melu_model = MeLU(mock_dataset, emb_dim=emb_dim, hidden=hidden_dim)
    proto_net = UserTextPrototype(text_dim=text_dim, proto_dim=proto_dim)
    proto_projector = ProtoProjector(
        proto_dim=proto_dim, model=melu_model, local_param_names=LOCAL_PARAMS
    )

    print("--- 2. Testing Single User Adaptation (adapt) ---")
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

    # Assert all local parameters were modified and match target shapes
    for name in LOCAL_PARAMS:
        assert name in adapted_params, f"Missing parameter {name} in adapted_params"
        expected_shape = dict(melu_model.named_parameters())[name].shape
        actual_shape = adapted_params[name].shape
        assert actual_shape == expected_shape, (
            f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
        )

    print("[SUCCESS] Single adaptation step executed with matching parameter shapes.")

    print("--- 3. Testing Full Meta-Step & Backpropagation ---")
    # Group all trainable parameters in optimizer
    optimizer = torch.optim.Adam(
        list(melu_model.parameters())
        + list(proto_net.parameters())
        + list(proto_projector.parameters()),
        lr=1e-3,
    )

    # Perform meta-step
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

    # Assert loss value
    assert not torch.isnan(torch.tensor(loss)), "Meta-step loss is NaN!"
    print(f"Meta-step Query Loss: {loss:.4f}")

    # Assert gradients reached prototype layers (gradient flow check)
    proto_encoder_grad = proto_net.proto_encoder[0].weight.grad
    projector_grad = list(proto_projector.projectors.values())[0].weight.grad

    assert proto_encoder_grad is not None, "Gradient did NOT backpropagate to UserTextPrototype!"
    assert projector_grad is not None, "Gradient did NOT backpropagate to ProtoProjector!"
    assert not torch.all(proto_encoder_grad == 0), "Gradients in UserTextPrototype are all zero!"

    print("[SUCCESS] End-to-end gradient flow verified across Meta-Learning graph!")
    print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! Proto-MAML integration is ready. <<<")


if __name__ == "__main__":
    run_integration_test()

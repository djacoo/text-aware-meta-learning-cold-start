import torch
import torch.nn as nn


class UserTextPrototype(nn.Module):
    """
    Computes a user text prototype p_u from support set item text embeddings and ratings.
    """

    def __init__(self, text_dim: int, proto_dim: int, rating_threshold: float = 3.5):
        super().__init__()
        self.rating_threshold = rating_threshold

        # Maps concatenated [pos_proto | neg_proto] to latent space
        self.proto_encoder = nn.Sequential(
            nn.Linear(text_dim * 2, proto_dim), nn.ReLU(), nn.Linear(proto_dim, proto_dim)
        )

    def forward(self, supp_text_emb: torch.Tensor, supp_ratings: torch.Tensor) -> torch.Tensor:
        pos_mask = (supp_ratings >= self.rating_threshold).float().unsqueeze(-1)
        neg_mask = (supp_ratings < self.rating_threshold).float().unsqueeze(-1)

        pos_count = pos_mask.sum(dim=0)
        neg_count = neg_mask.sum(dim=0)

        pos_proto = (supp_text_emb * pos_mask).sum(dim=0) / (pos_count + 1e-8)
        neg_proto = (supp_text_emb * neg_mask).sum(dim=0) / (neg_count + 1e-8)

        if pos_count == 0:
            pos_proto = torch.zeros_like(neg_proto)
        if neg_count == 0:
            neg_proto = torch.zeros_like(pos_proto)

        combined = torch.cat([pos_proto, neg_proto], dim=-1)
        return self.proto_encoder(combined)


class ProtoProjector(nn.Module):
    """
    Projects user prototype p_u into parameter deltas matching MeLU local parameters.
    """

    def __init__(self, proto_dim: int, model: nn.Module, local_param_names: tuple):
        super().__init__()
        self.projectors = nn.ModuleDict()
        named_params = dict(model.named_parameters())

        for name in local_param_names:
            param_numel = named_params[name].numel()
            key = name.replace(".", "_")
            self.projectors[key] = nn.Linear(proto_dim, param_numel)

    def forward(self, p_u: torch.Tensor, current_params: dict) -> dict[str, torch.Tensor]:
        deltas = {}
        for name, param_tensor in current_params.items():
            key = name.replace(".", "_")
            if key in self.projectors:
                # Project p_u and reshape to match target parameter shape
                delta_flat = self.projectors[key](p_u)
                deltas[name] = delta_flat.view_as(param_tensor)
        return deltas

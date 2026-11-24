import torch
from torch import nn

PROTO_TARGET_PARAMS = ("out.weight", "out.bias")
# ratings from here up count as liked
LIKED = 3.5


class UserTextPrototype(nn.Module):
    def __init__(self, item_text, proto_dim):
        super().__init__()
        # the text vectors are fixed and come with the data, they are not saved with the weights
        self.register_buffer("item_text", item_text, persistent=False)
        self.proto_encoder = nn.Sequential(
            nn.Linear(item_text.shape[1] * 2, proto_dim), nn.ReLU(), nn.Linear(proto_dim, proto_dim)
        )

    def forward(self, items, ratings):
        # mean text of the liked and of the disliked support items, an empty group gives zeros
        texts = self.item_text[items]
        liked = (ratings >= LIKED).float().unsqueeze(-1)
        disliked = 1 - liked
        pos = (texts * liked).sum(dim=0) / (liked.sum(dim=0) + 1e-8)
        neg = (texts * disliked).sum(dim=0) / (disliked.sum(dim=0) + 1e-8)
        return self.proto_encoder(torch.cat([pos, neg], dim=-1))


class ProtoProjector(nn.Module):
    def __init__(self, proto_dim, model):
        super().__init__()
        self.projectors = nn.ModuleDict()
        for name in PROTO_TARGET_PARAMS:
            proj = nn.Linear(proto_dim, model.get_parameter(name).numel())
            # zero at the start, so training begins from plain MeLU
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.projectors[name.replace(".", "_")] = proj

    def forward(self, prototype, params):
        return {
            name: self.projectors[name.replace(".", "_")](prototype).view_as(params[name])
            for name in PROTO_TARGET_PARAMS
        }


def add_prototype(model, data, proto_dim):
    # the prototype modules live inside the model, so they train and are saved with it
    model.proto_net = UserTextPrototype(data.text_embeddings, proto_dim)
    model.proto_projector = ProtoProjector(proto_dim, model)

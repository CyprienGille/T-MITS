from torch import Tensor, nn


class Attention(nn.Module):
    """A simple masked attention module (uses 'Additive attention', cf Bahdanau 2014)

    Computes attention weights through a fully connected two-layer network (one Tanh non-linearity)
    """

    def __init__(self, d_model, hid_dim) -> None:
        super().__init__()
        self.lin_1 = nn.Linear(in_features=d_model, out_features=hid_dim)
        self.lin_2 = nn.Linear(in_features=hid_dim, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x: Tensor, mask: Tensor, mask_value=-1e15):
        attn_weights = self.lin_2(self.tanh(self.lin_1(x)))
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            attn_weights = mask * attn_weights + (1 - mask) * mask_value
        attn_weights = self.softmax(attn_weights)
        return attn_weights

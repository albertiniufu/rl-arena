import torch
from torch import nn

from ..encoders import build_encoder, encoder_output_size


class DuelingQNetwork(nn.Module):
    """Rede dueling: Q(s, a) = V(s) + A(s, a) − média_a A(s, a).

    O encoder aceita imagem `(C, H, W)` ou vetor `(N,)`. A centralização das
    vantagens remove a ambiguidade entre valor e vantagem.
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.v_head = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))
        self.a_head = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        feats = self.features(x)
        v = self.v_head(feats)
        a = self.a_head(feats)
        return v + a - a.mean(dim=1, keepdim=True)

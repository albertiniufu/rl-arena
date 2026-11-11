import torch
from torch import nn

from ..encoders import build_encoder, encoder_output_size


class QNetwork(nn.Module):
    """Rede padrão: encoder (CNN ou MLP) + cabeça com um Q por ação.

    Entrada `(batch, *obs_shape)`; saída `(batch, n_actions)`. Sem dueling:
    uma única cabeça produz todos os valores de ação.
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.head = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        return self.head(self.features(x))

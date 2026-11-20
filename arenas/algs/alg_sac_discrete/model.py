"""Ator categórico do SAC discreto."""
import torch
from torch import nn

from ..encoders import build_encoder, encoder_output_size


class DiscreteActor(nn.Module):
    """Encoder + MLP que devolve logits por ação."""

    def __init__(self, input_shape, n_actions, hidden=128):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.head = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, int(n_actions)),
        )

    def forward(self, x):
        return self.head(self.features(x))

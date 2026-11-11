"""Rede actor-critic do A2C.

Mesma arquitetura do PPO: encoder compartilhado, ator categórico com um
logit por ação e crítico escalar. Cada pacote mantém a própria cópia para
continuar autocontido.
"""
import torch
from torch import nn

from ..encoders import build_encoder, encoder_output_size


class ActorCriticNetwork(nn.Module):
    """Encoder + ator categórico + crítico escalar."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.actor = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        feats = self.features(x)
        return self.actor(feats), self.critic(feats).squeeze(-1)

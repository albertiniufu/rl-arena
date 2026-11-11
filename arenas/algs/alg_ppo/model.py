"""Rede actor-critic compartilhada do PPO.

O encoder aceita a grade semântica `(C, H, W)` ou um vetor `(N,)`; a cabeça
do ator produz um logit por ação (política categórica) e a do crítico
produz V(s).
"""
import torch
from torch import nn

from ..encoders import build_encoder, encoder_output_size


class ActorCriticNetwork(nn.Module):
    """Encoder + ator categórico + crítico escalar.

    Saída `(logits, value)`, com `logits` de shape `(batch, n_actions)` e
    `value` de shape `(batch,)`.
    """
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

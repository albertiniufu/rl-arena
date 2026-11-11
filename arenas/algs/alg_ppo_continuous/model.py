"""Rede actor-critic com ator Gaussiano para ações contínuas."""
import torch
from torch import nn

from ..encoders import build_encoder


class GaussianActorCritic(nn.Module):
    """Encoder + média/log-desvio por dimensão da ação + crítico escalar.

    Aceita observação `(N,)` ou `(C, H, W)`; a saída é
    `(mean, log_std, value)`, com `mean`/`log_std` de shape
    `(batch, action_dim)` e `value` de shape `(batch,)`.
    """
    def __init__(self, input_shape, action_dim, hidden=128):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        with torch.no_grad():
            n = self.features(torch.zeros(1, *input_shape)).shape[1]
        self.actor_mean = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        feats = self.features(x)
        mean = self.actor_mean(feats)
        # O log-desvio é um parâmetro global por dimensão, limitado para
        # manter a exploração estável.
        log_std = self.actor_log_std.clamp(-5.0, 2.0).expand_as(mean)
        value = self.critic(feats).squeeze(-1)
        return mean, log_std, value

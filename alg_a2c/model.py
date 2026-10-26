"""Rede actor-critic do A2C.

A arquitetura é a mesma do PPO: CNN compartilhada, ator categórico com um
logit por ação e crítico escalar. Cada pacote mantém a própria cópia para
continuar autocontido.
"""
import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """CNN + ator categórico + crítico escalar."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, c, h, w)).shape[1]
        self.actor = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        feats = self.features(x)
        return self.actor(feats), self.critic(feats).squeeze(-1)

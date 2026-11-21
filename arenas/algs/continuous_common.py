"""Componentes compartilhados pelos algoritmos contínuos off-policy.

DDPG, TD3 e SAC usam o mesmo encoder, replay e ideia de ator/crítico. Este
módulo concentra o que é comum para que cada `alg_*` contenha só a sua regra
de update.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn

from .encoders import build_encoder, encoder_output_size


class ContinuousReplayBuffer:
    """Replay uniforme para ações contínuas."""

    def __init__(self, capacity: int = 100_000):
        self.data = deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, terminated):
        self.data.append((
            np.array(state, dtype=np.float32, copy=True),
            np.asarray(action, dtype=np.float32).copy(),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(terminated),
        ))

    def sample(self, batch_size: int):
        data = list(self.data)
        batch = random.sample(data, int(batch_size))
        s, a, r, ns, term = zip(*batch)
        return (np.stack(s), np.stack(a),
                np.asarray(r, dtype=np.float32), np.stack(ns),
                np.asarray(term, dtype=np.float32))

    def __len__(self):
        return len(self.data)


class DeterministicActor(nn.Module):
    """Ator determinístico com saída `tanh` em [-1, 1]."""

    def __init__(self, input_shape, action_dim, hidden=128):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.head = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.head(self.features(x))


class GaussianActor(nn.Module):
    """Ator gaussiano com `tanh` (SAC)."""

    def __init__(self, input_shape, action_dim, hidden=128,
                 log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.mean = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, x):
        feats = self.features(x)
        mean = self.mean(feats)
        log_std = self.log_std(feats).clamp(
            self.log_std_min, self.log_std_max)
        return mean, log_std


class TwinCritic(nn.Module):
    """Dois críticos Q(s, a) independentes."""

    def __init__(self, input_shape, action_dim, hidden=128):
        super().__init__()
        input_shape = tuple(input_shape)
        self.q1 = Critic(input_shape, action_dim, hidden)
        self.q2 = Critic(input_shape, action_dim, hidden)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


class Critic(nn.Module):
    """Q(s, a) com encoder compartilhado e ação concatenada às features."""

    def __init__(self, input_shape, action_dim, hidden=128):
        super().__init__()
        input_shape = tuple(input_shape)
        self.features = build_encoder(input_shape)
        n = encoder_output_size(input_shape)
        self.head = nn.Sequential(
            nn.Linear(n + int(action_dim), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        feats = self.features(state)
        return self.head(torch.cat([feats, action], dim=1)).squeeze(-1)


class Temperature(nn.Module):
    """`log(alpha)` como parâmetro treinável do SAC."""

    def __init__(self, initial=0.2):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.tensor(float(np.log(initial)), dtype=torch.float32))

    @property
    def alpha(self):
        return self.log_alpha.exp()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak: target <- (1 - tau) * target + tau * source."""
    with torch.no_grad():
        for target_param, source_param in zip(
                target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def soft_update_module_dict(target: nn.ModuleDict, source: nn.ModuleDict,
                            tau: float) -> None:
    for key in target.keys():
        soft_update(target[key], source[key], tau)


def scale_action(action: torch.Tensor, low, high) -> torch.Tensor:
    """Mapeia ação em [-1, 1] para o Box [low, high]."""
    if low is None or high is None:
        return action
    low_t = torch.as_tensor(low, dtype=action.dtype, device=action.device)
    high_t = torch.as_tensor(high, dtype=action.dtype, device=action.device)
    return low_t + (action + 1.0) * 0.5 * (high_t - low_t)


def clamp_action(action: np.ndarray, low, high) -> np.ndarray:
    if low is None or high is None:
        return action
    return np.clip(action, low, high).astype(np.float32)


def squash_log_prob(mean, log_std, action, eps=1e-6):
    """log π(a|s) para `a = tanh(u)`, com correção do jacobiano."""
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    u = 0.5 * torch.log((1 + action + eps) / (1 - action + eps))
    log_prob = dist.log_prob(u).sum(dim=1)
    log_prob -= torch.log(1 - action.pow(2) + eps).sum(dim=1)
    return log_prob


class MultiOptimizer:
    """Agrupa otimizadores para o contrato de checkpoint (`state_dict`)."""

    def __init__(self, optimizers: dict):
        self.optimizers = dict(optimizers)

    def state_dict(self):
        return {name: opt.state_dict()
                for name, opt in self.optimizers.items()}

    def load_state_dict(self, state):
        for name, opt in self.optimizers.items():
            if name in state:
                opt.load_state_dict(state[name])

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers.values():
            opt.step()

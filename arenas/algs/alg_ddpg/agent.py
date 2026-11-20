"""DDPG: ator determinístico + crítico Q para ações contínuas."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..continuous_common import (
    ContinuousReplayBuffer,
    Critic,
    DeterministicActor,
    MultiOptimizer,
    clamp_action,
    scale_action,
    soft_update,
)


class DDPGAgent:
    algorithm = "ddpg"
    arch_version = "alg-ddpg-v1"
    continuous = True
    on_policy = False

    def __init__(self, obs_shape, n_actions, lr=1e-3, gamma=.99, tau=.005,
                 batch_size=64, buffer_size=100_000, warmup=1000, epsilon_min=.1,
                 noise_std=.1, noise_decay=.9999, noise_min=.01,
                 device=None, action_low=None, action_high=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.noise_std = float(noise_std)
        self.noise_decay = float(noise_decay)
        self.noise_min = float(noise_min)
        self.online = nn.ModuleDict({
            "actor": DeterministicActor(self.obs_shape, self.n_actions),
            "critic": Critic(self.obs_shape, self.n_actions),
        }).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.online["actor"].parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            self.online["critic"].parameters(), lr=lr)
        self.optimizer = MultiOptimizer({
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
        })
        self.buffer = ContinuousReplayBuffer(buffer_size)
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0
        self.action_low = None
        self.action_high = None
        self.set_action_bounds(action_low, action_high)

    def set_action_bounds(self, low, high) -> None:
        self.action_low = None if low is None else np.asarray(low, dtype=np.float32)
        self.action_high = None if high is None else np.asarray(high, dtype=np.float32)

    def _scale(self, action: torch.Tensor) -> torch.Tensor:
        return scale_action(action, self.action_low, self.action_high)

    @torch.no_grad()
    def act(self, state, greedy=False, action_mask=None):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        action = self._scale(self.online["actor"](x))
        if not greedy and self.noise_std > 0:
            action = action + torch.randn_like(action) * self.noise_std
        return clamp_action(action.squeeze(0).cpu().numpy(),
                            self.action_low, self.action_high)

    @torch.no_grad()
    def act_batch(self, states, greedy=False, action_mask=None):
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        action = self._scale(self.online["actor"](x))
        if not greedy and self.noise_std > 0:
            action = action + torch.randn_like(action) * self.noise_std
        return clamp_action(action.cpu().numpy(),
                            self.action_low, self.action_high)

    @torch.no_grad()
    def q_stats(self, state):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        action = self._scale(self.online["actor"](x))
        q = self.online["critic"](x, action)
        return float(q.mean()), float(q.max())

    def head_weights(self):
        return self.online["actor"].head[-2].weight

    def learn(self):
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return None
        s, a, r, ns, term = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(term, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self._scale(self.target["actor"](ns))
            next_q = self.target["critic"](ns, next_action)
            target = r + self.gamma * (1.0 - term) * next_q

        q = self.online["critic"](s, a)
        critic_loss = nn.functional.mse_loss(q, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online["critic"].parameters(), 10.0))
        self.critic_optimizer.step()

        action = self._scale(self.online["actor"](s))
        actor_loss = -self.online["critic"](s, action).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online["actor"].parameters(), 10.0))
        self.actor_optimizer.step()

        soft_update(self.target["actor"], self.online["actor"], self.tau)
        soft_update(self.target["critic"], self.online["critic"], self.tau)
        self.noise_std = max(self.noise_min,
                             self.noise_std * self.noise_decay)
        self.steps += 1
        return float(critic_loss.item())


Agent = DDPGAgent

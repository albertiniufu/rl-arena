"""SAC: ator gaussiano com entropia, dois críticos e alpha automático."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..continuous_common import (
    ContinuousReplayBuffer,
    Critic,
    GaussianActor,
    MultiOptimizer,
    Temperature,
    clamp_action,
    scale_action,
    soft_update,
    squash_log_prob,
)


class SACAgent:
    algorithm = "sac"
    arch_version = "alg-sac-v1"
    continuous = True
    on_policy = False

    def __init__(self, obs_shape, n_actions, lr=3e-4, gamma=.99, tau=.005,
                 batch_size=64, buffer_size=100_000, warmup=1000, epsilon_min=.1,
                 alpha=0.2, auto_alpha=True, target_entropy=None,
                 device=None, action_low=None, action_high=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.warmup = int(warmup)
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy = (float(target_entropy)
                               if target_entropy is not None
                               else -float(self.n_actions))
        self.online = nn.ModuleDict({
            "actor": GaussianActor(self.obs_shape, self.n_actions),
            "critic1": Critic(self.obs_shape, self.n_actions),
            "critic2": Critic(self.obs_shape, self.n_actions),
            "temperature": Temperature(alpha),
        }).to(self.device)
        self.target = nn.ModuleDict({
            "critic1": copy.deepcopy(self.online["critic1"]),
            "critic2": copy.deepcopy(self.online["critic2"]),
        }).to(self.device)
        if not self.auto_alpha:
            self.online["temperature"].log_alpha.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.online["actor"].parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.online["critic1"].parameters())
            + list(self.online["critic2"].parameters()), lr=lr)
        self.alpha_optimizer = torch.optim.Adam(
            [self.online["temperature"].log_alpha], lr=lr)
        self.optimizer = MultiOptimizer({
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
            "alpha": self.alpha_optimizer,
        })
        self.buffer = ContinuousReplayBuffer(buffer_size)
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0
        self.action_low = None
        self.action_high = None
        self.set_action_bounds(action_low, action_high)

    @property
    def alpha(self):
        return float(self.online["temperature"].alpha.detach().cpu())

    def set_action_bounds(self, low, high) -> None:
        self.action_low = None if low is None else np.asarray(low, dtype=np.float32)
        self.action_high = None if high is None else np.asarray(high, dtype=np.float32)

    def _scale(self, action):
        return scale_action(action, self.action_low, self.action_high)

    def _sample_action(self, state, greedy=False):
        mean, log_std = self.online["actor"](state)
        if greedy:
            return torch.tanh(mean)
        return torch.tanh(mean + log_std.exp() * torch.randn_like(mean))

    @torch.no_grad()
    def act(self, state, greedy=False, action_mask=None):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        action = self._scale(self._sample_action(x, greedy=greedy))
        return clamp_action(action.squeeze(0).cpu().numpy(),
                            self.action_low, self.action_high)

    @torch.no_grad()
    def act_batch(self, states, greedy=False, action_mask=None):
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        action = self._scale(self._sample_action(x, greedy=greedy))
        return clamp_action(action.cpu().numpy(),
                            self.action_low, self.action_high)

    @torch.no_grad()
    def q_stats(self, state):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        action = self._scale(self._sample_action(x, greedy=True))
        q1 = self.online["critic1"](x, action)
        q2 = self.online["critic2"](x, action)
        q = torch.minimum(q1, q2)
        return float(q.mean()), float(q.max())

    def head_weights(self):
        return self.online["actor"].mean[-1].weight

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
            mean, log_std = self.online["actor"](ns)
            u = mean + log_std.exp() * torch.randn_like(mean)
            next_action = torch.tanh(u)
            log_prob = squash_log_prob(mean, log_std, next_action)
            next_scaled = self._scale(next_action)
            q1_next = self.target["critic1"](ns, next_scaled)
            q2_next = self.target["critic2"](ns, next_scaled)
            next_q = torch.minimum(q1_next, q2_next) - \
                self.online["temperature"].alpha * log_prob
            target = r + self.gamma * (1.0 - term) * next_q

        q1 = self.online["critic1"](s, a)
        q2 = self.online["critic2"](s, a)
        critic_loss = nn.functional.mse_loss(q1, target) + \
            nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            list(self.online["critic1"].parameters())
            + list(self.online["critic2"].parameters()), 10.0))
        self.critic_optimizer.step()

        mean, log_std = self.online["actor"](s)
        u = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(u)
        log_prob = squash_log_prob(mean, log_std, action)
        scaled = self._scale(action)
        q = torch.minimum(self.online["critic1"](s, scaled),
                          self.online["critic2"](s, scaled))
        alpha = self.online["temperature"].alpha.detach()
        actor_loss = (alpha * log_prob - q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online["actor"].parameters(), 10.0))
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.online["temperature"].log_alpha *
                           (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        soft_update(self.target["critic1"], self.online["critic1"], self.tau)
        soft_update(self.target["critic2"], self.online["critic2"], self.tau)
        self.steps += 1
        return float((critic_loss + actor_loss).item())


Agent = SACAgent

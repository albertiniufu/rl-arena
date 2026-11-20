"""SAC discreto: ator categórico, dois críticos e entropia automática."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..alg_dqn.model import QNetwork
from ..alg_dqn.replay_buffer import ReplayBuffer
from ..continuous_common import MultiOptimizer, Temperature, soft_update
from .model import DiscreteActor


class SACDiscreteAgent:
    algorithm = "sac_discrete"
    arch_version = "alg-sac-discrete-v1"
    continuous = False
    on_policy = False

    def __init__(self, obs_shape, n_actions, lr=3e-4, gamma=.99, tau=.005,
                 batch_size=64, buffer_size=100_000, warmup=1000, epsilon_min=.1,
                 alpha=0.2, auto_alpha=True, target_entropy=None,
                 device=None):
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
                               else float(math.log(self.n_actions)))
        self.online = nn.ModuleDict({
            "actor": DiscreteActor(self.obs_shape, self.n_actions),
            "critic1": QNetwork(self.obs_shape, self.n_actions),
            "critic2": QNetwork(self.obs_shape, self.n_actions),
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
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0

    @property
    def alpha(self):
        return float(self.online["temperature"].alpha.detach().cpu())

    def set_action_bounds(self, low, high) -> None:
        return None

    @staticmethod
    def _masked_logits(logits, action_mask):
        if action_mask is None:
            return logits
        mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool,
                               device=logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return logits.masked_fill(~mask, float("-inf"))

    @torch.no_grad()
    def act(self, state, greedy=False, action_mask=None):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits = self._masked_logits(self.online["actor"](x), action_mask)
        if greedy:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False, action_mask=None):
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        logits = self._masked_logits(self.online["actor"](x), action_mask)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def q_stats(self, state):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        q = torch.minimum(self.online["critic1"](x),
                          self.online["critic2"](x))
        return float(q.mean()), float(q.max())

    def head_weights(self):
        return self.online["actor"].head[-1].weight

    def learn(self):
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return None
        s, a, r, ns, term = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(term, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_logits = self.online["actor"](ns)
            next_probs = torch.softmax(next_logits, dim=1)
            next_log_probs = torch.log_softmax(next_logits, dim=1)
            next_actions = Categorical(probs=next_probs).sample()
            next_log_prob = next_log_probs.gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            q1_next = self.target["critic1"](ns).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            q2_next = self.target["critic2"](ns).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            next_q = torch.minimum(q1_next, q2_next) - \
                self.online["temperature"].alpha * next_log_prob
            target = r + self.gamma * (1.0 - term) * next_q

        q1 = self.online["critic1"](s).gather(1, a).squeeze(1)
        q2 = self.online["critic2"](s).gather(1, a).squeeze(1)
        critic_loss = nn.functional.mse_loss(q1, target) + \
            nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            list(self.online["critic1"].parameters())
            + list(self.online["critic2"].parameters()), 10.0))
        self.critic_optimizer.step()

        logits = self.online["actor"](s)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        q = torch.minimum(self.online["critic1"](s),
                          self.online["critic2"](s))
        alpha = self.online["temperature"].alpha.detach()
        actor_loss = (probs * (alpha * log_probs - q)).sum(dim=1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online["actor"].parameters(), 10.0))
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.online["temperature"].log_alpha * (
                probs.detach() * (log_probs.detach()
                                  + self.target_entropy)).sum(dim=1)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        soft_update(self.target["critic1"], self.online["critic1"], self.tau)
        soft_update(self.target["critic2"], self.online["critic2"], self.tau)
        self.steps += 1
        return float((critic_loss + actor_loss).item())


Agent = SACDiscreteAgent

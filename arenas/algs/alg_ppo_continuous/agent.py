"""PPO com política Gaussiana para espaços de ação contínuos.

O contrato é o mesmo dos demais algoritmos (`obs_shape`, `n_actions`,
`online`, `target`, `optimizer`, `buffer`, `steps`, `epsilon`, `act`,
`act_batch`, `q_stats`, `head_weights`, `learn`, `end_episode`,
`finish_training`), mas `n_actions` representa a **dimensão** do vetor de
ação. O treinador passa os limites do `Box` via `set_action_bounds`.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from ..alg_ppo.agent import compute_gae
from .model import GaussianActorCritic
from .rollout_buffer import ContinuousRolloutBuffer


class PPOContinuousAgent:
    algorithm = "ppo_continuous"
    arch_version = "alg-ppo-continuous-v1"
    on_policy = True
    continuous = True

    def __init__(self, obs_shape, n_actions, lr=3e-4, gamma=.99,
                 batch_size=64, rollout_steps=1024, epochs=10, clip=0.2,
                 gae_lambda=.95, value_coef=.5, entropy_coef=0.0,
                 max_grad_norm=0.5, device=None,
                 action_low=None, action_high=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)  # dimensão da ação contínua
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.online = GaussianActorCritic(
            self.obs_shape, self.n_actions).to(self.device)
        self.target = self.online
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ContinuousRolloutBuffer()
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0
        self.action_low = None
        self.action_high = None
        self.set_action_bounds(action_low, action_high)

    # ------------------------------------------------------------------ #
    # Ação
    # ------------------------------------------------------------------ #
    def set_action_bounds(self, low, high) -> None:
        self.action_low = None if low is None else np.asarray(
            low, dtype=np.float32)
        self.action_high = None if high is None else np.asarray(
            high, dtype=np.float32)

    def _clamp(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_low is None or self.action_high is None:
            return action
        low = torch.as_tensor(
            self.action_low, dtype=torch.float32, device=action.device)
        high = torch.as_tensor(
            self.action_high, dtype=torch.float32, device=action.device)
        return torch.max(torch.min(action, high), low)

    @torch.no_grad()
    def act(self, state, greedy=False):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        mean, log_std, _ = self.online(x)
        if greedy:
            action = mean
        else:
            action = mean + log_std.exp() * torch.randn_like(mean)
        return self._clamp(action).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        mean, log_std, _ = self.online(x)
        if greedy:
            action = mean
        else:
            action = mean + log_std.exp() * torch.randn_like(mean)
        return self._clamp(action).cpu().numpy()

    @torch.no_grad()
    def q_stats(self, state):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        _, _, value = self.online(x)
        return float(value.mean()), float(value.max())

    def head_weights(self):
        return self.online.actor_mean[-1].weight

    # ------------------------------------------------------------------ #
    # Aprendizado
    # ------------------------------------------------------------------ #
    def end_episode(self, info=None):
        self.buffer.mark_episode_end()
        return None

    def finish_training(self):
        if len(self.buffer) == 0:
            return None
        return self._update()

    def learn(self):
        if len(self.buffer) < self.rollout_steps:
            return None
        return self._update()

    def _update(self):
        states, actions, rewards, next_states, terminated, episode_ends = \
            self.buffer.as_arrays()
        if len(states) == 0:
            return None
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32)
        term = torch.as_tensor(terminated, dtype=torch.float32)
        ends = torch.as_tensor(episode_ends, dtype=torch.float32)

        with torch.no_grad():
            mean, log_std, values = self.online(s)
            _, _, next_values = self.online(ns)
            dist = Normal(mean, log_std.exp())
            old_log_probs = dist.log_prob(a).sum(dim=1)
            advantages, returns = compute_gae(
                r, values.cpu(), next_values.cpu(), term, ends,
                self.gamma, self.gae_lambda)
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)

        n = s.shape[0]
        last_loss = 0.0
        for _ in range(self.epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                mean, log_std, values = self.online(s[idx])
                dist = Normal(mean, log_std.exp())
                log_probs = dist.log_prob(a[idx]).sum(dim=1)
                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy = dist.entropy().sum(dim=1).mean()
                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.online.parameters(), self.max_grad_norm))
                self.optimizer.step()
                last_loss = float(loss.item())
        self.steps += 1
        self.buffer.clear()
        return last_loss


Agent = PPOContinuousAgent

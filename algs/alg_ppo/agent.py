"""PPO com política categórica e GAE.

O treino é síncrono: o buffer acumula transições e o update só acontece
quando o rollout enche (ou no `finish_training`). Como nenhum peso muda
durante a coleta, a política que gerou o rollout é exatamente a política
online no início do update; por isso os log-probs antigos e os valores
podem ser recalculados antes das épocas de otimização, sem guardar
metadados em cada transição.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer


def compute_gae(rewards, values, next_values, terminated, episode_ends,
                gamma, gae_lambda):
    """Vantagens e retornos GAE(λ), respeitando fronteiras de episódio.

    `terminated` zera o bootstrap de terminais verdadeiros; `episode_ends`
    interrompe a propagação da vantagem entre episódios, mas ainda permite
    bootstrap quando o fim é truncamento por tempo.
    """
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - terminated[t].float()
        delta = (rewards[t] + gamma * nonterminal * next_values[t]
                 - values[t])
        # Se o episódio acabou (inclusive por truncamento), a vantagem não
        # atravessa para o episódio seguinte.
        propagates = (1.0 - episode_ends[t].float()) * nonterminal
        gae = delta + gamma * gae_lambda * propagates * gae
        advantages[t] = gae
    return advantages, advantages + values


class PPOAgent:
    algorithm = "ppo"
    arch_version = "alg-ppo-v1"
    on_policy = True

    def __init__(self, obs_shape, n_actions, lr=2.5e-4, gamma=.99,
                 batch_size=64, rollout_steps=1024, epochs=4, clip=0.2,
                 gae_lambda=.95, value_coef=.5, entropy_coef=.01,
                 max_grad_norm=10.0, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.online = ActorCriticNetwork(self.obs_shape, self.n_actions).to(self.device)
        # O contrato de checkpoint pede uma rede `target`; o PPO não tem
        # rede-alvo, então o alias evita duplicar parâmetros e o checkpoint
        # grava os mesmos pesos nos dois campos.
        self.target = self.online
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0

    @torch.no_grad()
    def act(self, state, greedy=False):
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits, _ = self.online(x)
        if greedy:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """Ações para um lote de estados, uma por linha (shape `(n,)`)."""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        logits, _ = self.online(x)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def q_stats(self, state):
        """Valores do crítico para o estado (média e máximo)."""
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        _, value = self.online(x)
        return float(value.mean()), float(value.max())

    def head_weights(self):
        """Pesos da última camada do ator, para o histograma do TensorBoard."""
        return self.online.actor[-1].weight

    def end_episode(self, info=None):
        """Marca a fronteira de episódio no rollout; o update vem depois."""
        self.buffer.mark_episode_end()
        return None

    def finish_training(self):
        """Consome o rollout parcial que sobrou no fim do treino."""
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
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        ends = torch.as_tensor(episode_ends, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits_old, values = self.online(s)
            _, next_values = self.online(ns)
            old_log_probs = torch.log_softmax(logits_old, dim=1).gather(
                1, a.unsqueeze(1)).squeeze(1)
            advantages, returns = compute_gae(
                r, values, next_values, term, ends, self.gamma, self.gae_lambda)
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)
        n = s.shape[0]
        last_loss = 0.0
        for _ in range(self.epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                logits, values = self.online(s[idx])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(a[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
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


Agent = PPOAgent

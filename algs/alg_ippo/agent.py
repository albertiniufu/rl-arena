"""IPPO (Independent PPO) com política compartilhada entre os agentes.

Cada agente tem o seu próprio buffer de rollout, mas todos compartilham a
mesma rede ActorCritic e o mesmo otimizador. O update acontece quando todos os
buffers atingem `rollout_steps` (horizonte por agente); o GAE é calculado por
trajetória e as transições são concatenadas para as épocas de PPO. Como nenhum
peso muda durante a coleta, os log-probs e valores antigos podem ser
recalculados antes do update — não é preciso guardá-los em cada transição.

Em todos-contra-todos e em times a mesma rede aprende os dois papéis: a
observação egocêntrica informa quem é aliado e quem é inimigo, então a
política compartilhada é simétrica.
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
        propagates = (1.0 - episode_ends[t].float()) * nonterminal
        gae = delta + gamma * gae_lambda * propagates * gae
        advantages[t] = gae
    return advantages, advantages + values


class MultiRolloutBuffer:
    """Um `RolloutBuffer` por agente, com índice explícito nas operações."""

    def __init__(self, n_agents):
        self.buffers = [RolloutBuffer() for _ in range(int(n_agents))]

    def add(self, agent_index, state, action, reward, next_state, terminated,
            log_prob=None):
        self.buffers[agent_index].add(
            state, action, reward, next_state, terminated, log_prob)

    def mark_episode_end(self, agent_index):
        self.buffers[agent_index].mark_episode_end()

    def as_arrays(self):
        return [buffer.as_arrays() for buffer in self.buffers]

    def clear(self):
        for buffer in self.buffers:
            buffer.clear()

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)


class IPPOAgent:
    algorithm = "ippo"
    arch_version = "alg-ippo-v1"
    on_policy = True
    multi_agent = True

    def __init__(self, obs_shape, n_actions, n_agents=2, team_ids=None,
                 lr=2.5e-4, gamma=.99, batch_size=64, rollout_steps=1024,
                 epochs=4, clip=0.2, gae_lambda=.95, value_coef=.5,
                 entropy_coef=.01, max_grad_norm=10.0, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.n_agents = int(n_agents)
        self.team_ids = (tuple(int(t) for t in team_ids)
                         if team_ids is not None
                         else tuple(range(self.n_agents)))
        if len(self.team_ids) != self.n_agents:
            raise ValueError(
                f"team_ids deve ter {self.n_agents} entradas "
                f"(recebido {len(self.team_ids)})")
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        # Uma única rede/otimizador para todos os agentes.
        self.online = ActorCriticNetwork(
            self.obs_shape, self.n_actions).to(self.device)
        # O contrato de checkpoint pede uma rede `target`; o IPPO não tem
        # rede-alvo, então o alias evita duplicar parâmetros.
        self.target = self.online
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = MultiRolloutBuffer(self.n_agents)
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
        """Ações para um lote de estados `(batch, C, H, W)`."""
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

    def add_transition(self, agent_index, state, action, reward, next_state,
                       terminated, log_prob=None):
        self.buffer.add(agent_index, state, action, reward, next_state,
                        terminated, log_prob)

    def end_episode(self, info=None):
        """Marca a fronteira de episódio no rollout de todos os agentes."""
        for i in range(self.n_agents):
            self.buffer.mark_episode_end(i)
        return None

    def finish_training(self):
        """Consome o rollout parcial que sobrou no fim do treino."""
        if len(self.buffer) == 0:
            return None
        return self._update()

    def learn(self):
        if any(len(buffer) < self.rollout_steps
               for buffer in self.buffer.buffers):
            return None
        return self._update()

    def _update(self):
        arrays = self.buffer.as_arrays()
        if all(len(row[0]) == 0 for row in arrays):
            return None
        states, actions, old_log_probs = [], [], []
        advantages, returns = [], []
        for (s_np, a_np, r_np, ns_np, term_np, end_np) in arrays:
            if len(s_np) == 0:
                continue
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            a = torch.as_tensor(a_np, dtype=torch.int64, device=self.device)
            ns = torch.as_tensor(ns_np, dtype=torch.float32, device=self.device)
            r = torch.as_tensor(r_np, dtype=torch.float32)
            term = torch.as_tensor(term_np, dtype=torch.float32)
            ends = torch.as_tensor(end_np, dtype=torch.float32)
            with torch.no_grad():
                logits_old, values = self.online(s)
                _, next_values = self.online(ns)
                # Coleta e update usam a mesma política: recalcular é exato.
                log_probs = torch.log_softmax(logits_old, dim=1).gather(
                    1, a.unsqueeze(1)).squeeze(1)
                adv, ret = compute_gae(
                    r, values.cpu(), next_values.cpu(), term, ends,
                    self.gamma, self.gae_lambda)
            states.append(s)
            actions.append(a)
            old_log_probs.append(log_probs)
            advantages.append(adv.to(self.device))
            returns.append(ret.to(self.device))

        s = torch.cat(states, dim=0)
        a = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)
        advantages = torch.cat(advantages, dim=0)
        returns = torch.cat(returns, dim=0)
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


Agent = IPPOAgent

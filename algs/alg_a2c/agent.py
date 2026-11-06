"""A2C (advantage actor-critic) com retornos n-step.

O ambiente tem uma ação discreta por passo; a política é categórica. O
buffer acumula `n_steps` transições contíguas e o update acontece quando
ele enche ou quando o episódio termina (para não descartar o rabo). Como
não há update no meio da coleta, os valores e log-probs usados no alvo
podem ser recalculados no início do update, exatamente como no PPO.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer


def n_step_returns(rewards, values, next_values, terminated, episode_ends,
                   gamma, n_steps):
    """Retornos n-step, com bootstrap no truncamento e corte no fim do episódio.

    `values` entra na assinatura para deixar explícito que o retorno é
    comparado com V(s) no update; o cálculo do retorno em si usa apenas as
    recompensas e o valor do estado seguinte. Um terminal verdadeiro não
    faz bootstrap; um fim de episódio truncado faz bootstrap com
    `next_values[j]` e interrompe a soma.
    """
    del values  # usado pelo agente no cálculo da vantagem
    returns = torch.zeros_like(rewards)
    n = rewards.shape[0]
    for i in range(n):
        total = rewards.new_zeros(())
        discount = 1.0
        for k in range(n_steps):
            j = i + k
            if j >= n:
                break
            total = total + discount * rewards[j]
            done = bool(terminated[j].item())
            end = bool(episode_ends[j].item())
            if end or done:
                if not done:
                    total = total + discount * gamma * next_values[j]
                break
            if k == n_steps - 1 or j == n - 1:
                total = total + discount * gamma * next_values[j]
                break
            discount *= gamma
        returns[i] = total
    return returns


class A2CAgent:
    algorithm = "a2c"
    arch_version = "alg-a2c-v1"
    on_policy = True

    def __init__(self, obs_shape, n_actions, lr=1e-3, gamma=.99,
                 batch_size=64, n_steps=None, value_coef=.5,
                 entropy_coef=.01, max_grad_norm=10.0, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.batch_size = int(batch_size)
        # `batch_size` vem da CLI comum; por padrão ele também define o
        # horizonte n-step do A2C (64 passos no protocolo de comparação).
        self.n_steps = int(n_steps if n_steps is not None else batch_size)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.online = ActorCriticNetwork(self.obs_shape, self.n_actions).to(self.device)
        # Sem rede-alvo no A2C; o alias satisfaz o contrato de checkpoint.
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
        """Fecha o episódio; se o rollout ficou incompleto, atualiza agora."""
        self.buffer.mark_episode_end()
        if len(self.buffer) == 0:
            return None
        return self._update()

    def finish_training(self):
        """Consome um rollout parcial que sobrou no fim do treino."""
        if len(self.buffer) == 0:
            return None
        return self._update()

    def learn(self):
        if len(self.buffer) < self.n_steps:
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
            _, values = self.online(s)
            _, next_values = self.online(ns)
            returns = n_step_returns(
                r, values, next_values, term, ends, self.gamma, self.n_steps)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)
        logits, values = self.online(s)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(a)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy = dist.entropy().mean()
        loss = (policy_loss + self.value_coef * value_loss
                - self.entropy_coef * entropy)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.max_grad_norm))
        self.optimizer.step()
        self.steps += 1
        self.buffer.clear()
        return float(loss.item())


Agent = A2CAgent

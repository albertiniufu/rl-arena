"""HAPPO — Heterogeneous-Agent PPO com atualização sequencial.

O cenário coletores × empurradores quebra a simetria do IPPO/MAPPO: metade
dos times só coleta, a outra metade só empurra. O HAPPO [Kuba et al., 2021]
foi proposto exatamente para esse caso:

* **Atores heterogêneos** — cada agente tem a própria rede de política. Não
  há sharing de parâmetros, então um update de um agente não invalida
  silenciosamente a razão de importância dos outros.
* **Crítico centralizado** — no treino, V é estimado com as features de
  todos os agentes (CTDE). Na execução o crítico desaparece e cada ator age
  só com a própria observação.
* **Update sequencial** — os agentes são atualizados em ordem. Para o agente
  ``i``, a razão de política é multiplicada pelo produto acumulado
  ``M = Π_{j<i} π_j_novo(a_j|o_j) / π_j_antigo(a_j|o_j)`` dos agentes já
  atualizados. Esse fator é o que corrige a vantagem conjunta decomposta
  (Heterogeneous-Agent Advantage Decomposition) sob a nova política dos
  agentes anteriores. ``M`` é recalculado sem gradiente depois de cada
  update, porque as políticas anteriores estão congeladas naquele momento.

A máscara de ações é derivada dos papéis do `Scenario`: coletores têm
`PUSH_LEFT`/`PUSH_RIGHT` mascarados com probabilidade zero (a ação nunca é
amostrada nem entra na perda). O ambiente também converte empurrões de
coletores em `STOP`, como defesa em profundidade.

O buffer é indexado por passo conjunto (como no MAPPO) e o GAE é calculado
por agente com os valores centralizados. Não há replay nem rede-alvo.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from arenas.seek_coin.arena.actions import PUSH_LEFT, PUSH_RIGHT
from arenas.seek_coin.arena.config import COLLECTOR_ROLE

from .model import HAPPONetwork
from .rollout_buffer import HAPPORolloutBuffer


def compute_gae(rewards, values, next_values, terminated, episode_ends,
                gamma, gae_lambda):
    """Vantagens e retornos GAE(λ) por agente, respeitando episódios.

    Todas as entradas têm shape `(T, n_agents)`. `terminated` zera o
    bootstrap de terminais verdadeiros; `episode_ends` interrompe a
    propagação entre episódios, mas ainda permite bootstrap no truncamento
    por tempo.
    """
    n_steps, n_agents = rewards.shape
    advantages = torch.zeros_like(rewards)
    for agent in range(n_agents):
        gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for t in reversed(range(n_steps)):
            nonterminal = 1.0 - terminated[t, agent].float()
            delta = (rewards[t, agent]
                     + gamma * nonterminal * next_values[t, agent]
                     - values[t, agent])
            propagates = (1.0 - episode_ends[t, agent].float()) * nonterminal
            gae = delta + gamma * gae_lambda * propagates * gae
            advantages[t, agent] = gae
    return advantages, advantages + values


class HAPPOAgent:
    algorithm = "happo"
    arch_version = "alg-happo-v1"
    on_policy = True
    multi_agent = True
    # Sinaliza ao train.py/evaluate.py/play.py que o construtor aceita os
    # papéis do cenário (e que a política os usa para mascarar ações).
    accepts_roles = True

    def __init__(self, obs_shape, n_actions, n_agents=2, team_ids=None,
                 agent_roles=None, lr=2.5e-4, gamma=.99, batch_size=64,
                 rollout_steps=1024, epochs=4, clip=0.2, gae_lambda=.95,
                 value_coef=.5, entropy_coef=.01, max_grad_norm=10.0,
                 value_clip=None, eval_chunk_steps=64, device=None):
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
        self.agent_roles = tuple(agent_roles) if agent_roles else ()
        if self.agent_roles and len(self.agent_roles) != self.n_agents:
            raise ValueError(
                f"agent_roles deve ter {self.n_agents} entradas "
                f"(recebido {len(self.agent_roles)})")
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.value_clip = None if value_clip is None else float(value_clip)
        self.eval_chunk_steps = max(1, int(eval_chunk_steps))

        # Máscara de ações por agente (True = permitida). Sem papéis, tudo é
        # permitido e o HAPPO se comporta como um MAPPO com atores separados.
        masks = torch.ones(self.n_agents, self.n_actions, dtype=torch.bool)
        for i, role in enumerate(self.agent_roles):
            if role == COLLECTOR_ROLE:
                masks[i, PUSH_LEFT] = False
                masks[i, PUSH_RIGHT] = False
        self.action_masks = masks.to(self.device)
        # `masks.all()` acelera o caminho simétrico (sem mascaramento).
        self._has_action_mask = not bool(self.action_masks.all().item())

        self.online = HAPPONetwork(
            self.obs_shape, self.n_actions, self.n_agents).to(self.device)
        # O checkpoint exige `target`; não há rede-alvo no PPO.
        self.target = self.online
        # Um Adam só para todo o container: o update sequencial zera os
        # gradientes e faz backward apenas com a perda do ator corrente, então
        # o passo toca só os parâmetros daquele ator. O crítico é atualizado
        # em um passo separado. O estado do Adam é serializado normalmente.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = HAPPORolloutBuffer(self.n_agents)
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0

    # --------------------------------------------------------------- execução
    def _masked_logits(self, logits, agent_index):
        if not self._has_action_mask:
            return logits
        mask = self.action_masks[int(agent_index)]
        return logits.masked_fill(~mask, -1e9)

    @torch.no_grad()
    def act(self, state, greedy=False):
        """Ação do agente 0 (contrato genérico de avaliação/política)."""
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits = self._masked_logits(self.online.actor_logits(x, 0), 0)
        if greedy:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """Ações para o lote `(n_agents, C, H, W)` na ordem dos agentes."""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        if x.ndim != 4 or x.shape[0] != self.n_agents:
            raise ValueError(
                f"act_batch espera ({self.n_agents}, C, H, W); "
                f"recebido {tuple(x.shape)}")
        logits = torch.stack([
            self._masked_logits(self.online.actor_logits(x[i:i + 1], i), i)
            for i in range(self.n_agents)
        ]).squeeze(1)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def q_stats(self, state):
        """Valores do crítico centralizado (média e máximo), para telemetria."""
        x = np.asarray(state)
        if x.ndim == 3:
            x = np.repeat(x[None], self.n_agents, axis=0)
        if x.ndim != 4 or x.shape[0] != self.n_agents:
            raise ValueError(
                f"q_stats espera (C,H,W) ou ({self.n_agents},C,H,W); "
                f"recebido {x.shape}")
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        values = self.online.joint_values(tensor.unsqueeze(0))[0]
        return float(values.mean()), float(values.max())

    def head_weights(self):
        """Pesos da última camada do ator 0, para o histograma do TensorBoard."""
        return self.online.actors[0].actor[-1].weight

    # --------------------------------------------------------------- coleta
    def add_transition(self, agent_index, state, action, reward, next_state,
                       terminated, log_prob=None):
        self.buffer.add(agent_index, state, action, reward, next_state,
                        terminated, log_prob)

    def end_episode(self, info=None):
        for i in range(self.n_agents):
            self.buffer.mark_episode_end(i)
        return None

    def finish_training(self):
        if self.buffer.n_steps == 0:
            return None
        return self._update()

    # --------------------------------------------------------------- update
    def learn(self):
        if self.buffer.n_steps < self.rollout_steps:
            return None
        return self._update()

    def _old_log_probs_and_values(self, states, next_states, actions):
        """Log-probs antigos e V por agente, em blocos e sem gradiente."""
        n_steps = states.shape[0]
        old_log_probs = torch.zeros(n_steps, self.n_agents)
        values = torch.zeros(n_steps, self.n_agents)
        next_values = torch.zeros(n_steps, self.n_agents)
        with torch.no_grad():
            for start in range(0, n_steps, self.eval_chunk_steps):
                stop = start + self.eval_chunk_steps
                s = states[start:stop].to(self.device)
                a = actions[start:stop].to(self.device)
                for i in range(self.n_agents):
                    logits = self._masked_logits(
                        self.online.actor_logits(s[:, i], i), i)
                    logp = torch.log_softmax(logits, dim=-1).gather(
                        -1, a[:, i].unsqueeze(-1)).squeeze(-1)
                    old_log_probs[start:stop, i] = logp.cpu()
                values[start:stop] = self.online.joint_values(s).cpu()
                ns = next_states[start:stop].to(self.device)
                next_values[start:stop] = self.online.joint_values(ns).cpu()
        return old_log_probs, values, next_values

    def _update(self):
        arrays = self.buffer.as_arrays()
        if arrays is None:
            return None
        # O rollout fica na CPU; só o minibatch corrente vai para o device.
        states = torch.as_tensor(arrays["states"], dtype=torch.float32)
        next_states = torch.as_tensor(
            arrays["next_states"], dtype=torch.float32)
        actions = torch.as_tensor(arrays["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(arrays["rewards"], dtype=torch.float32)
        terminated = torch.as_tensor(arrays["terminated"], dtype=torch.bool)
        ends = torch.as_tensor(arrays["episode_ends"], dtype=torch.bool)

        n_steps = actions.shape[0]
        old_log_probs, values, next_values = self._old_log_probs_and_values(
            states, next_states, actions)

        advantages, returns = compute_gae(
            rewards, values, next_values, terminated, ends,
            self.gamma, self.gae_lambda)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values_old = values.to(self.device)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8)

        steps_per_batch = max(1, self.batch_size // self.n_agents)
        last_loss = 0.0

        # 1) Crítico centralizado: um V por agente contra os retornos do GAE.
        for _ in range(self.epochs):
            order = torch.randperm(n_steps)
            for start in range(0, n_steps, steps_per_batch):
                idx = order[start:start + steps_per_batch]
                mb_states = states[idx].to(self.device)
                mb_values = self.online.joint_values(mb_states)
                mb_returns = returns[idx]
                if self.value_clip is None:
                    value_loss = nn.functional.mse_loss(
                        mb_values, mb_returns)
                else:
                    old = values_old[idx]
                    clipped_values = old + torch.clamp(
                        mb_values - old, -self.value_clip, self.value_clip)
                    value_loss = torch.max(
                        nn.functional.mse_loss(mb_values, mb_returns),
                        nn.functional.mse_loss(clipped_values, mb_returns))
                self.optimizer.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                self.last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.online.critic.parameters(), self.max_grad_norm))
                self.optimizer.step()
                last_loss = float(value_loss.item())

        # 2) Atores em ordem, com o produto acumulado das razões anteriores.
        #    `ratio_pre` é atualizado após cada agente com a política nova,
        #    sem gradiente: os agentes anteriores estão congelados.
        ratio_pre = torch.ones(n_steps, device=self.device)
        for i in range(self.n_agents):
            actor = self.online.actors[i]
            for _ in range(self.epochs):
                order = torch.randperm(n_steps)
                for start in range(0, n_steps, steps_per_batch):
                    idx = order[start:start + steps_per_batch]
                    mb_states = states[idx, i].to(self.device)
                    mb_actions = actions[idx, i].to(self.device)
                    logits = self._masked_logits(
                        actor(mb_states), i)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(mb_actions)
                    ratio = torch.exp(
                        log_probs - old_log_probs[idx, i].to(self.device))
                    # M multiplica a razão do agente corrente pela razão
                    # conjunta dos agentes já atualizados (papel do HAPPO:
                    # a vantagem condiciona em a_{1:i-1}).
                    ratio = ratio * ratio_pre[idx]
                    adv = advantages[idx, i]
                    unclipped = ratio * adv
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss - self.entropy_coef * entropy
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(), self.max_grad_norm))
                    self.optimizer.step()
                    last_loss = float(loss.item())
            # Recalcula a razão do agente i já atualizado e acumula em M.
            with torch.no_grad():
                for start in range(0, n_steps, self.eval_chunk_steps):
                    stop = start + self.eval_chunk_steps
                    s = states[start:stop, i].to(self.device)
                    a = actions[start:stop, i].to(self.device)
                    logits = self._masked_logits(actor(s), i)
                    new_logp = torch.log_softmax(logits, dim=-1).gather(
                        -1, a.unsqueeze(-1)).squeeze(-1)
                    ratio_i = torch.exp(
                        new_logp - old_log_probs[start:stop, i].to(
                            self.device))
                    ratio_pre[start:stop] = ratio_pre[start:stop] * ratio_i

        self.steps += 1
        self.buffer.clear()
        return last_loss


Agent = HAPPOAgent

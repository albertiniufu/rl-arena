"""MAPPO (Multi-Agent PPO) com ator compartilhado e crítico centralizado.

MAPPO é o PPO adaptado a multiagente no paradigma CTDE (treino centralizado,
execução descentralizada):

* **Ator compartilhado** — todos os agentes usam a mesma política e só veem a
  própria observação egocêntrica. Em execução (`act`/`act_batch`) nada global é
  consultado, então `evaluate.py`/`play.py` funcionam como no IPPO.
* **Crítico centralizado** — o valor `V(s_global)` é estimado a partir das
  features de **todos** os agentes no mesmo passo, mais um one-hot que
  identifica o agente consultado. O crítico só é usado no update; é o que
  reduz a variância do gradiente em relação ao crítico local do IPPO, ao custo
  de depender da observação conjunta durante a coleta.
* **PPO por agente** — o GAE é calculado por trajetória (respeitando a
  fronteira de episódio) e as vantagens são normalizadas no conjunto de
  transições de todos os agentes. Cada minibatch é um bloco de passos
  conjuntos: as features da CNN são calculadas uma vez por agente e
  alimentam as duas cabeças.

A política é estocástica na coleta e gulosa em `act(..., greedy=True)`; não há
replay nem rede-alvo (`target` é alias de `online`).
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import CentralizedActorCriticNetwork
from .rollout_buffer import MultiAgentRolloutBuffer


def compute_gae(rewards, values, next_values, terminated, episode_ends,
                gamma, gae_lambda):
    """Vantagens e retornos GAE(λ) para cada agente.

    Todas as entradas têm shape `(T, n_agents)`. `terminated` zera o
    bootstrap de terminais verdadeiros; `episode_ends` interrompe a
    propagação da vantagem entre episódios, mas ainda permite bootstrap
    quando o fim é truncamento por tempo.
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


class MAPPOAgent:
    algorithm = "mappo"
    arch_version = "alg-mappo-v1"
    on_policy = True
    multi_agent = True

    def __init__(self, obs_shape, n_actions, n_agents=2, team_ids=None,
                 lr=2.5e-4, gamma=.99, batch_size=64, rollout_steps=1024,
                 epochs=4, clip=0.2, gae_lambda=.95, value_coef=.5,
                 entropy_coef=.01, max_grad_norm=10.0, value_clip=None,
                 device=None):
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
        # `batch_size` é interpretado em transições e arredondado para baixo
        # em blocos de passos conjuntos (batch_size // n_agents).
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.epochs = int(epochs)
        self.clip = float(clip)
        self.gae_lambda = float(gae_lambda)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.value_clip = None if value_clip is None else float(value_clip)
        # Uma única rede (ator + crítico centralizado) para todos os agentes.
        self.online = CentralizedActorCriticNetwork(
            self.obs_shape, self.n_actions, self.n_agents).to(self.device)
        # O contrato de checkpoint pede uma rede `target`; o MAPPO não tem
        # rede-alvo, então o alias evita duplicar parâmetros.
        self.target = self.online
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = MultiAgentRolloutBuffer(self.n_agents)
        self.steps = 0
        self.epsilon = 0.0
        self.last_grad_norm = 0.0

    # --------------------------------------------------------------- execução
    @torch.no_grad()
    def act(self, state, greedy=False):
        """Ação de um agente a partir da observação local."""
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        logits = self.online(x)
        if greedy:
            return int(logits.argmax(dim=1).item())
        return int(Categorical(logits=logits).sample().item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """Ações para um lote local `(n_agents, C, H, W)`."""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        logits = self.online(x)
        if greedy:
            actions = logits.argmax(dim=1)
        else:
            actions = Categorical(logits=logits).sample()
        return actions.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def q_stats(self, state):
        """Valores do crítico centralizado (média e máximo), para telemetria.

        Aceita a observação local `(C, H, W)` — nesse caso replica o estado
        para todos os agentes — ou o lote conjunto `(n_agents, C, H, W)` já
        na ordem dos agentes. O crítico é centralizado, então o valor depende
        do estado conjunto.
        """
        x = np.asarray(state)
        if x.ndim == 3:
            x = np.repeat(x[None], self.n_agents, axis=0)
        if x.ndim != 4 or x.shape[0] != self.n_agents:
            raise ValueError(
                f"q_stats espera (C,H,W) ou ({self.n_agents},C,H,W); "
                f"recebido {x.shape}")
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        feats = self.online.features(tensor).reshape(
            1, self.n_agents, self.online.n_features)
        values = self.online.joint_values(feats)[0]
        return float(values.mean()), float(values.max())

    def head_weights(self):
        """Pesos da última camada do ator, para o histograma do TensorBoard."""
        return self.online.actor[-1].weight

    # --------------------------------------------------------------- coleta
    def add_transition(self, agent_index, state, action, reward, next_state,
                       terminated, log_prob=None):
        """Registra a transição do agente; o buffer monta o passo conjunto."""
        self.buffer.add(agent_index, state, action, reward, next_state,
                        terminated, log_prob)

    def end_episode(self, info=None):
        """Marca a fronteira do episódio para todos os agentes."""
        for i in range(self.n_agents):
            self.buffer.mark_episode_end(i)
        return None

    def finish_training(self):
        """Consome o rollout parcial que sobrou no fim do treino."""
        if self.buffer.n_steps == 0:
            return None
        return self._update()

    # --------------------------------------------------------------- update
    def learn(self):
        """Atualiza quando o rollout tem `rollout_steps` passos conjuntos."""
        if self.buffer.n_steps < self.rollout_steps:
            return None
        return self._update()

    def _features(self, states):
        """Features da CNN para `(..., C, H, W)` -> `(..., n_features)`."""
        shape = states.shape[:-3]
        flat = states.reshape(-1, *self.obs_shape)
        feats = self.online.features(flat)
        return feats.reshape(*shape, self.online.n_features)

    def _update(self):
        arrays = self.buffer.as_arrays()
        if arrays is None:
            return None
        states = torch.as_tensor(arrays["states"], dtype=torch.float32,
                                 device=self.device)
        next_states = torch.as_tensor(arrays["next_states"],
                                      dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(arrays["actions"], dtype=torch.int64,
                                  device=self.device)
        returns_np = arrays["rewards"]
        terminated_cpu = torch.as_tensor(arrays["terminated"], dtype=torch.bool)
        ends_cpu = torch.as_tensor(arrays["episode_ends"], dtype=torch.bool)

        n_steps, n_agents = actions.shape
        with torch.no_grad():
            # A política não muda durante a coleta: recalcular os log-probs
            # antigos com os pesos atuais é exato.
            feats = self._features(states)
            logits = self.online.actor(feats)
            old_log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1, actions.unsqueeze(-1)).squeeze(-1)
            values = self.online.joint_values(feats)
            next_feats = self._features(next_states)
            next_values = self.online.joint_values(next_feats)

        rewards_cpu = torch.as_tensor(returns_np, dtype=torch.float32)
        advantages, returns = compute_gae(
            rewards_cpu, values.cpu(), next_values.cpu(), terminated_cpu,
            ends_cpu, self.gamma, self.gae_lambda)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values_old = values.detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8)

        # Cada minibatch é um bloco de passos conjuntos: as features da CNN
        # são calculadas uma vez e servem ao ator e ao crítico.
        steps_per_batch = max(1, self.batch_size // n_agents)
        last_loss = 0.0
        for _ in range(self.epochs):
            order = torch.randperm(n_steps, device=self.device)
            for start in range(0, n_steps, steps_per_batch):
                idx = order[start:start + steps_per_batch]
                mb_feats = self._features(states[idx])  # (S, n, F)
                mb_logits = self.online.actor(mb_feats)
                dist = Categorical(logits=mb_logits)
                log_probs = dist.log_prob(actions[idx])
                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                mb_values = self.online.joint_values(mb_feats)
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


Agent = MAPPOAgent

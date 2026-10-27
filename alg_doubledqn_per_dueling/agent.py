"""Double DQN + PER rank-based + dueling, com o mesmo alvo fixo do DQN.

As três diferenças em relação ao `alg_dqn`:
  * Double DQN: a ação do próximo estado é escolhida pela online e o valor
    vem da target (menos superestimação);
  * PER: a amostragem do replay favorece transições com erro de TD alto
    (`rank^-α`, com pesos de importância corrigindo o viés no loss e β
    agendado de `beta_start` até 1);
  * dueling: a rede separa V(s) de A(s, a).
"""
import random

import numpy as np
import torch
from torch import nn

from .model import DuelingQNetwork
from .replay_buffer import PrioritizedReplayBuffer

# Mesmo alvo fixo do alg_dqn (C=10), para a comparação isolar as três
# técnicas deste pacote.
C = 10


@torch.no_grad()
def double_td_target(online, target, next_states, reward, terminated, gamma):
    """Alvo do Double DQN: `online` escolhe a ação, `target` avalia.

    `terminated` só é 1 em terminais verdadeiros; truncamento por tempo
    mantém o bootstrap.
    """
    next_actions = online(next_states).argmax(dim=1, keepdim=True)
    next_q = target(next_states).gather(1, next_actions).squeeze(1)
    return reward + gamma * (1.0 - terminated) * next_q


class DoubleDQNPerDuelingAgent:
    algorithm = "doubledqn_per_dueling"
    arch_version = "alg-ddqn-per-dueling-v1"
    C = C

    def __init__(self, obs_shape, n_actions, lr=1e-3, gamma=.99,
                 epsilon=1.0, epsilon_min=.05, epsilon_decay=.9995,
                 batch_size=64, buffer_size=50000, warmup=0, per_alpha=0.6,
                 beta_start=0.4, beta_steps=100_000, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        # `warmup` adia o primeiro update até o buffer ter transições
        # suficientes (evita aprender só com o começo do episódio).
        self.warmup = int(warmup)
        self.per_alpha = float(per_alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = max(1, int(beta_steps))
        self.online = DuelingQNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target = DuelingQNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        # O PER devolve um peso por amostra; a perda é reduzida depois de
        # multiplicar esses pesos (por isso `reduction="none"`).
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha=self.per_alpha)
        self.steps = 0
        self.last_grad_norm = 0.0
        # Índices do último sample() priorizado, expostos para inspeção/testes.
        self._last_indices = None

    def _beta(self):
        """β do PER: sobe linearmente de `beta_start` até 1."""
        frac = min(1.0, self.steps / self.beta_steps)
        return self.beta_start + (1.0 - self.beta_start) * frac

    @torch.no_grad()
    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """Ações para um lote de estados, uma por linha (shape `(n,)`)."""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        actions = self.online(x).argmax(dim=1).cpu().numpy().astype(np.int64)
        if not greedy:
            for i in range(actions.shape[0]):
                if random.random() < self.epsilon:
                    actions[i] = random.randrange(self.n_actions)
        return actions

    @torch.no_grad()
    def q_stats(self, state):
        """(média, máximo) dos Q-valores da rede online para um estado."""
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        q = self.online(x)
        return float(q.mean()), float(q.max())

    def head_weights(self):
        """Pesos da última camada, para o histograma do TensorBoard.

        No dueling, o ramo de valor (escalar) faz o papel da cabeça.
        """
        return self.online.v_head[-1].weight

    def learn(self):
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return None
        states, actions, rewards, next_states, terms, indices, weights = \
            self.buffer.sample(self.batch_size, alpha=self.per_alpha,
                               beta=self._beta())
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(terms, dtype=torch.float32, device=self.device)
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        q = self.online(s).gather(1, a).squeeze(1)
        target = double_td_target(self.online, self.target, ns, r, term, self.gamma)
        # Peso de importância por amostra (β agendado) corrige o viés do PER.
        loss = (self.loss_fn(q, target) * w).mean()
        # O erro de TD do batch vira a prioridade da próxima amostragem.
        # Nenhum `add` aconteceu desde o sample, então os índices valem.
        self.buffer.update_priorities(
            indices, (target - q).detach().abs().cpu().numpy())
        self._last_indices = indices
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ devolve a norma total ANTES do clip.
        self.last_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0))
        self.optimizer.step()
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.steps % self.C == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

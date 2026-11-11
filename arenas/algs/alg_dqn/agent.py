"""DQN padrão (vanilla): alvo fixo, replay uniforme e Q direto por ação.

Sem Double DQN, sem PER e sem dueling — é a referência simples para estudar
o algoritmo e comparar com o `alg_doubledqn_per_dueling`.
"""
import random

import numpy as np
import torch
from torch import nn

from .model import QNetwork
from .replay_buffer import ReplayBuffer

# Alvo fixo: a cada C learn() bem-sucedidos a target recebe uma cópia dura
# da online (o "C" da notação clássica do DQN). A busca de hiperparâmetros
# mostrou que C=10 é agressivo demais: com a rampa de ε e --learn-every 4,
# C=1000 coletou 15,8 moedas de treino contra 8,8 do C=10, e a avaliação
# final coletou 12,3 contra 5,9, com Q e gradiente bem menores.
C = 1000


def td_target(reward, next_q, terminated, gamma):
    """Alvo de TD de 1 passo: r + γ·Q_alvo(s', a*)·(1 − terminal).

    `terminated` só é 1 em terminais verdadeiros; truncamento por tempo
    mantém o bootstrap.
    """
    return reward + gamma * (1.0 - terminated) * next_q


class DQNAgent:
    algorithm = "dqn"
    arch_version = "alg-dqn-v1"
    C = C

    def __init__(self, obs_shape, n_actions, lr=1e-3, gamma=.99,
                 epsilon=1.0, epsilon_min=.05, epsilon_decay=.9995,
                 batch_size=64, buffer_size=50000, warmup=0, device=None):
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
        self.online = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.last_grad_norm = 0.0

    @torch.no_grad()
    def act(self, state, greedy=False, action_mask=None):
        legal = None
        mask_np = None
        if action_mask is not None:
            mask_np = np.asarray(action_mask, dtype=bool)
            legal = np.flatnonzero(mask_np)
            if legal.size == 0:
                return 0
        if not greedy and random.random() < self.epsilon:
            if legal is not None:
                return int(random.choice(legal))
            return random.randrange(self.n_actions)
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        q = self.online(x)
        if mask_np is not None:
            mask = torch.as_tensor(
                mask_np, dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask.unsqueeze(0), float("-inf"))
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False, action_mask=None):
        """Ações para um lote de estados, uma por linha (shape `(n,)`)."""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=self.device)
        q = self.online(x)
        mask_np = None
        if action_mask is not None:
            mask_np = np.asarray(action_mask, dtype=bool)
            if mask_np.ndim == 1:
                mask_np = np.broadcast_to(
                    mask_np, (x.shape[0], mask_np.shape[0]))
            mask = torch.as_tensor(
                mask_np, dtype=torch.bool, device=self.device)
            q = q.masked_fill(~mask, float("-inf"))
        actions = q.argmax(dim=1).cpu().numpy().astype(np.int64)
        if not greedy:
            for i in range(actions.shape[0]):
                if random.random() < self.epsilon:
                    if mask_np is not None:
                        legal = np.flatnonzero(mask_np[i])
                        if legal.size:
                            actions[i] = int(random.choice(legal))
                    else:
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
        """Pesos da última camada, para o histograma do TensorBoard."""
        return self.online.head[-1].weight

    def learn(self):
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return None
        if getattr(self.buffer, "has_masks", lambda: False)():
            (states, actions, rewards, next_states, terms,
             next_masks) = self.buffer.sample_with_masks(self.batch_size)
        else:
            states, actions, rewards, next_states, terms = \
                self.buffer.sample(self.batch_size)
            next_masks = None
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(terms, dtype=torch.float32, device=self.device)
        q = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(ns)
            if next_masks is not None:
                mask = torch.as_tensor(
                    next_masks, dtype=torch.bool, device=self.device)
                next_q_values = next_q_values.masked_fill(
                    ~mask, float("-inf"))
            next_q = next_q_values.max(dim=1).values
            if next_masks is not None:
                # Estados terminais sem ação legal: não há futuro a estimar.
                next_q = torch.where(torch.isinf(next_q),
                                     torch.zeros_like(next_q), next_q)
            target = td_target(r, next_q, term, self.gamma)
        loss = self.loss_fn(q, target)
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

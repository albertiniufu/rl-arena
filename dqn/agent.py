import random
import numpy as np
import torch
from torch import nn
from .model import DQN
from .replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, obs_shape, n_actions=12, lr=1e-3, gamma=.99,
                 epsilon=1.0, epsilon_min=.1,
                 batch_size=64, target_update=500, buffer_size=50000, device=None,
                 warmup=1000):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions, self.gamma = n_actions, gamma
        self.epsilon, self.epsilon_min = epsilon, epsilon_min
        self.batch_size, self.target_update = batch_size, target_update
        self.warmup = warmup
        self.online = DQN(obs_shape, n_actions).to(self.device)
        self.target = DQN(obs_shape, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.last_grad_norm = 0.0

    @torch.no_grad()
    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    @torch.no_grad()
    def q_stats(self, state):
        """(média, máximo) dos Q-valores da rede online para um estado."""
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(x)
        return float(q.mean()), float(q.max())

    def learn(self):
        # Warmup: enche o buffer antes de aprender, para as primeiras amostras
        # não serem quase todas do começo do episódio. O batch_size é um piso
        # (o sample() exige pelo menos isso).
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return None
        states, actions, rewards, next_states, terminated, truncated = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        # Truncation ends collection but is not a true terminal state, so bootstrap is retained.
        q = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=1).values
            target = r + self.gamma * (1.0 - term) * next_q
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ devolve a norma total ANTES do clip.
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0))
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


class CheckpointPolicy:
    """Adapta um `DQNAgent` ao contrato de política `act(ctx) -> int`.

    Age sempre de forma gulosa; permite colocar um agente treinado no lugar
    do adversário (ex.: `--opponent auto` no train.py).
    """
    def __init__(self, agent):
        self.agent = agent

    def act(self, ctx):
        return int(self.agent.act(ctx.obs, greedy=True))

import random
import numpy as np
import torch
from torch import nn
from .model import DQN
from .replay_buffer import ReplayBuffer

def _td_targets(online, target, r_n, ns, term, gamma, window, double=True):
    """Alvo de TD para um batch amostrado do buffer (retornos n-step).

    `r_n` já carrega os descontos internos da janela, então gamma não incide
    de novo sobre ele. O bootstrap é descontado por `gamma ** window`, onde
    `window` é o número real de transições acumuladas na amostra (pode ser
    menor que n quando a janela para em terminal, truncamento ou no fim do
    buffer). Com Double DQN (`double=True`), a ação é escolhida pela rede
    online e o valor vem da target; sem double, usa o máximo da target (DQN
    clássico). `term == 1` só para `terminated` verdadeiro: truncamento
    encerra a coleta mas não é estado terminal, então o bootstrap é mantido.
    """
    with torch.no_grad():
        if double:
            actions = online(ns).argmax(dim=1, keepdim=True)
            next_q = target(ns).gather(1, actions).squeeze(1)
        else:
            next_q = target(ns).max(dim=1).values
        discount = gamma ** window
        return r_n + discount * (1.0 - term) * next_q


class DQNAgent:
    def __init__(self, obs_shape, n_actions=12, lr=1e-3, gamma=.99,
                 epsilon=1.0, epsilon_min=.1,
                 batch_size=64, target_update=500, buffer_size=50000, device=None,
                 warmup=1000, double=True, n_step=1):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions, self.gamma = n_actions, gamma
        self.epsilon, self.epsilon_min = epsilon, epsilon_min
        self.batch_size, self.target_update = batch_size, target_update
        self.warmup = warmup
        self.double, self.n_step = double, n_step
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
        states, actions, r_n, next_states, term_np, window_np = self.buffer.sample(
            self.batch_size, n=self.n_step, gamma=self.gamma)
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r_n, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        term = torch.as_tensor(term_np, dtype=torch.float32, device=self.device)
        window = torch.as_tensor(window_np, dtype=torch.int64, device=self.device)
        q = self.online(s).gather(1, a).squeeze(1)
        target = _td_targets(self.online, self.target, r, ns, term,
                             self.gamma, window, self.double)
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
        # Com frame stack, o DQN espera a pilha `(5 * frame_stack, H, W)`;
        # sem stack, cai no frame único `ctx.obs` (comportamento histórico).
        state = ctx.stacked_obs if ctx.stacked_obs is not None else ctx.obs
        return int(self.agent.act(state, greedy=True))

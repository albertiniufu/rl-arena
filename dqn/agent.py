import random
import numpy as np
import torch
from torch import nn
from .model import DQN
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

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
                 epsilon=1.0, epsilon_min=.05, epsilon_decay=.9995,
                 batch_size=64, target_update=500, buffer_size=50000, device=None,
                 double=True, n_step=1, dueling=False, per=False, tau=0.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions, self.gamma = n_actions, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.batch_size, self.target_update = batch_size, target_update
        self.double, self.n_step = double, n_step
        self.dueling, self.per = dueling, per
        self.tau = tau
        # Online e target precisam da mesma cabeça para o load_state_dict da
        # sincronização periódica casar as chaves.
        self.online = DQN(obs_shape, n_actions, dueling=dueling).to(self.device)
        self.target = DQN(obs_shape, n_actions, dueling=dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = (PrioritizedReplayBuffer(buffer_size)
                       if per else ReplayBuffer(buffer_size))
        self.steps = 0
        self.last_grad_norm = 0.0
        # Índices do último sample() priorizado (None sem PER), expostos para
        # inspeção/testes; o `learn()` os usa antes de qualquer outro `add`.
        self._last_indices = None

    @torch.no_grad()
    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    @torch.no_grad()
    def act_batch(self, states, greedy=False):
        """Ações para um lote de estados, uma por linha (int64, shape `(n,)`).

        Mesma semântica de `act`: com `greedy=True`, argmax por linha; sem
        greedy, cada linha troca a ação pela aleatória com probabilidade
        `epsilon`, usando o módulo `random` global (como em `act`, o RNG é
        semeado pelo `train.py`).
        """
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
        x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(x)
        return float(q.mean()), float(q.max())

    def learn(self):
        # O batch_size é um piso: o sample() exige pelo menos isso.
        if len(self.buffer) < self.batch_size:
            return None
        sample = self.buffer.sample(
            self.batch_size, n=self.n_step, gamma=self.gamma)
        if self.per:
            # Replay priorizado devolve também as posições amostradas.
            states, actions, r_n, next_states, term_np, window_np, indices = sample
        else:
            states, actions, r_n, next_states, term_np, window_np = sample
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
        if self.per:
            # O erro de TD do batch vira a prioridade da próxima amostragem.
            # Nenhum `add` aconteceu desde o sample, então os índices valem.
            self.buffer.update_priorities(
                indices, (target - q).detach().abs().cpu().numpy())
            self._last_indices = indices
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # clip_grad_norm_ devolve a norma total ANTES do clip.
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0))
        self.optimizer.step()
        self.steps += 1
        # Decaimento por passo (configuração da Fase 0): a rampa por episódio
        # da Fase 1 foi reprovada no A/B e revertida.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.tau > 0.0:
            # Soft update (Polyak): a target avança uma fração `tau` na
            # direção da online a cada passo de aprendizado, no lugar da
            # sincronização dura por `target_update` (mecanismos mutuamente
            # exclusivos). Vale para pesos e buffers, como o load_state_dict.
            with torch.no_grad():
                for param_t, param_o in zip(self.target.parameters(),
                                            self.online.parameters()):
                    param_t.data.mul_(1.0 - self.tau).add_(param_o.data, alpha=self.tau)
                for buf_t, buf_o in zip(self.target.buffers(),
                                        self.online.buffers()):
                    buf_t.data.mul_(1.0 - self.tau).add_(buf_o.data, alpha=self.tau)
        elif self.steps % self.target_update == 0:
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

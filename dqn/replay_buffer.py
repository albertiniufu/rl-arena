from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, terminated, truncated):
        self.data.append((np.array(state, copy=True), int(action), float(reward),
                         np.array(next_state, copy=True), bool(terminated), bool(truncated)))
    def sample(self, batch_size, n=1, gamma=0.99):
        """Sorteia transições e devolve retornos n-step
        `(s, a, r_n, ns, term, window)`.

        Para cada índice sorteado, acumula `r_n = Σ_{k=0..n-1} γ^k·r_{t+k}`
        caminhando até `n` posições à frente no deque; a janela para na
        fronteira de episódio (`terminated` ou `truncated`) e `ns`/`term` vêm
        da transição onde parou. `window` é o número de transições realmente
        acumuladas (1..n, menor que `n` quando a janela para em terminal,
        truncamento ou no fim do buffer), usado para descontar o bootstrap
        por `γ**window`. Com `n=1`, o comportamento é o histórico:
        `r_n == r`, `ns == next_state`, `term == terminated` e `window == 1`.
        """
        idxs = random.sample(range(len(self.data)), batch_size)
        s, a, r_n, ns, term, window = [], [], [], [], [], []
        for i in idxs:
            state, action = self.data[i][0], self.data[i][1]
            reward = 0.0
            length = 0
            for k in range(n):
                if i + k >= len(self.data):
                    break  # fim do buffer: fica com a última transição visitada
                _, _, r, next_state, terminated, truncated = self.data[i + k]
                reward += (gamma ** k) * r
                length += 1
                if terminated or truncated:
                    break  # não atravessa fronteira de episódio
            s.append(state)
            a.append(action)
            r_n.append(reward)
            ns.append(next_state)
            term.append(terminated)
            window.append(length)
        return (np.stack(s), np.asarray(a), np.asarray(r_n, dtype=np.float32),
                np.stack(ns), np.asarray(term, dtype=np.float32),
                np.asarray(window, dtype=np.int64))
    def __len__(self):
        return len(self.data)

    def state_dict(self):
        """Formato empilhado: um array por campo, não um pickle por transição.

        Empilhar evita o custo de serializar cada transição e torna os bytes
        determinísticos: cada campo tem um único dtype, criado agora, igual em
        qualquer execução — inclusive após carregar um checkpoint.
        """
        if not self.data:
            return {}
        s, a, r, ns, term, trunc = zip(*self.data)
        return {
            "states": np.stack(s),
            "actions": np.asarray(a, dtype=np.int64),
            "rewards": np.asarray(r, dtype=np.float32),
            "next_states": np.stack(ns),
            "terminated": np.asarray(term, dtype=np.bool_),
            "truncated": np.asarray(trunc, dtype=np.bool_),
        }

    def load_state_dict(self, state):
        if isinstance(state, dict):
            items = []
            if state:
                for s, a, r, ns, t, tr in zip(state["states"], state["actions"], state["rewards"],
                                             state["next_states"], state["terminated"], state["truncated"]):
                    items.append((np.array(s, copy=True), int(a), float(r),
                                  np.array(ns, copy=True), bool(t), bool(tr)))
            self.data = deque(items, maxlen=self.data.maxlen)
        else:
            # Formato antigo: lista de tuplas por transição.
            self.data = deque(state, maxlen=self.data.maxlen)


class PrioritizedReplayBuffer:
    """Replay priorizado rank-based, sem α/β nem annealing de pesos.

    `priorities` anda em paralelo com `data` (mesmo `maxlen`): a posição `i`
    é a chave usada por `update_priorities`. A amostragem é proporcional a
    `1/rank(prioridade)` — rank 1 é a maior prioridade —, e novas transições
    entram com a maior prioridade atual para serem visitadas logo.
    """

    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated, truncated,
            priority=None):
        """Guarda a transição e sua prioridade (padrão: a maior atual)."""
        self.data.append((np.array(state, copy=True), int(action), float(reward),
                          np.array(next_state, copy=True), bool(terminated), bool(truncated)))
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(float(priority))

    def sample(self, batch_size, n=1, gamma=0.99):
        """Sorteia por rank e devolve retornos n-step e os índices amostrados
        `(s, a, r_n, ns, term, window, indices)`.

        Os pesos são `1/rank` sobre as prioridades do snapshot, normalizados;
        o sorteio é com reposição e usa o RNG global `np.random` (semeado
        pelo `train.py`). O walk n-step é o mesmo do `ReplayBuffer`: acumula
        `r_n = Σ_{k=0..n-1} γ^k·r_{t+k}` até `n` posições à frente, parando
        na fronteira de episódio (`terminated`/`truncated`) ou no fim do
        snapshot; `window` é o número real de transições acumuladas (1..n),
        usado para descontar o bootstrap por `γ**window`. `indices` são as
        posições no snapshot, para `update_priorities` depois do erro de TD.
        """
        data = list(self.data)
        prios = np.asarray(self.priorities, dtype=float)
        order = np.argsort(-prios, kind="stable")
        ranks = np.empty(len(prios), dtype=np.int64)
        ranks[order] = np.arange(1, len(prios) + 1)
        weights = 1.0 / ranks
        probs = weights / weights.sum()
        idxs = np.random.choice(len(data), size=batch_size, replace=True, p=probs)
        s, a, r_n, ns, term, window = [], [], [], [], [], []
        for i in idxs:
            state, action = data[i][0], data[i][1]
            reward = 0.0
            length = 0
            for k in range(n):
                if i + k >= len(data):
                    break  # fim do buffer: fica com a última transição visitada
                _, _, r, next_state, terminated, truncated = data[i + k]
                reward += (gamma ** k) * r
                length += 1
                if terminated or truncated:
                    break  # não atravessa fronteira de episódio
            s.append(state)
            a.append(action)
            r_n.append(reward)
            ns.append(next_state)
            term.append(terminated)
            window.append(length)
        return (np.stack(s), np.asarray(a), np.asarray(r_n, dtype=np.float32),
                np.stack(ns), np.asarray(term, dtype=np.float32),
                np.asarray(window, dtype=np.int64), np.asarray(idxs, dtype=np.int64))

    def update_priorities(self, indices, errors):
        """Reescreve as prioridades: `prioridade[i] = |erro| + 1e-6`.

        Invariante: os índices só valem enquanto não houver `add` entre o
        `sample` e o `update_priorities` (o `learn()` garante isso). Com o
        deque cheio, um `add` descartaria a transição mais antiga e
        deslocaria as posições; no intervalo entre amostrar e atualizar nada
        é adicionado, então as posições continuam válidas.
        """
        for i, err in zip(indices, errors):
            self.priorities[int(i)] = abs(float(err)) + 1e-6

    def __len__(self):
        return len(self.data)

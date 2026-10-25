from collections import deque

import numpy as np


class PrioritizedReplayBuffer:
    """Replay priorizado rank-based, sem α/β nem annealing de pesos.

    `priorities` anda em paralelo com `data` (mesmo `maxlen`): a posição `i`
    é a chave usada por `update_priorities`. A amostragem é proporcional a
    `1/rank(prioridade)` — rank 1 é a maior prioridade —, e novas transições
    entram com a maior prioridade atual para serem visitadas logo.

    Guarda `terminated` (não `truncated`): truncamento por tempo não é estado
    terminal, então o alvo continua bootstrapping na última transição.
    """
    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated,
            priority=None):
        """Guarda a transição e sua prioridade (padrão: a maior atual)."""
        self.data.append((np.array(state, copy=True), int(action), float(reward),
                          np.array(next_state, copy=True), bool(terminated)))
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(float(priority))

    def add_batch(self, transitions):
        """Adiciona em ordem; cada item é uma 5-tupla aceita por `add`."""
        for transition in transitions:
            self.add(*transition)

    def sample(self, batch_size):
        """Sorteia por rank e devolve `(s, a, r, ns, term, indices)`.

        Os pesos são `1/rank` sobre as prioridades do snapshot, normalizados;
        o sorteio é com reposição e usa o RNG global `np.random` (semeado
        pelo `train.py`). `indices` são as posições no snapshot, para
        `update_priorities` depois do erro de TD.
        """
        data = list(self.data)
        prios = np.asarray(self.priorities, dtype=float)
        order = np.argsort(-prios, kind="stable")
        ranks = np.empty(len(prios), dtype=np.int64)
        ranks[order] = np.arange(1, len(prios) + 1)
        weights = 1.0 / ranks
        probs = weights / weights.sum()
        idxs = np.random.choice(len(data), size=batch_size, replace=True, p=probs)
        batch = [data[i] for i in idxs]
        s, a, r, ns, term = zip(*batch)
        return (np.stack(s), np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32), np.stack(ns),
                np.asarray(term, dtype=np.float32),
                np.asarray(idxs, dtype=np.int64))

    def update_priorities(self, indices, errors):
        """Reescreve as prioridades: `prioridade[i] = |erro| + 1e-6`.

        Invariante: os índices só valem enquanto não houver `add` entre o
        `sample` e o `update_priorities` (o `learn()` garante isso).
        """
        for i, err in zip(indices, errors):
            self.priorities[int(i)] = abs(float(err)) + 1e-6

    def __len__(self):
        return len(self.data)

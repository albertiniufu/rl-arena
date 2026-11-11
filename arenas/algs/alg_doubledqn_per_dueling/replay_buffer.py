from collections import deque

import numpy as np


class PrioritizedReplayBuffer:
    """Replay priorizado rank-based, com α e pesos de importância (β).

    `priorities` anda em paralelo com `data` (mesmo `maxlen`): a posição `i`
    é a chave usada por `update_priorities`. A amostragem é proporcional a
    `1/rank(prioridade)^α` — rank 1 é a maior prioridade —, e novas transições
    entram com a maior prioridade atual para serem visitadas logo. Com α = 1
    o topo do rank domina (num buffer de 20 mil, as 10 primeiras posições
    recebem ~28% dos sorteios e o tamanho efetivo da amostra fica em ~67
    transições); α < 1 espalha a amostragem.

    Cada amostra volta com um peso de importância `(N·p)^-β` normalizado pelo
    máximo, que corrige o viés de amostragem no loss. O agente agenda β de
    `beta_start` até 1 ao longo do treino.

    Guarda `terminated` (não `truncated`): truncamento por tempo não é estado
    terminal, então o alvo continua bootstrapping na última transição.
    """
    def __init__(self, capacity=50000, alpha=0.6):
        self.data = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = float(alpha)

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

    def sample(self, batch_size, alpha=None, beta=0.4):
        """Sorteia por rank e devolve `(s, a, r, ns, term, indices, pesos)`.

        As probabilidades são `rank^-α` (α vem do construtor quando omitido),
        normalizadas; o sorteio é com reposição e usa o RNG global
        `np.random` (semeado pelo `train.py`). `indices` são as posições no
        snapshot, para `update_priorities` depois do erro de TD, e `pesos`
        são os pesos de importância `(N·p)^-β` normalizados pelo máximo
        (todos iguais a 1 quando β=0).
        """
        data = list(self.data)
        prios = np.asarray(self.priorities, dtype=float)
        alpha = self.alpha if alpha is None else float(alpha)
        order = np.argsort(-prios, kind="stable")
        ranks = np.empty(len(prios), dtype=np.int64)
        ranks[order] = np.arange(1, len(prios) + 1)
        probs = ranks.astype(float) ** (-alpha)
        probs /= probs.sum()
        idxs = np.random.choice(len(data), size=batch_size, replace=True, p=probs)
        weights = (len(data) * probs[idxs]) ** (-beta)
        weights /= weights.max()
        batch = [data[i] for i in idxs]
        s, a, r, ns, term = zip(*batch)
        return (np.stack(s), np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32), np.stack(ns),
                np.asarray(term, dtype=np.float32),
                np.asarray(idxs, dtype=np.int64),
                np.asarray(weights, dtype=np.float32))

    def update_priorities(self, indices, errors):
        """Reescreve as prioridades: `prioridade[i] = |erro| + 1e-6`.

        Invariante: os índices só valem enquanto não houver `add` entre o
        `sample` e o `update_priorities` (o `learn()` garante isso).
        """
        for i, err in zip(indices, errors):
            self.priorities[int(i)] = abs(float(err)) + 1e-6

    def __len__(self):
        return len(self.data)

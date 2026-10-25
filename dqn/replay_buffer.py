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
        """Sorteia transições e devolve retornos n-step `(s, a, r_n, ns, term)`.

        Para cada índice sorteado, acumula `r_n = Σ_{k=0..n-1} γ^k·r_{t+k}`
        caminhando até `n` posições à frente no deque; a janela para na
        fronteira de episódio (`terminated` ou `truncated`) e `ns`/`term` vêm
        da transição onde parou. Com `n=1`, o comportamento é o histórico:
        `r_n == r`, `ns == next_state` e `term == terminated`.
        """
        idxs = random.sample(range(len(self.data)), batch_size)
        s, a, r_n, ns, term = [], [], [], [], []
        for i in idxs:
            state, action = self.data[i][0], self.data[i][1]
            reward = 0.0
            for k in range(n):
                if i + k >= len(self.data):
                    break  # fim do buffer: fica com a última transição visitada
                _, _, r, next_state, terminated, truncated = self.data[i + k]
                reward += (gamma ** k) * r
                if terminated or truncated:
                    break  # não atravessa fronteira de episódio
            s.append(state)
            a.append(action)
            r_n.append(reward)
            ns.append(next_state)
            term.append(terminated)
        return (np.stack(s), np.asarray(a), np.asarray(r_n, dtype=np.float32),
                np.stack(ns), np.asarray(term, dtype=np.float32))
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

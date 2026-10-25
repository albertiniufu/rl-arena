from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """Replay uniforme: toda transição tem a mesma chance de ser sorteada.

    Guarda `terminated` (não `truncated`): truncamento por tempo não é estado
    terminal, então o alvo continua bootstrapping na última transição.
    """
    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated):
        self.data.append((np.array(state, copy=True), int(action), float(reward),
                          np.array(next_state, copy=True), bool(terminated)))

    def add_batch(self, transitions):
        """Adiciona em ordem; cada item é uma 5-tupla aceita por `add`."""
        for transition in transitions:
            self.add(*transition)

    def sample(self, batch_size):
        """Sorteia `batch_size` transições com chance igual.

        Devolve `(s, a, r, ns, term)`; o bootstrap usa `term == 1` apenas
        para terminais verdadeiros.
        """
        # Uma cópia em lista por amostragem: indexar o deque no meio é O(n),
        # e o buffer cheio tem dezenas de milhares de transições.
        data = list(self.data)
        idxs = random.sample(range(len(data)), batch_size)
        batch = [data[i] for i in idxs]
        s, a, r, ns, term = zip(*batch)
        return (np.stack(s), np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32), np.stack(ns),
                np.asarray(term, dtype=np.float32))

    def __len__(self):
        return len(self.data)

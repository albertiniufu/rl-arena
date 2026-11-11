from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """Replay uniforme: toda transição tem a mesma chance de ser sorteada.

    Guarda `terminated` (não `truncated`): truncamento por tempo não é estado
    terminal, então o alvo continua bootstrapping na última transição.

    Ambientes com máscara de ações (risk) podem passar `next_mask`; o DQN usa
    a máscara no alvo para não superestimar ações inválidas no próximo estado.
    """
    accepts_masks = True

    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)
        self.masks = deque(maxlen=capacity)
        self._has_masks = False

    def add(self, state, action, reward, next_state, terminated,
            next_mask=None):
        self.data.append((np.array(state, copy=True), int(action), float(reward),
                          np.array(next_state, copy=True), bool(terminated)))
        self.masks.append(None if next_mask is None
                          else np.asarray(next_mask, dtype=bool))
        if next_mask is not None:
            self._has_masks = True

    def add_batch(self, transitions):
        """Adiciona em ordem; cada item é aceito por `add`."""
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

    def has_masks(self) -> bool:
        return self._has_masks

    def sample_with_masks(self, batch_size):
        """Como `sample`, mas devolve também a máscara do próximo estado."""
        data = list(self.data)
        masks = list(self.masks)
        idxs = random.sample(range(len(data)), batch_size)
        batch = [data[i] for i in idxs]
        mask_batch = [masks[i] for i in idxs]
        s, a, r, ns, term = zip(*batch)

        known = next((mask for mask in mask_batch if mask is not None), None)
        n_actions = len(known) if known is not None else 0
        filled = [
            mask if mask is not None else np.ones(n_actions, dtype=bool)
            for mask in mask_batch
        ]
        next_masks = (np.stack(filled) if n_actions
                      else np.ones((len(batch), 0), dtype=bool))
        return (np.stack(s), np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32), np.stack(ns),
                np.asarray(term, dtype=np.float32), next_masks)

    def __len__(self):
        return len(self.data)

"""Buffer de rollout n-step do A2C."""
import numpy as np


class RolloutBuffer:
    """Transições na ordem da coleta, com marca de fim de episódio.

    O A2C consome o buffer quando ele atinge `n_steps` e o descarta. Não há
    amostragem aleatória: os retornos n-step são calculados sobre a sequência
    contígua, usando a marca de fim de episódio para não atravessar episódios.
    """
    def __init__(self):
        self.data = []

    def add(self, state, action, reward, next_state, terminated):
        self.data.append((
            np.array(state, copy=True), int(action), float(reward),
            np.array(next_state, copy=True), bool(terminated), False,
        ))

    def add_batch(self, transitions):
        """Adiciona em ordem; cada item é uma 5-tupla aceita por `add`."""
        for transition in transitions:
            self.add(*transition)

    def mark_episode_end(self):
        """Marca a última transição como fronteira de episódio."""
        if not self.data:
            return
        state, action, reward, next_state, terminated, _ = self.data[-1]
        self.data[-1] = (state, action, reward, next_state, terminated, True)

    def as_arrays(self):
        """Devolve arrays `(s, a, r, ns, terminated, episode_end)`."""
        if not self.data:
            empty = np.empty((0,), dtype=np.float32)
            return (empty, np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32),
                    empty, np.empty((0,), dtype=bool),
                    np.empty((0,), dtype=bool))
        s, a, r, ns, term, end = zip(*self.data)
        return (np.stack(s), np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32), np.stack(ns),
                np.asarray(term, dtype=bool), np.asarray(end, dtype=bool))

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)

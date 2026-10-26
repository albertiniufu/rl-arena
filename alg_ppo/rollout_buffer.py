"""Buffer de rollout on-policy do PPO."""
import numpy as np


class RolloutBuffer:
    """Transições na ordem em que foram coletadas, com marca de fim de episódio.

    Diferente do replay do DQN, não há amostragem aleatória nem capacidade
    circular: o PPO consome o rollout inteiro e o descarta depois do update.
    `mark_episode_end()` é chamado pelo `train.py` ao fim de cada episódio;
    o GAE usa essa marca para não propagar vantagem entre episódios, mas
    ainda faz bootstrap no truncamento por tempo.
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

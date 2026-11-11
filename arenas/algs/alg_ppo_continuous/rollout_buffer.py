"""Buffer de rollout on-policy para ações contínuas."""
import numpy as np


class ContinuousRolloutBuffer:
    """Transições na ordem da coleta, com marca de fim de episódio."""

    def __init__(self):
        self.data = []

    def add(self, state, action, reward, next_state, terminated):
        self.data.append((
            np.array(state, copy=True),
            np.asarray(action, dtype=np.float32).copy(),
            float(reward),
            np.array(next_state, copy=True),
            bool(terminated),
            False,
        ))

    def mark_episode_end(self):
        if not self.data:
            return
        state, action, reward, next_state, terminated, _ = self.data[-1]
        self.data[-1] = (
            state, action, reward, next_state, terminated, True)

    def as_arrays(self):
        """Arrays `(s, a, r, ns, terminated, episode_end)`."""
        if not self.data:
            empty = np.empty((0,), dtype=np.float32)
            return (empty, np.empty((0, 0), dtype=np.float32),
                    empty, empty,
                    np.empty((0,), dtype=bool),
                    np.empty((0,), dtype=bool))
        s, a, r, ns, term, end = zip(*self.data)
        return (np.stack(s), np.stack(a), np.asarray(r, dtype=np.float32),
                np.stack(ns), np.asarray(term, dtype=bool),
                np.asarray(end, dtype=bool))

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)

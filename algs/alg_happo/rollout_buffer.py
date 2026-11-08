"""Buffer de rollout do HAPPO: um registro por passo conjunto.

Idêntico em contrato ao buffer do MAPPO: `train.py` chama
`add_transition(i, ...)` uma vez por agente no mesmo passo do ambiente e o
buffer só grava quando todos os agentes contribuíram. O crítico centralizado
e o cálculo do produto acumulado de razões do update sequencial precisam das
observações/ações de todos os agentes alinhadas no tempo.

O HAPPO não precisa guardar `log_prob` na coleta: os pesos dos atores não
mudam durante o rollout, então o update recalcula os log-probs antigos de
forma exata antes de começar a atualização sequencial.
"""
import numpy as np


class HAPPORolloutBuffer:
    """Rollout on-policy indexado por passo conjunto."""

    def __init__(self, n_agents):
        self.n_agents = int(n_agents)
        self._pending = {}
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._terminated = []
        self._ends = []

    def add(self, agent_index, state, action, reward, next_state, terminated,
            log_prob=None):
        index = int(agent_index)
        if not 0 <= index < self.n_agents:
            raise ValueError(
                f"índice de agente inválido: {index} "
                f"(esperado 0..{self.n_agents - 1})")
        if index in self._pending:
            raise ValueError(
                f"agente {index} adicionado duas vezes no mesmo passo "
                f"conjunto; complete os {self.n_agents} agentes antes de "
                f"recomeçar")
        self._pending[index] = (
            np.array(state, copy=True), int(action), float(reward),
            np.array(next_state, copy=True), bool(terminated),
        )
        if len(self._pending) == self.n_agents:
            self._commit()

    def _commit(self):
        order = range(self.n_agents)
        self._states.append(np.stack([self._pending[i][0] for i in order]))
        self._actions.append([self._pending[i][1] for i in order])
        self._rewards.append([self._pending[i][2] for i in order])
        self._next_states.append(
            np.stack([self._pending[i][3] for i in order]))
        self._terminated.append([self._pending[i][4] for i in order])
        self._ends.append([False] * self.n_agents)
        self._pending.clear()

    def mark_episode_end(self, agent_index):
        index = int(agent_index)
        if not 0 <= index < self.n_agents:
            raise ValueError(f"índice de agente inválido: {index}")
        if self._ends:
            self._ends[-1][index] = True

    def as_arrays(self):
        """Arrays do rollout ou `None` se não há passo conjunto completo.

        Formas: `states`/`next_states` `(T, n, C, H, W)`; `actions`,
        `rewards`, `terminated` e `episode_ends` `(T, n)`.
        """
        if not self._states:
            return None
        return {
            "states": np.stack(self._states).astype(np.float32),
            "actions": np.asarray(self._actions, dtype=np.int64),
            "rewards": np.asarray(self._rewards, dtype=np.float32),
            "next_states": np.stack(self._next_states).astype(np.float32),
            "terminated": np.asarray(self._terminated, dtype=bool),
            "episode_ends": np.asarray(self._ends, dtype=bool),
        }

    def clear(self):
        self._pending.clear()
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._next_states.clear()
        self._terminated.clear()
        self._ends.clear()

    @property
    def n_steps(self):
        return len(self._states)

    def __len__(self):
        return len(self._states) * self.n_agents + len(self._pending)

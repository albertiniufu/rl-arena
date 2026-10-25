import numpy as np
import pytest

from dqn.agent import DQNAgent
from train import epsilon_for_episode


def test_epsilon_for_episode():
    # Rampa linear de 1.0 no episódio 1 até o piso no episódio `ramp_end`.
    assert epsilon_for_episode(1, 60, 0.1) == pytest.approx(1.0 - (1 - 0.1) / 60 * 1, abs=1e-9)
    assert epsilon_for_episode(60, 60, 0.1) == pytest.approx(0.1)
    assert epsilon_for_episode(61, 60, 0.1) == pytest.approx(0.1)


def test_warmup():
    agent = DQNAgent((5, 32, 20), 12, warmup=1000)
    for _ in range(999):
        agent.buffer.add(np.zeros((5, 32, 20)), 0, 0.0, np.zeros((5, 32, 20)), False, False)
    assert agent.learn() is None
    agent.buffer.add(np.zeros((5, 32, 20)), 0, 0.0, np.zeros((5, 32, 20)), False, False)
    assert agent.learn() is not None  # 1000 transições


def test_learn_does_not_decay_epsilon():
    """O epsilon é agendado por episódio em train.py; o agente só o lê."""
    agent = DQNAgent((5, 32, 20), 12, epsilon=0.5)
    zeros = np.zeros((5, 32, 20), dtype=np.float32)
    for _ in range(2000):
        agent.buffer.add(zeros, 0, 0.0, zeros, False, False)
    assert agent.learn() is not None  # buffer cheio: o passo de aprendizado roda
    assert agent.epsilon == 0.5

"""Testes dos algoritmos contínuos off-policy: DDPG, TD3 e SAC."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from arenas.algs.alg_ddpg.agent import DDPGAgent
from arenas.algs.alg_sac.agent import SACAgent
from arenas.algs.alg_td3.agent import TD3Agent
from arenas.algs.alg_registry import load_checkpoint, save_checkpoint
from arenas.common.agent_factory import build_agent

OBS = (4,)
ACTIONS = 2
LOW = np.array([-1.0, 0.0], dtype=np.float32)
HIGH = np.array([1.0, 2.0], dtype=np.float32)


def _fill(agent, n=16):
    rng = np.random.default_rng(1)
    for _ in range(n):
        state = rng.normal(size=OBS).astype(np.float32)
        action = rng.uniform(LOW, HIGH).astype(np.float32)
        next_state = rng.normal(size=OBS).astype(np.float32)
        agent.buffer.add(state, action, float(rng.normal()),
                         next_state, False)


@pytest.mark.parametrize("cls", [DDPGAgent, TD3Agent, SACAgent])
def test_continuous_agent_learns_acts_and_checkpoints(cls, tmp_path):
    agent = cls(OBS, ACTIONS, batch_size=8, warmup=0, buffer_size=100)
    agent.set_action_bounds(LOW, HIGH)
    _fill(agent)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    action = agent.act(np.zeros(OBS, dtype=np.float32))
    assert action.shape == (ACTIONS,)
    assert np.all(action >= LOW) and np.all(action <= HIGH)
    batch = agent.act_batch(np.zeros((3, *OBS), dtype=np.float32))
    assert batch.shape == (3, ACTIONS)
    assert np.all(batch >= LOW) and np.all(batch <= HIGH)
    q_mean, q_max = agent.q_stats(np.zeros(OBS, dtype=np.float32))
    assert np.isfinite(q_mean) and np.isfinite(q_max)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, agent, config=None)
    fresh = cls(OBS, ACTIONS, batch_size=8, warmup=0)
    fresh.set_action_bounds(LOW, HIGH)
    load_checkpoint(path, fresh, resume=True)
    assert fresh.steps == agent.steps
    assert all(torch.equal(a, b) for a, b in zip(
        fresh.online["critic1" if "critic1" in fresh.online else "critic"].parameters(),
        agent.online["critic1" if "critic1" in agent.online else "critic"].parameters()))


def test_build_agent_accepts_continuous_algorithms():
    action_space = spaces.Box(LOW, HIGH, dtype=np.float32)
    for name in ("ddpg", "td3", "sac"):
        agent = build_agent(name, OBS, action_space, batch_size=8)
        assert agent.continuous is True
        assert agent.n_actions == ACTIONS

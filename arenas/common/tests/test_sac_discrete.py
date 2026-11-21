"""Testes do SAC discreto no contrato do seek_coin."""
import numpy as np
import torch

from arenas.algs.alg_registry import load_checkpoint, save_checkpoint
from arenas.algs.alg_sac_discrete.agent import SACDiscreteAgent

OBS = (5, 20, 32)


def _fill(agent, n=16):
    rng = np.random.default_rng(2)
    for _ in range(n):
        state = rng.random(OBS).astype(np.float32)
        next_state = rng.random(OBS).astype(np.float32)
        agent.buffer.add(state, int(rng.integers(6)), float(rng.normal()),
                         next_state, False)


def test_sac_discrete_learns_and_acts():
    agent = SACDiscreteAgent(OBS, 6, batch_size=8, warmup=0)
    _fill(agent)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    action = agent.act(np.zeros(OBS, dtype=np.float32))
    assert 0 <= action < 6
    batch = agent.act_batch(np.zeros((3, *OBS), dtype=np.float32))
    assert batch.shape == (3,)
    mask = np.array([True, True, False, False, True, True])
    assert mask[agent.act(np.zeros(OBS, dtype=np.float32), action_mask=mask)]


def test_sac_discrete_checkpoint(tmp_path):
    agent = SACDiscreteAgent(OBS, 6, batch_size=8, warmup=0)
    _fill(agent)
    agent.learn()
    path = tmp_path / "sac_discrete.pt"
    save_checkpoint(path, agent, config=None)
    fresh = SACDiscreteAgent(OBS, 6, batch_size=8, warmup=0)
    load_checkpoint(path, fresh, resume=True)
    assert fresh.steps == agent.steps
    assert all(torch.equal(a, b) for a, b in zip(
        fresh.online["actor"].parameters(), agent.online["actor"].parameters()))

"""Testes do ppo_aes (Adaptive Entropy Scheduling, arXiv:2601.19624)."""
import math

import numpy as np
import pytest
import torch

from arenas.algs.alg_ppo_aes.aes import AESScheduler
from arenas.algs.alg_ppo_aes.agent import PPOAESAgent
from arenas.algs.alg_registry import (
    algorithm_from_checkpoint,
    available_algorithms,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)

OBS_SHAPE = (5, 20, 32)


def test_ppo_aes_is_discovered_by_the_cli_registry():
    assert "ppo_aes" in available_algorithms()
    assert load_algorithm("ppo_aes").Agent is PPOAESAgent


def test_aes_first_update_uses_quantile_square_root_and_clips():
    aes = AESScheduler()
    # quantil 0,9 de [1,2,3,4] = 3,7; sqrt(3,7) = 1,92 > lambda_max=0,1.
    value = aes.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert value == pytest.approx(0.1)
    assert aes.ema == pytest.approx(3.7)
    assert aes.accumulated == pytest.approx(3.7)
    assert aes.step == 1


def test_aes_ema_and_running_mean_follow_the_official_rule():
    aes = AESScheduler(beta=0.95, lambda_max=1.0)
    assert aes.beta == 0.95
    first = aes.update(torch.full((4,), 0.001))
    assert first == pytest.approx(math.sqrt(0.001), rel=1e-6)
    second = aes.update(torch.zeros(4))
    # EMA = 0,95·0,001; acumulado = 0,001 + 0,00095; média sobre 2 passos.
    expected = math.sqrt((0.001 + 0.95 * 0.001) / 2.0)
    assert second == pytest.approx(expected, rel=1e-6)


def test_aes_kappa_scales_the_schedule_and_lambda_min_clips():
    aes = AESScheduler(kappa=2.0, lambda_max=1.0)
    assert aes.update(torch.full((4,), 0.001)) == pytest.approx(
        2.0 * math.sqrt(0.001), rel=1e-6)
    zeros = AESScheduler()
    assert zeros.update(torch.zeros(4)) == pytest.approx(zeros.lambda_min)


def test_aes_rejects_invalid_configuration_and_residuals():
    with pytest.raises(ValueError):
        AESScheduler(q=1.5)
    with pytest.raises(ValueError):
        AESScheduler(beta=1.0)
    with pytest.raises(ValueError):
        AESScheduler(lambda_min=0.0)
    aes = AESScheduler()
    with pytest.raises(ValueError):
        aes.update(torch.empty(0))
    with pytest.raises(ValueError):
        aes.update(torch.tensor([float("nan")]))


def test_aes_state_dict_round_trip():
    aes = AESScheduler()
    aes.update(torch.tensor([0.0, 0.01, 0.02]))
    state = aes.state_dict()
    fresh = AESScheduler()
    fresh.load_state_dict(state)
    assert fresh.ema == pytest.approx(aes.ema)
    assert fresh.accumulated == pytest.approx(aes.accumulated)
    assert fresh.step == aes.step
    assert fresh.value == pytest.approx(aes.value)
    with pytest.raises(ValueError):
        AESScheduler(lambda_max=0.5).load_state_dict(state)


def test_ppo_aes_agent_waits_for_rollout_then_updates_with_scheduled_entropy():
    agent = PPOAESAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=4,
                        epochs=1)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for i in range(3):
        agent.buffer.add(obs, i % 6, 0.1, obs, False)
        assert agent.learn() is None
    agent.buffer.add(obs, 0, 0.1, obs, False)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert len(agent.buffer) == 0
    assert agent.steps == 1
    assert agent.aes.step == 2  # um update por minibatch
    assert agent.aes.lambda_min <= agent.entropy_coef <= agent.aes.lambda_max


def test_ppo_aes_agent_act_contract_and_checkpoint(tmp_path):
    agent = PPOAESAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=2,
                        epochs=1)
    state = np.zeros(OBS_SHAPE, dtype=np.float32)
    assert 0 <= agent.act(state) < 6
    assert 0 <= agent.act(state, greedy=True) < 6
    assert agent.act_batch(np.stack([state, state])).shape == (2,)
    assert agent.on_policy is True and agent.target is agent.online
    assert agent.epsilon == 0.0
    q_mean, q_max = agent.q_stats(state)
    assert np.isfinite(q_mean) and np.isfinite(q_max)
    assert agent.head_weights().shape[0] == 6

    path = tmp_path / "ppo_aes.pt"
    save_checkpoint(path, agent, episodes=3)
    assert algorithm_from_checkpoint(path)[1]["algorithm"] == "ppo_aes"
    fresh = PPOAESAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=2,
                        epochs=1)
    load_checkpoint(path, fresh, resume=False)
    for p1, p2 in zip(agent.online.parameters(), fresh.online.parameters()):
        assert torch.equal(p1, p2)

"""Testes do GTR (Gaussian-Reshaped Trust Region, arXiv:2606.03382)."""
import math

import numpy as np
import pytest
import torch

from arenas.algs.alg_gtr.agent import GTRAgent, gaussian_penalty
from arenas.algs.alg_registry import (
    algorithm_from_checkpoint,
    available_algorithms,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)

OBS_SHAPE = (5, 20, 32)


def test_gtr_is_discovered_by_the_cli_registry():
    assert "gtr" in available_algorithms()
    assert load_algorithm("gtr").Agent is GTRAgent


def test_gaussian_penalty_is_zero_at_one_and_bounded():
    zero = torch.zeros(1)
    # r = 1 => sem penalidade (política não mudou).
    assert gaussian_penalty(zero, [zero], zero).item() == pytest.approx(0.0)
    # r = 2 => ½(1 − e^{−1}).
    log_two = torch.tensor([math.log(2.0)])
    assert gaussian_penalty(
        log_two, [zero], zero).item() == pytest.approx(
            0.5 * (1.0 - math.exp(-1.0)), rel=1e-6)
    # r → 0 e r → ∞ são limitados: ½(1−e^{−1}) e ½, com gradiente nulo
    # nos extremos ("esquecimento" e "liberação").
    very_small = torch.tensor([-30.0])
    assert gaussian_penalty(
        very_small, [zero], zero).item() == pytest.approx(
            0.5 * (1.0 - math.exp(-1.0)), rel=1e-6)
    assert gaussian_penalty(
        -very_small, [zero], zero).item() == pytest.approx(0.5, abs=1e-6)


def test_mixture_anchor_averages_the_importance_weighted_penalties():
    current = torch.zeros(1)
    behavior = torch.zeros(1)
    anchor_a = torch.zeros(1)                       # π_i = μ
    anchor_b = torch.tensor([math.log(2.0)])        # π_i/μ = 2
    # âncora A: razões 1 e 1 -> 0; âncora B: r_i/μ=2, πθ/π_i=0,5.
    expected = 0.5 * (2.0 * 0.5 * (1.0 - math.exp(-0.25)))
    value = gaussian_penalty(
        current, [anchor_a, anchor_b], behavior).item()
    assert value == pytest.approx(expected, rel=1e-6)


def test_gaussian_penalty_gradient_is_finite_for_extreme_ratios():
    log_probs = torch.tensor([50.0, -50.0, 0.0], requires_grad=True)
    behavior = torch.zeros(3)
    penalty = gaussian_penalty(log_probs, [behavior], behavior).mean()
    penalty.backward()
    assert torch.isfinite(log_probs.grad).all()


def test_gtr_agent_waits_for_rollout_then_updates_with_anchors():
    agent = GTRAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=4,
                     epochs=3, anchor_count=2)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for i in range(3):
        agent.buffer.add(obs, i % 6, 0.1, obs, False)
        assert agent.learn() is None
    agent.buffer.add(obs, 0, 0.1, obs, False)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert np.isfinite(agent.last_penalty) and agent.last_penalty >= 0.0
    assert len(agent.buffer) == 0
    assert agent.steps == 1
    assert agent.anchor_count == 2


def test_gtr_agent_act_contract_and_checkpoint(tmp_path):
    agent = GTRAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=2, epochs=1)
    state = np.zeros(OBS_SHAPE, dtype=np.float32)
    assert 0 <= agent.act(state) < 6
    assert 0 <= agent.act(state, greedy=True) < 6
    assert agent.act_batch(np.stack([state, state])).shape == (2,)
    assert agent.on_policy is True and agent.target is agent.online
    assert agent.epsilon == 0.0
    q_mean, q_max = agent.q_stats(state)
    assert np.isfinite(q_mean) and np.isfinite(q_max)
    assert agent.head_weights().shape[0] == 6

    path = tmp_path / "gtr.pt"
    save_checkpoint(path, agent, episodes=3)
    assert algorithm_from_checkpoint(path)[1]["algorithm"] == "gtr"
    fresh = GTRAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=2, epochs=1)
    load_checkpoint(path, fresh, resume=False)
    for p1, p2 in zip(agent.online.parameters(), fresh.online.parameters()):
        assert torch.equal(p1, p2)

"""Testes do EPPO (crítico evidencial NIG + vantagem UCB)."""
import numpy as np
import pytest
import torch

from arenas.algs.alg_eppo.agent import (
    EPPOAgent,
    compute_gae_with_uncertainty,
    evidential_nll,
    evidential_prior_log_prob,
)
from arenas.algs.alg_eppo.model import (
    EvidentialActorCriticNetwork,
    evidential_moments,
    evidential_parameters,
)
from arenas.algs.alg_ppo.agent import compute_gae
from arenas.algs.alg_registry import (
    algorithm_from_checkpoint,
    available_algorithms,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)

OBS_SHAPE = (5, 20, 32)


def test_eppo_is_discovered_by_the_cli_registry():
    assert "eppo" in available_algorithms()
    assert load_algorithm("eppo").Agent is EPPOAgent


def test_evidential_critic_outputs_four_valid_parameters():
    net = EvidentialActorCriticNetwork(OBS_SHAPE, 6)
    logits, raw = net(torch.zeros(3, *OBS_SHAPE))
    assert logits.shape == (3, 6)
    assert raw.shape == (3, 4)
    omega, nu, alpha, beta = evidential_parameters(raw)
    assert omega.shape == (3,)
    assert torch.all(nu > 0) and torch.all(alpha > 1) and torch.all(beta > 0)
    mean, variance = evidential_moments((omega, nu, alpha, beta))
    assert torch.equal(mean, omega)
    assert torch.all(variance > 0) and torch.all(torch.isfinite(variance))


def test_evidential_moments_match_the_nig_formula():
    omega = torch.tensor([2.0])
    nu = torch.tensor([4.0])
    alpha = torch.tensor([3.0])
    beta = torch.tensor([2.0])
    mean, variance = evidential_moments((omega, nu, alpha, beta))
    # Var = β/(α−1)·(1 + 1/ν) = 2/2·(1 + 1/4)
    assert torch.allclose(mean, torch.tensor([2.0]))
    assert torch.allclose(variance, torch.tensor([1.25]))


def test_evidential_nll_prefers_the_predicted_mean():
    omega = torch.zeros(2)
    nu = torch.ones(2)
    alpha = torch.full((2,), 2.0)
    beta = torch.ones(2)
    near = evidential_nll(torch.zeros(2), omega, nu, alpha, beta)
    far = evidential_nll(torch.full((2,), 10.0), omega, nu, alpha, beta)
    assert torch.isfinite(near) and torch.isfinite(far)
    assert near < far


def test_evidential_prior_is_maximal_near_the_hyperprior_modes():
    def logprob(nu, alpha, beta):
        return float(evidential_prior_log_prob(
            torch.zeros(1), torch.tensor([nu]), torch.tensor([alpha]),
            torch.tensor([beta])))

    # Modas: ω=0, ν=4, α−1=4, β=4.
    near = logprob(4.0, 5.0, 4.0)
    far = logprob(1e3, 1.001, 1e3)
    assert np.isfinite(near) and np.isfinite(far)
    assert near > far


def test_eppo_gae_mean_matches_ppo_when_uncertainty_is_zero():
    rewards = torch.tensor([1.0, 0.5, -0.2])
    values = torch.tensor([0.5, 1.0, 0.0])
    next_values = torch.tensor([1.0, 0.0, 2.0])
    terminated = torch.tensor([0.0, 0.0, 0.0])
    ends = torch.tensor([0.0, 0.0, 1.0])
    zeros = torch.zeros(3)
    mean_adv, variances, returns = compute_gae_with_uncertainty(
        rewards, values, next_values, zeros, zeros, terminated, ends,
        gamma=0.9, gae_lambda=0.8)
    expected_adv, expected_returns = compute_gae(
        rewards, values, next_values, terminated, ends,
        gamma=0.9, gae_lambda=0.8)
    assert torch.allclose(mean_adv, expected_adv)
    assert torch.allclose(returns, expected_returns)
    assert torch.equal(variances, zeros)


@pytest.mark.parametrize("variant", ["cor", "ind"])
def test_eppo_variance_propagation_matches_the_paper(variant):
    # γ=0.9, λ=0.5 => c_acc=(γλ)²=0.2025, c_next=1, c_var=1/3.
    rewards = torch.zeros(2)
    values = torch.zeros(2)
    next_values = torch.zeros(2)
    terminated = torch.zeros(2)
    ends = torch.zeros(2)
    values_var = torch.tensor([1.0, 2.0])
    next_values_var = torch.tensor([3.0, 4.0])
    _, variances, _ = compute_gae_with_uncertainty(
        rewards, values, next_values, values_var, next_values_var,
        terminated, ends, gamma=0.9, gae_lambda=0.5, variant=variant)
    # t=1: C_1 = c_acc·Var[V_2] = 0.81
    # t=0: C_0 = c_acc·(C_1 + Var[V_1]) = 0.2025·3.81 = 0.771525
    if variant == "cor":
        expected = torch.tensor([1.0 + 0.771525, 2.0 + 0.81])
    else:
        expected = torch.tensor([1.0 / 3.0 + 0.771525, 2.0 / 3.0 + 0.81])
    assert torch.allclose(variances, expected, atol=1e-6)


def test_eppo_variance_does_not_cross_episode_boundary():
    values_var = torch.tensor([1.0, 100.0])
    next_values_var = torch.tensor([3.0, 4.0])
    zeros = torch.zeros(2)
    _, variances, _ = compute_gae_with_uncertainty(
        zeros, zeros, zeros, values_var, next_values_var, zeros,
        torch.tensor([1.0, 0.0]), gamma=0.9, gae_lambda=0.5,
        variant="cor")
    # O passo 0 termina o episódio: só Var[V_1] do bootstrap entra, nada do
    # passo 1 (que pertence à trajetória seguinte).
    assert torch.allclose(
        variances[0], torch.tensor(1.0 + 0.2025 * 3.0), atol=1e-6)


def test_eppo_rejects_invalid_lambda():
    zeros = torch.zeros(1)
    with pytest.raises(ValueError):
        compute_gae_with_uncertainty(
            zeros, zeros, zeros, zeros, zeros, zeros, zeros,
            gamma=0.9, gae_lambda=0.5 * 2)


def test_eppo_agent_waits_for_rollout_then_updates():
    agent = EPPOAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=4,
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
    assert agent.last_grad_norm >= 0.0
    assert np.isfinite(agent.last_uncertainty) and agent.last_uncertainty >= 0.0


def test_eppo_agent_finish_training_flushes_partial_rollout():
    agent = EPPOAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=4,
                      epochs=1)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 0.0, obs, False)
    loss = agent.finish_training()
    assert loss is not None and np.isfinite(loss)
    assert agent.steps == 1
    assert len(agent.buffer) == 0


def test_eppo_agent_act_and_act_batch():
    agent = EPPOAgent(OBS_SHAPE, 6)
    state = np.zeros(OBS_SHAPE, dtype=np.float32)
    assert 0 <= agent.act(state) < 6
    assert 0 <= agent.act(state, greedy=True) < 6
    acts = agent.act_batch(np.stack([state, state]), greedy=False)
    assert acts.shape == (2,)
    assert np.issubdtype(acts.dtype, np.integer)
    assert np.array_equal(
        agent.act_batch(np.stack([state]), greedy=True),
        np.asarray([agent.act(state, greedy=True)]))


def test_eppo_agent_contract_attributes_and_checkpoint_round_trip(tmp_path):
    agent = EPPOAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=2,
                      epochs=1)
    assert agent.on_policy is True
    assert agent.target is agent.online
    assert agent.epsilon == 0.0
    assert agent.gamma == pytest.approx(0.99)
    q_mean, q_max = agent.q_stats(np.zeros(OBS_SHAPE, dtype=np.float32))
    assert np.isfinite(q_mean) and np.isfinite(q_max)
    assert agent.head_weights().shape[0] == 6

    path = tmp_path / "eppo.pt"
    save_checkpoint(path, agent, episodes=3)
    assert algorithm_from_checkpoint(path)[1]["algorithm"] == "eppo"
    fresh = EPPOAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=2,
                      epochs=1)
    load_checkpoint(path, fresh, resume=False)
    for p1, p2 in zip(agent.online.parameters(), fresh.online.parameters()):
        assert torch.equal(p1, p2)

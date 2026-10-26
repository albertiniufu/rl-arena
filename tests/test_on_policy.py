"""Testes dos algoritmos on-policy (PPO e A2C)."""
import numpy as np
import pytest
import torch

from alg_a2c.agent import A2CAgent, n_step_returns
from alg_a2c.model import ActorCriticNetwork as A2CNetwork
from alg_a2c.rollout_buffer import RolloutBuffer as A2CRolloutBuffer
from alg_ppo.agent import PPOAgent, compute_gae
from alg_ppo.model import ActorCriticNetwork as PPONetwork
from alg_ppo.rollout_buffer import RolloutBuffer as PPORolloutBuffer
from alg_registry import (
    algorithm_from_checkpoint,
    available_algorithms,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)


OBS_SHAPE = (5, 20, 32)


def test_actor_critic_networks_return_logits_and_value():
    for cls in (PPONetwork, A2CNetwork):
        net = cls(OBS_SHAPE, 6)
        logits, value = net(torch.zeros(2, *OBS_SHAPE))
        assert logits.shape == (2, 6)
        assert value.shape == (2,)
        assert hasattr(net, "features")
        assert hasattr(net, "actor")
        assert hasattr(net, "critic")


def test_rollout_buffers_store_episode_boundaries_and_clear():
    for cls in (PPORolloutBuffer, A2CRolloutBuffer):
        buf = cls()
        state = np.zeros(OBS_SHAPE, dtype=np.float32)
        buf.add(state, 1, 1.0, state, False)
        buf.add(state, 2, 2.0, state, False)
        buf.mark_episode_end()
        states, actions, rewards, next_states, terminated, episode_ends = buf.as_arrays()
        assert len(buf) == 2
        assert states.shape == (2, *OBS_SHAPE)
        assert actions.tolist() == [1, 2]
        assert rewards.tolist() == [1.0, 2.0]
        assert terminated.tolist() == [False, False]
        assert episode_ends.tolist() == [False, True]
        buf.clear()
        assert len(buf) == 0
        assert buf.as_arrays()[0].shape[0] == 0


def test_ppo_gae_stops_at_episode_end_but_bootstraps_truncation():
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.5])
    next_values = torch.tensor([9.0, 9.0])
    terminated = torch.tensor([0.0, 0.0])
    # Fim de episódio no primeiro passo: a vantagem não atravessa para o
    # episódio seguinte (o passo 1 começa uma nova trajetória).
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated,
        torch.tensor([1.0, 0.0]), gamma=0.9, gae_lambda=0.8)
    assert torch.allclose(advantages, torch.tensor([8.6, 8.6]))
    assert torch.allclose(returns, torch.tensor([9.1, 9.1]))
    # Sem fim de episódio, a vantagem do passo 0 incorpora a do passo 1.
    advantages, _ = compute_gae(
        rewards, values, next_values, terminated,
        torch.tensor([0.0, 0.0]), gamma=0.9, gae_lambda=0.8)
    assert torch.allclose(advantages, torch.tensor([14.792, 8.6]))


def test_ppo_gae_zeroes_bootstrap_on_true_terminal():
    advantages, returns = compute_gae(
        torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([10.0]),
        torch.tensor([1.0]), torch.tensor([1.0]), gamma=0.9, gae_lambda=0.8)
    assert torch.allclose(advantages, torch.tensor([1.0]))
    assert torch.allclose(returns, torch.tensor([1.0]))


def test_a2c_n_step_returns_bootstrap_at_truncation_and_stop_at_episode_end():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.0, 0.0, 0.0])
    next_values = torch.tensor([10.0, 20.0, 30.0])
    terminated = torch.tensor([0.0, 0.0, 0.0])
    episode_ends = torch.tensor([0.0, 0.0, 1.0])
    out = n_step_returns(
        rewards, values, next_values, terminated, episode_ends,
        gamma=0.9, n_steps=2)
    # i=0: r0 + γr1 + γ²V(s2); i=1: r1 + γr2 + γ²V(s3);
    # i=2: r2 + γV(s3) por ser fim de episódio truncado.
    assert torch.allclose(out, torch.tensor([1 + 1.8 + 16.2,
                                             2 + 2.7 + 24.3,
                                             3 + 27.0]))


def test_a2c_n_step_returns_do_not_bootstrap_true_terminal():
    out = n_step_returns(
        torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]),
        torch.tensor([10.0, 20.0]), torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]), gamma=0.9, n_steps=2)
    assert torch.allclose(out, torch.tensor([1 + 1.8, 2.0]))


@pytest.mark.parametrize("agent_cls, kwargs", [
    (PPOAgent, {"rollout_steps": 4, "epochs": 1}),
    (A2CAgent, {"n_steps": 4}),
])
def test_on_policy_agent_waits_for_rollout_then_updates(agent_cls, kwargs):
    rollout = kwargs.get("rollout_steps", kwargs.get("n_steps"))
    agent = agent_cls(OBS_SHAPE, 6, batch_size=2, **kwargs)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for i in range(rollout - 1):
        agent.buffer.add(obs, i % 6, 0.1, obs, False)
        assert agent.learn() is None
    agent.buffer.add(obs, 0, 0.1, obs, False)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert len(agent.buffer) == 0
    assert agent.steps == 1
    assert agent.last_grad_norm >= 0.0


@pytest.mark.parametrize("agent_cls", [PPOAgent, A2CAgent])
def test_on_policy_act_and_act_batch(agent_cls):
    agent = agent_cls(OBS_SHAPE, 6)
    state = np.zeros(OBS_SHAPE, dtype=np.float32)
    assert 0 <= agent.act(state) < 6
    assert 0 <= agent.act(state, greedy=True) < 6
    acts = agent.act_batch(np.stack([state, state]), greedy=False)
    assert acts.shape == (2,)
    assert np.issubdtype(acts.dtype, np.integer)
    assert np.array_equal(
        agent.act_batch(np.stack([state]), greedy=True),
        np.asarray([agent.act(state, greedy=True)]))


def test_ppo_end_episode_marks_boundary_without_updating():
    agent = PPOAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=4)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    agent.buffer.add(obs, 0, 0.0, obs, False)
    agent.buffer.add(obs, 1, 0.0, obs, False)
    assert agent.end_episode() is None
    assert agent.steps == 0
    assert len(agent.buffer) == 2
    assert agent.buffer.as_arrays()[5].tolist() == [False, True]


def test_ppo_finish_training_flushes_partial_rollout():
    agent = PPOAgent(OBS_SHAPE, 6, batch_size=2, rollout_steps=4, epochs=1)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 0.0, obs, False)
    loss = agent.finish_training()
    assert loss is not None and np.isfinite(loss)
    assert agent.steps == 1
    assert len(agent.buffer) == 0


def test_a2c_end_episode_flushes_partial_rollout():
    agent = A2CAgent(OBS_SHAPE, 6, batch_size=4)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for action in (0, 1):
        agent.buffer.add(obs, action, 0.0, obs, False)
    loss = agent.end_episode()
    assert loss is not None and np.isfinite(loss)
    assert agent.steps == 1
    assert len(agent.buffer) == 0


@pytest.mark.parametrize("agent_cls", [PPOAgent, A2CAgent])
def test_on_policy_agent_exposes_contract_attributes_and_target_alias(agent_cls):
    agent = agent_cls(OBS_SHAPE, 6)
    assert agent.on_policy is True
    assert agent.target is agent.online
    assert agent.epsilon == 0.0
    assert agent.gamma == pytest.approx(0.99)
    q_mean, q_max = agent.q_stats(np.zeros(OBS_SHAPE, dtype=np.float32))
    assert np.isfinite(q_mean) and np.isfinite(q_max)
    assert agent.head_weights().shape[0] == 6


@pytest.mark.parametrize("agent_cls, alloc", [
    (PPOAgent, {"rollout_steps": 2, "epochs": 1}),
    (A2CAgent, {"n_steps": 2}),
])
def test_on_policy_checkpoint_round_trip(tmp_path, agent_cls, alloc):
    agent = agent_cls(OBS_SHAPE, 6, batch_size=2, **alloc)
    path = tmp_path / "agent.pt"
    save_checkpoint(path, agent, episodes=3)
    meta = algorithm_from_checkpoint(path)[1]
    assert meta["algorithm"] == agent.algorithm
    fresh = agent_cls(OBS_SHAPE, 6, batch_size=2, **alloc)
    load_checkpoint(path, fresh, resume=False)
    for p1, p2 in zip(agent.online.parameters(), fresh.online.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.parametrize("agent_cls, alloc", [
    (PPOAgent, {"rollout_steps": 2, "epochs": 1}),
    (A2CAgent, {"n_steps": 2}),
])
def test_on_policy_resume_restores_counters_and_optimizer(tmp_path, agent_cls, alloc):
    agent = agent_cls(OBS_SHAPE, 6, batch_size=2, **alloc)
    agent.steps = 7
    path = tmp_path / "agent.pt"
    save_checkpoint(path, agent, episodes=5)
    fresh = agent_cls(OBS_SHAPE, 6, batch_size=2, **alloc)
    load_checkpoint(path, fresh, resume=True)
    assert fresh.steps == 7
    assert fresh.epsilon == 0.0
    # O otimizador restaurado precisa aceitar um update sem recriar estado.
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for _ in range(2):
        fresh.buffer.add(obs, 0, 0.0, obs, False)
    assert fresh.learn() is not None


def test_algorithm_discovery_includes_on_policy():
    names = available_algorithms()
    assert "ppo" in names and "a2c" in names
    assert load_algorithm("ppo").Agent is PPOAgent
    assert load_algorithm("a2c").Agent is A2CAgent


def test_train_cli_accepts_on_policy_algorithms():
    from train import parse_args

    assert parse_args(["--alg", "ppo"]).alg == "ppo"
    assert parse_args(["--alg", "a2c"]).alg == "a2c"


def test_vectorized_training_rejects_on_policy():
    from train import parse_args, validate_train_args

    for alg in ("ppo", "a2c"):
        with pytest.raises(SystemExit):
            validate_train_args(parse_args(
                ["--alg", alg, "--envs", "2", "--opponent", "rule"]))

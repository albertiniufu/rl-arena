"""Testes do MAPPO: crítico centralizado, buffer conjunto, update e CLI."""
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from arena.config import ArenaConfig
from arena.multi_environment import ArenaMultiEnv
from arena.scenario import Scenario
from algs.alg_ippo.agent import IPPOAgent
from algs.alg_mappo.agent import MAPPOAgent, compute_gae
from algs.alg_mappo.model import CentralizedActorCriticNetwork
from algs.alg_mappo.rollout_buffer import MultiAgentRolloutBuffer
from algs.alg_registry import (
    apply_env_config,
    available_algorithms,
    checkpoint_meta,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)

OBS_SHAPE = (6, 20, 32)


def make_agent(n_agents=2, **kwargs):
    params = dict(n_agents=n_agents, rollout_steps=2, batch_size=4,
                  epochs=1, device="cpu")
    params.update(kwargs)
    return MAPPOAgent(OBS_SHAPE, 6, **params)


def fill_step(agent, step=0, reward=0.0):
    """Adiciona as transições de um passo conjunto para todos os agentes."""
    obs = np.full((agent.n_agents, *OBS_SHAPE), step, dtype=np.float32)
    for i in range(agent.n_agents):
        agent.add_transition(i, obs[i], 0, reward, obs[i], False)


# ------------------------------------------------------------------ modelo
def test_mappo_critic_is_centralized():
    """V de um agente depende das observações dos outros agentes."""
    torch.manual_seed(0)
    net = CentralizedActorCriticNetwork(OBS_SHAPE, 6, n_agents=2)
    feats = torch.randn(1, 2, net.n_features)
    values = net.joint_values(feats)
    assert values.shape == (1, 2)

    changed = feats.clone()
    changed[:, 1] += 1.0
    values_changed = net.joint_values(changed)
    # O agente 0 vê o estado conjunto, então o V dele muda quando o agente 1
    # muda — a assinatura do crítico centralizado.
    assert not torch.allclose(values[:, 0], values_changed[:, 0])
    assert values.shape == net.joint_values(feats).shape

    logits = net(torch.zeros(2, *OBS_SHAPE))
    assert logits.shape == (2, 6)


# ------------------------------------------------------------------ buffer
def test_mappo_buffer_assembles_joint_steps():
    buffer = MultiAgentRolloutBuffer(3)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    assert buffer.as_arrays() is None

    buffer.add(0, obs, 1, 0.0, obs, False)
    assert buffer.n_steps == 0 and len(buffer) == 1
    # O mesmo agente não pode contribuir duas vezes no mesmo passo conjunto.
    with pytest.raises(ValueError):
        buffer.add(0, obs, 1, 0.0, obs, False)

    buffer.add(1, obs, 2, 1.0, obs, False)
    buffer.add(2, obs, 3, -1.0, obs, True)
    assert buffer.n_steps == 1 and len(buffer) == 3

    buffer.mark_episode_end(1)
    arrays = buffer.as_arrays()
    assert arrays["states"].shape == (1, 3, *OBS_SHAPE)
    assert arrays["actions"].tolist() == [[1, 2, 3]]
    assert arrays["rewards"].shape == (1, 3)
    assert arrays["episode_ends"][0].tolist() == [False, True, False]
    assert arrays["terminated"][0].tolist() == [False, False, True]

    with pytest.raises(ValueError):
        MultiAgentRolloutBuffer(2).add(2, obs, 0, 0.0, obs, False)
    with pytest.raises(ValueError):
        buffer.mark_episode_end(3)

    buffer.clear()
    assert len(buffer) == 0 and buffer.as_arrays() is None


def test_mappo_gae_respects_episode_boundary():
    rewards = torch.tensor([[0.0], [5.0]])
    values = torch.zeros(2, 1)
    next_values = torch.ones(2, 1)
    terminated = torch.zeros(2, 1, dtype=torch.bool)
    ends = torch.tensor([[True], [False]])
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, ends, gamma=.9,
        gae_lambda=1.0)
    # t=1 termina o episódio com retorno 5 + 0.9; em t=0 a vantagem não
    # pode carregar o futuro de outro episódio (só 0 + 0.9).
    assert advantages[1, 0] == pytest.approx(5.9)
    assert advantages[0, 0] == pytest.approx(0.9)
    assert returns.shape == (2, 1)


# ------------------------------------------------------------------ agente
def test_mappo_learn_waits_for_full_rollout_then_updates():
    agent = make_agent(n_agents=2, rollout_steps=3, batch_size=4)
    fill_step(agent, 0)
    fill_step(agent, 1)
    assert agent.learn() is None
    assert agent.steps == 0
    fill_step(agent, 2)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert agent.steps == 1
    assert len(agent.buffer) == 0
    assert agent.last_grad_norm >= 0.0


def test_mappo_act_batch_episode_end_and_finish():
    agent = make_agent(n_agents=3, rollout_steps=4)
    obs = np.zeros((3, *OBS_SHAPE), dtype=np.float32)
    assert agent.act_batch(obs).shape == (3,)
    assert agent.act_batch(obs, greedy=True).shape == (3,)
    # q_stats aceita a observação local e o lote conjunto.
    assert len(agent.q_stats(obs[0])) == 2
    assert len(agent.q_stats(obs)) == 2
    with pytest.raises(ValueError):
        agent.q_stats(np.zeros((2, *OBS_SHAPE), dtype=np.float32))

    fill_step(agent)
    agent.end_episode()
    arrays = agent.buffer.as_arrays()
    assert arrays["episode_ends"][-1].all()

    assert agent.finish_training() is not None
    assert len(agent.buffer) == 0


def test_mappo_update_with_value_clip():
    agent = make_agent(n_agents=2, value_clip=0.2)
    fill_step(agent)
    fill_step(agent, 1)
    assert agent.learn() is not None


def test_mappo_trains_in_real_multi_env():
    scenario = Scenario.teams(4, 2)
    env = ArenaMultiEnv(
        ArenaConfig(episode_seconds=0.3, frame_skip=1), scenario=scenario)
    agent = MAPPOAgent(
        env.observation_space.shape, env.action_space.n, n_agents=4,
        team_ids=scenario.team_ids, rollout_steps=4, batch_size=8,
        epochs=2, device="cpu")
    obs, _ = env.reset(seed=0)
    updates = 0
    while True:
        actions = agent.act_batch(obs)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(4):
            agent.add_transition(
                i, obs[i], int(actions[i]), float(rewards[i]),
                next_obs[i], bool(terminated[i]))
        if agent.learn() is not None:
            updates += 1
        obs = next_obs
        if bool(np.all(truncated)):
            break
    agent.end_episode(info)
    assert updates >= 1
    # O rabo do rollout é consumido sem erro.
    assert agent.finish_training() is not None
    env.close()


# -------------------------------------------------------------- checkpoint
def test_mappo_checkpoint_round_trip_and_env_config(tmp_path):
    agent = MAPPOAgent(OBS_SHAPE, 6, n_agents=4, team_ids=(0, 0, 1, 1),
                       device="cpu")
    cfg = ArenaConfig(n_agents=4, team_ids=(0, 0, 1, 1),
                      frame_stack=2, frame_skip=3)
    path = tmp_path / "mappo.pt"
    save_checkpoint(path, agent, cfg, {"multi_agent": True}, episodes=3)

    meta = checkpoint_meta(path)
    assert meta["algorithm"] == "mappo"
    restored = apply_env_config(ArenaConfig(), meta)
    assert restored.n_agents == 4
    assert restored.team_ids == (0, 0, 1, 1)
    assert restored.frame_stack == 2 and restored.frame_skip == 3

    other = MAPPOAgent(OBS_SHAPE, 6, n_agents=4, team_ids=(0, 0, 1, 1),
                       device="cpu")
    load_checkpoint(path, other)
    with pytest.raises(ValueError):
        load_checkpoint(
            path, MAPPOAgent((5, 20, 32), 6, n_agents=4, device="cpu"))
    # Checkpoint de MAPPO não carrega no IPPO (algoritmos diferentes).
    with pytest.raises(ValueError):
        load_checkpoint(
            path, IPPOAgent(OBS_SHAPE, 6, n_agents=4, device="cpu"))


# --------------------------------------------------------------------- CLI
def test_mappo_algorithm_discovery():
    names = available_algorithms()
    assert "mappo" in names and "ippo" in names
    assert load_algorithm("mappo").Agent is MAPPOAgent


def test_train_cli_accepts_mappo():
    from train import parse_args, validate_train_args

    args = parse_args(["--alg", "mappo"])
    assert args.agents == 4 and args.scenario == "all-vs-all"
    validate_train_args(args)
    validate_train_args(parse_args(
        ["--alg", "mappo", "--agents", "4", "--scenario", "teams",
         "--teams", "2"]))

    for bad in (
        ["--alg", "mappo", "--agents", "1"],
        ["--alg", "mappo", "--envs", "2"],
        ["--alg", "mappo", "--opponent", "rule"],
        ["--alg", "mappo", "--overlap"],
        ["--alg", "mappo", "--agents", "3", "--scenario", "teams",
         "--teams", "2"],
    ):
        with pytest.raises(SystemExit):
            validate_train_args(parse_args(bad))

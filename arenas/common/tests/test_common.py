"""Testes da infraestrutura comum e do CLI unificado."""
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from arenas import cli
from arenas.algs.alg_a2c.model import ActorCriticNetwork as A2CNetwork
from arenas.algs.alg_doubledqn_per_dueling.model import DuelingQNetwork
from arenas.algs.alg_dqn.model import QNetwork
from arenas.algs.alg_ppo.model import ActorCriticNetwork as PPONetwork
from arenas.common.agent_factory import action_dim, build_agent
from arenas.common.checkpoint import agent_from_checkpoint
from arenas.common.trainer import schedule_epsilon, train_agent
from arenas.registry import available_arenas, get_arena, load_entry


class TinyDiscrete(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {"winner": None}

    def step(self, action):
        reward = 1.0 if int(action) == 0 else 0.0
        return (np.zeros(4, dtype=np.float32), reward, True, False,
                {"winner": None})


class TinyBox(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {"winner": None}

    def step(self, action):
        return (np.zeros(4, dtype=np.float32), 0.5, True, False,
                {"winner": None})


def test_vector_models_accept_1d_observations():
    x = torch.zeros(2, 32)
    assert QNetwork((32,), 5)(x).shape == (2, 5)
    assert DuelingQNetwork((32,), 5)(x).shape == (2, 5)
    logits, value = PPONetwork((32,), 5)(x)
    assert logits.shape == (2, 5) and value.shape == (2,)
    logits, value = A2CNetwork((32,), 5)(x)
    assert logits.shape == (2, 5) and value.shape == (2,)


def test_image_models_keep_working():
    x = torch.zeros(2, 5, 20, 32)
    assert QNetwork((5, 20, 32), 6)(x).shape == (2, 6)
    assert DuelingQNetwork((5, 20, 32), 6)(x).shape == (2, 6)
    assert PPONetwork((5, 20, 32), 6)(x)[0].shape == (2, 6)


def test_agent_factory_validates_action_space_kind():
    with pytest.raises(SystemExit):
        build_agent("dqn", (4,), TinyBox().action_space)
    with pytest.raises(SystemExit):
        build_agent("ppo_continuous", (4,), TinyDiscrete().action_space)
    assert action_dim(TinyDiscrete().action_space) == 3
    assert action_dim(TinyBox().action_space) == 2


def test_schedule_epsilon_only_affects_off_policy():
    dqn = build_agent("dqn", (4,), TinyDiscrete().action_space)
    schedule_epsilon(dqn, 1, 10, 0.1)
    assert dqn.epsilon == pytest.approx(1.0)
    schedule_epsilon(dqn, 6, 10, 0.1)
    assert dqn.epsilon == pytest.approx(0.55)
    ppo = build_agent("ppo", (4,), TinyDiscrete().action_space)
    ppo.epsilon = 0.0
    schedule_epsilon(ppo, 6, 10, 0.1)
    assert ppo.epsilon == 0.0


def test_dqn_trainer_saves_checkpoint(tmp_path):
    env = TinyDiscrete()
    agent = build_agent(
        "dqn", env.observation_space.shape, env.action_space,
        batch_size=2, warmup=0)
    train_agent(
        env, agent,
        config=SimpleNamespace(nome="tiny"),
        episodes=1,
        seed=1,
        checkpoint_dir=tmp_path,
        log_path=tmp_path / "training.csv",
        metric_fn=lambda info: {"won": 1},
        metric_keys=("won",),
    )
    assert (tmp_path / "latest.pt").exists()
    loaded, meta = agent_from_checkpoint(
        tmp_path / "latest.pt", TinyDiscrete())
    assert loaded.n_actions == 3
    assert meta["algorithm"] == "dqn"


def test_continuous_trainer_saves_checkpoint(tmp_path):
    env = TinyBox()
    agent = build_agent(
        "ppo_continuous", env.observation_space.shape, env.action_space,
        batch_size=2, rollout_steps=4, epochs=1)
    train_agent(
        env, agent,
        config=SimpleNamespace(nome="tiny-box"),
        episodes=2,
        seed=2,
        checkpoint_dir=tmp_path,
        log_path=tmp_path / "training.csv",
    )
    assert (tmp_path / "latest.pt").exists()
    loaded, meta = agent_from_checkpoint(
        tmp_path / "latest.pt", TinyBox())
    assert loaded.n_actions == 2
    assert meta["algorithm"] == "ppo_continuous"


def test_registry_lists_and_loads_arenas():
    assert set(available_arenas()) == {"seek_coin", "risk", "economy"}
    assert get_arena("risk").default_algorithm == "dqn"
    assert get_arena("economy").default_algorithm == "ppo_continuous"
    with pytest.raises(SystemExit):
        get_arena("inexistente")
    assert callable(load_entry("risk", "train").main)
    assert callable(load_entry("economy", "play").main)


def test_dispatch_delegates_remaining_arguments(monkeypatch):
    seen = {}

    class FakeModule:
        def main(self):
            seen["argv"] = list(sys.argv)

    monkeypatch.setattr(cli, "load_entry", lambda name, kind: FakeModule())
    cli.dispatch("train", ["--arena", "risk", "--foo", "1"])
    assert seen["argv"] == ["risk.train", "--foo", "1"]


def test_dispatch_global_help_without_arena(capsys):
    cli.dispatch("train", ["--help"])
    output = capsys.readouterr().out
    assert "economy" in output and "risk" in output

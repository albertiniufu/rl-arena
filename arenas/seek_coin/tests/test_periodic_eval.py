import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from arenas.algs.alg_dqn.agent import DQNAgent
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.train import greedy_evaluation, parse_args, validate_train_args

ROOT = Path(__file__).resolve().parents[1]


def _assert_rng_unchanged(before):
    random_state, np_state, torch_state = before
    assert random.getstate() == random_state
    after = np.random.get_state()
    assert after[0] == np_state[0]
    assert np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_parse_args_periodic_eval_defaults_and_flags():
    args = parse_args([])
    assert args.eval_every == 0
    assert args.eval_episodes == 5
    assert args.eval_seed == 1000
    assert args.eval_opponent == "rule"
    args = parse_args([
        "--eval-every", "10", "--eval-episodes", "3",
        "--eval-seed", "42", "--eval-opponent", "coins",
    ])
    assert (args.eval_every, args.eval_episodes, args.eval_seed,
            args.eval_opponent) == (10, 3, 42, "coins")


def test_validate_train_args_rejects_bad_periodic_eval_values():
    with pytest.raises(SystemExit):
        validate_train_args(parse_args(["--eval-every", "-1"]))
    with pytest.raises(SystemExit):
        validate_train_args(parse_args(
            ["--eval-every", "10", "--eval-episodes", "0"]))
    # O caminho vetorizado não tem a avaliação periódica.
    with pytest.raises(SystemExit):
        validate_train_args(parse_args(
            ["--envs", "2", "--opponent", "rule", "--eval-every", "10"]))


def test_greedy_evaluation_keeps_training_rng_untouched():
    """A avaliação usa os RNGs do próprio env, não os do treino."""
    cfg = ArenaConfig()
    cfg.episode_seconds = 0.2  # episódios curtos: o teste avalia só o contrato
    env = ArenaEnv(cfg, opponent_policy="rule")
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    before = (random.getstate(), np.random.get_state(),
              torch.get_rng_state())
    metrics = greedy_evaluation(agent, env, episodes=2, seed=123)
    _assert_rng_unchanged(before)
    assert set(metrics) == {
        "reward_mean", "reward_std", "coins_mean", "coins_std",
        "deaths_mean", "deaths_std",
    }
    assert all(np.isfinite(list(metrics.values())))
    env.close()


def test_periodic_eval_logs_tensorboard_scalars(tmp_path):
    """O treino com --eval-every grava eval/* no run do TensorBoard."""
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    run_name = "periodic_eval_smoke"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--alg", "dqn", "--episodes", "1", "--opponent", "rule",
        "--frame-skip", "10", "--eval-every", "1", "--eval-episodes", "1",
        "--tensorboard", "--run-name", run_name,
        "--checkpoint-dir", str(tmp_path / "ckpt"),
    ]
    env = dict(os.environ, TORCH_CPP_LOG_LEVEL="ERROR", PYTHONUNBUFFERED="1")
    subprocess.run(cmd, cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True)
    events = sorted(
        (tmp_path / "logs/runs" / run_name).glob("events.out.tfevents.*"))
    assert events
    acc = EventAccumulator(
        str(events[0]),
        size_guidance={"scalars": 0, "images": 1, "histograms": 1,
                       "tensors": 1},
    )
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    for tag in ("eval/coins_mean", "eval/coins_std", "eval/reward_mean",
                "eval/deaths_mean"):
        assert tag in tags
        assert acc.Scalars(tag)[0].step == 1

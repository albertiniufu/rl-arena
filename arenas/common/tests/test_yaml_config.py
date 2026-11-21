"""Testes da integração YAML dos entry points."""
import argparse

import pytest

from arenas.common.yaml_config import (
    ConfigError,
    add_config_args,
    apply_yaml_defaults,
    arena_from_config,
    config_defaults,
    load_yaml,
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_config_defaults_maps_nested_sections(tmp_path):
    path = _write(tmp_path / "run.yaml", """
arena: seek_coin
setup: S6
algorithm: ppo
seed: 7
level: 3
training:
  episodes: 123
  max_hours: 0.5
  checkpoint_dir: checkpoints/test
  agent_params:
    lr: 0.0003
    batch_size: 128
evaluation:
  episodes: 10
  seed: 1000
  checkpoint: checkpoints/test/latest.pt
""")
    data = load_yaml(path)
    train = config_defaults(data, "train")
    assert train["alg"] == "ppo"
    assert train["episodes"] == 123
    assert train["max_hours"] == 0.5
    assert train["checkpoint_dir"] == "checkpoints/test"
    assert train["agent_param"] == ["lr=0.0003", "batch_size=128"]
    ev = config_defaults(data, "evaluate")
    assert ev["episodes"] == 10
    assert ev["checkpoint"] == "checkpoints/test/latest.pt"
    assert arena_from_config(path) == "seek_coin"


def test_unknown_yaml_key_raises(tmp_path):
    path = _write(tmp_path / "bad.yaml", """
arena: seek_coin
chave_que_nao_existe: 1
""")
    with pytest.raises(ConfigError):
        config_defaults(load_yaml(path), "train")


def test_apply_yaml_defaults_cli_wins(tmp_path):
    path = _write(tmp_path / "run.yaml", """
arena: seek_coin
alg: ppo
training:
  episodes: 5
  agent_params:
    rollout_steps: 16
""")
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--alg", default="dqn")
    parser.add_argument("--agent-param", action="append", default=[])
    apply_yaml_defaults(parser, ["--config", str(path)], "train")
    args = parser.parse_args(["--config", str(path)])
    assert args.episodes == 5
    assert args.alg == "ppo"
    assert args.agent_param == ["rollout_steps=16"]
    # CLI vence o YAML.
    args = parser.parse_args(["--config", str(path), "--episodes", "7"])
    assert args.episodes == 7


def test_set_overrides_yaml(tmp_path):
    path = _write(tmp_path / "run.yaml", """
arena: seek_coin
training:
  episodes: 5
""")
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--episodes", type=int, default=300)
    apply_yaml_defaults(
        parser,
        ["--config", str(path), "--set", "training.episodes=9"],
        "train")
    args = parser.parse_args([
        "--config", str(path), "--set", "training.episodes=9"])
    assert args.episodes == 9


def test_seek_coin_train_parse_args_uses_yaml(tmp_path):
    from arenas.seek_coin.train import parse_args

    path = _write(tmp_path / "run.yaml", """
arena: seek_coin
alg: ppo
level: 3
episode_seconds: 90
training:
  episodes: 4
  agent_params:
    rollout_steps: 32
    epochs: 1
""")
    args = parse_args(["--config", str(path)])
    assert args.alg == "ppo"
    assert args.level == 3
    assert args.episode_seconds == 90
    assert args.episodes == 4
    assert "rollout_steps=32" in args.agent_param
    assert "epochs=1" in args.agent_param
    args = parse_args(["--config", str(path), "--episodes", "8"])
    assert args.episodes == 8


def test_seek_coin_evaluate_parse_args_uses_yaml(tmp_path):
    from arenas.seek_coin.evaluate import parse_args

    path = _write(tmp_path / "eval.yaml", """
arena: seek_coin
level: 3
episode_seconds: 90
evaluation:
  checkpoint: checkpoints/test/latest.pt
  episodes: 3
  seed: 1000
""")
    args = parse_args(["--config", str(path)])
    assert args.checkpoint == "checkpoints/test/latest.pt"
    assert args.episodes == 3
    assert args.level == 3
    assert args.episode_seconds == 90

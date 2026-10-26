"""Testes das opções de adversário do play.py."""
import pytest

from arena.actions import N_ACTIONS
from arena.config import ArenaConfig
from alg_dqn.agent import DQNAgent
from alg_registry import CheckpointPolicy, save_checkpoint
from play import needs_render, parse_args, resolve_opponent, validate_play_args


def save_dqn(path, frame_stack=1, frame_skip=1):
    cfg = ArenaConfig(frame_stack=frame_stack, frame_skip=frame_skip)
    obs_shape = (5 * cfg.frame_stack, cfg.height // cfg.cell, cfg.width // cfg.cell)
    agent = DQNAgent(obs_shape, N_ACTIONS)
    save_checkpoint(path, agent, cfg)
    return path


def test_parse_args_accepts_human_and_checkpoint_opponents(tmp_path):
    assert parse_args(["--opponent", "human"]).opponent == "human"
    path = save_dqn(tmp_path / "opp.pt", frame_stack=4, frame_skip=3)
    assert parse_args(["--opponent", str(path)]).opponent == str(path)


def test_two_human_players_are_rejected():
    """Os dois lados leriam o mesmo teclado; escolha apenas um."""
    args = parse_args(["--player", "human", "--opponent", "human"])
    with pytest.raises(SystemExit, match="apenas um controle humano"):
        validate_play_args(args)


def test_human_opponent_forces_render():
    """Sem janela não há teclado: adversário humano ativa o render sozinho."""
    assert needs_render(parse_args(["--opponent", "human"]))
    assert not needs_render(parse_args(["--opponent", "rule"]))


def test_resolve_opponent_human_returns_keyboard_policy():
    from agents.human import HumanPolicy

    assert isinstance(resolve_opponent("human", ArenaConfig()), HumanPolicy)


def test_resolve_opponent_checkpoint_applies_observation_config(tmp_path):
    path = save_dqn(tmp_path / "opp.pt", frame_stack=4, frame_skip=3)
    cfg = ArenaConfig()
    policy = resolve_opponent(str(path), cfg)
    assert (cfg.frame_stack, cfg.frame_skip) == (4, 3)
    assert isinstance(policy, CheckpointPolicy)
    assert policy.agent.obs_shape == (20, 20, 32)


def test_resolve_opponent_rejects_frame_mismatch(tmp_path):
    """Dois checkpoints com pilhas diferentes não cabem no mesmo ambiente."""
    path = save_dqn(tmp_path / "opp.pt", frame_stack=1, frame_skip=1)
    with pytest.raises(SystemExit, match="frame_stack"):
        resolve_opponent(str(path), ArenaConfig(), player_frame=(4, 3))


def test_resolve_opponent_rejects_missing_checkpoint():
    with pytest.raises(SystemExit, match="Checkpoint de adversário"):
        resolve_opponent("nao/existe.pt", ArenaConfig())

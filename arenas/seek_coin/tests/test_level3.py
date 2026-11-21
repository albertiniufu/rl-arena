"""Testes do nível 3: botões, super moeda e canais de observação."""
import numpy as np
import pytest

from arenas.seek_coin.arena.actions import STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.arena.levels import (
    get_buttons,
    get_level_channel_names,
    get_super_coin_position,
)
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.observations import render_grid
from arenas.seek_coin.arena.scenario import Scenario


def _place(engine, index=0, x=50.0, y=347.5, platform_index=0):
    character = engine.characters[index]
    character.x, character.y = x, y
    character.vx = character.vy = 0.0
    character.on_ground = True
    character.platform_index = platform_index
    return character


def test_level_three_metadata():
    assert get_level_channel_names(3) == (
        "activate_button", "cancel_button", "super_coin")
    assert get_super_coin_position(3) == (610.0, 342.0)
    assert [b.kind for b in get_buttons(3)] == ["activate", "cancel"]


def test_level_three_env_shapes():
    single = ArenaEnv(ArenaConfig(level=3), opponent_policy="random")
    assert single.observation_space.shape == (8, 20, 32)
    multi = ArenaMultiEnv(ArenaConfig(level=3),
                          scenario=Scenario.all_vs_all(2))
    assert multi.observation_space.shape == (9, 20, 32)
    single.close()
    multi.close()


def test_activate_button_spawns_super_coin():
    engine = ArenaEngine(ArenaConfig(level=3))
    engine.reset(seed=1)
    _place(engine)
    event = engine.step([STOP], 1 / engine.cfg.fps)
    assert event["super_coin_activated"] is True
    assert engine.super_coin is not None
    assert engine.super_coin.timer == pytest.approx(
        engine.cfg.super_coin_seconds)


def test_collecting_super_coin_gives_coins_and_event():
    cfg = ArenaConfig(level=3, frame_skip=1)
    env = ArenaEnv(cfg, opponent_policy="random")
    env.reset(seed=1)
    _place(env.engine)
    env.step(STOP)
    _place(env.engine, x=610.0, platform_index=1)
    _, reward, _, _, info = env.step(STOP)
    assert info["super_coin_collected_this_step"] == 1
    assert info["coins"] == cfg.super_coin_coins
    # STOP custa 0.1; a super moeda paga super_coin_reward.
    assert reward == pytest.approx(cfg.super_coin_reward + cfg.stop_penalty)
    env.close()


def test_cancel_button_deactivates_super_coin():
    engine = ArenaEngine(ArenaConfig(level=3))
    engine.reset(seed=1)
    _place(engine)
    engine.step([STOP], 1 / engine.cfg.fps)
    assert engine.super_coin is not None
    _place(engine, x=300.0, y=202.5, platform_index=4)
    event = engine.step([STOP], 1 / engine.cfg.fps)
    assert event["super_coin_cancelled"] is True
    assert engine.super_coin is None
    assert engine.super_coin_cooldown == pytest.approx(
        engine.cfg.super_coin_cooldown)


def test_super_coin_expires_and_respects_cooldown():
    engine = ArenaEngine(ArenaConfig(level=3))
    engine.reset(seed=1)
    _place(engine)
    engine.step([STOP], 1 / engine.cfg.fps)
    event = engine.step([STOP], engine.cfg.super_coin_seconds + 0.1)
    assert event["super_coin_expired"] is True
    assert engine.super_coin is None
    # Ainda no botão de ativação: sem nova ativação durante o cooldown.
    event = engine.step([STOP], 1 / engine.cfg.fps)
    assert event["super_coin_activated"] is False


def test_observation_marks_buttons_and_super_coin():
    cfg = ArenaConfig(level=3)
    engine = ArenaEngine(cfg)
    engine.reset(seed=1)
    grid = render_grid(engine, cfg, channels=5 + 3)
    assert grid[5].sum() > 0  # botão de ativação
    assert grid[6].sum() > 0  # botão de cancelamento
    assert grid[7].sum() == 0  # nenhuma super moeda ainda
    _place(engine)
    engine.step([STOP], 1 / cfg.fps)
    grid = render_grid(engine, cfg, channels=5 + 3)
    assert grid[7].sum() > 0  # super moeda ativa


def test_env_state_roundtrip_preserves_super_coin():
    env = ArenaEnv(ArenaConfig(level=3), opponent_policy="random")
    env.reset(seed=1)
    _place(env.engine)
    env.step(STOP)
    state = env.get_state()
    assert state["super_coin"] is not None
    env2 = ArenaEnv(ArenaConfig(level=3), opponent_policy="random")
    env2.reset(seed=2)
    env2.set_state(state)
    assert env2.engine.super_coin is not None
    assert env2.engine.super_coin.timer == pytest.approx(
        env.engine.super_coin.timer)
    env.close()
    env2.close()


def test_coin_policy_prioritizes_super_coin():
    from types import SimpleNamespace

    from arenas.seek_coin.agents.baseline import CoinRulePolicy, PolicyContext
    from arenas.seek_coin.arena.actions import RIGHT

    ctx = PolicyContext(
        obs=np.zeros((5, 20, 32), dtype=np.float32),
        rng=np.random.default_rng(1),
        self_character=SimpleNamespace(x=200.0, y=347.5, alive=True),
        super_coin=SimpleNamespace(x=610.0, y=342.0),
    )
    assert CoinRulePolicy().act(ctx) == RIGHT

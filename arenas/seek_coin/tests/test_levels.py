"""Testes do sistema de níveis e da plataforma móvel (nível 1)."""
import numpy as np

from arenas.algs.alg_registry import apply_env_config
from arenas.seek_coin.arena.actions import STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine
from arenas.seek_coin.arena.levels import STATIC_PLATFORMS
from arenas.seek_coin.train import parse_args


def _engine(level: int) -> ArenaEngine:
    engine = ArenaEngine(ArenaConfig(level=level, opponent_count=0))
    engine.reset(seed=0, n_characters=1)
    return engine


def test_level_zero_keeps_the_historical_map():
    engine = _engine(0)
    assert engine.platforms == list(STATIC_PLATFORMS[0])
    assert engine.moving_specs == ()


def test_level_one_has_a_moving_platform():
    engine = _engine(1)
    assert len(engine.platforms) == len(STATIC_PLATFORMS[0]) + 1
    moving = engine.platforms[-1]
    assert moving[1] == 250 and moving[1] == engine.moving_specs[0].y
    x0 = moving[0]
    engine.step([STOP], 0.1)
    assert engine.platforms[-1][0] > x0


def test_moving_platform_carries_the_character():
    engine = _engine(1)
    moving_index = len(STATIC_PLATFORMS[0])
    character = engine.characters[0]
    character.x = 300.0
    character.y = float(engine.moving_specs[0].y)
    character.vx = character.vy = 0.0
    character.on_ground = True
    character.platform_index = moving_index
    engine.step([STOP], 0.1)
    assert character.x > 300.05
    assert character.on_ground is True
    assert character.platform_index == moving_index


def test_coins_spawn_only_on_static_platforms():
    engine = _engine(1)
    for coin in engine.coins:
        assert any(
            x <= coin.x <= x + w and abs(coin.y - (y - 18)) < 1e-6
            for x, y, w, _ in engine.static_platforms
        )


def test_checkpoint_config_applies_level():
    cfg = ArenaConfig(level=0)
    apply_env_config(cfg, {"config": {"level": 1}})
    assert cfg.level == 1


def test_train_cli_accepts_level():
    assert parse_args(["--level", "1"]).level == 1

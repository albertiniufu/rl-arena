"""Testes do sistema de níveis e das plataformas móveis (nível 1)."""
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


def test_level_one_has_two_moving_platforms():
    engine = _engine(1)
    assert len(engine.static_platforms) == len(STATIC_PLATFORMS[0]) - 1
    assert len(engine.moving_specs) == 2
    assert len(engine.platforms) == len(STATIC_PLATFORMS[0]) + 1

    central, highest = engine.moving_specs
    assert central.axis == "x" and central.y == 250
    assert highest.axis == "y" and highest.y == 145

    central_index = len(engine.static_platforms)
    highest_index = central_index + 1
    x0 = engine.platforms[central_index][0]
    y0 = engine.platforms[highest_index][1]
    engine.step([STOP], 0.1)
    assert engine.platforms[central_index][0] > x0
    assert engine.platforms[highest_index][1] > y0


def test_central_platform_carries_the_character():
    engine = _engine(1)
    central_index = len(engine.static_platforms)
    character = engine.characters[0]
    character.x = 300.0
    character.y = float(engine.moving_specs[0].y)
    character.vx = character.vy = 0.0
    character.on_ground = True
    character.platform_index = central_index
    engine.step([STOP], 0.1)
    assert character.x > 300.05
    assert character.on_ground is True
    assert character.platform_index == central_index


def test_highest_platform_carries_the_character_vertically():
    engine = _engine(1)
    highest_index = len(engine.static_platforms) + 1
    spec = engine.moving_specs[1]
    character = engine.characters[0]
    character.x = 100.0
    character.y = float(spec.y)
    character.vx = character.vy = 0.0
    character.on_ground = True
    character.platform_index = highest_index
    engine.step([STOP], 0.1)
    assert character.y > spec.y + 0.05
    assert character.on_ground is True
    assert character.platform_index == highest_index


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

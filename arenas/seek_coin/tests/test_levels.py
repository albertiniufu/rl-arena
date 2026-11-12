"""Testes do sistema de níveis e das plataformas móveis (nível 1)."""
from arenas.algs.alg_registry import apply_env_config
from arenas.seek_coin.arena.actions import STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine
from arenas.seek_coin.arena.levels import STATIC_PLATFORMS, get_fixed_coins
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


def test_high_fixed_coins_and_static_random_coins():
    engine = _engine(1)
    fixed = get_fixed_coins(1)
    assert len(engine.coins) == engine.cfg.coin_count
    for x, y in fixed:
        assert any(abs(coin.x - x) < 1e-9 and abs(coin.y - y) < 1e-9
                   for coin in engine.coins)
    for coin in engine.coins:
        if any(abs(coin.x - x) < 1e-9 and abs(coin.y - y) < 1e-9
               for x, y in fixed):
            continue
        assert any(
            x <= coin.x <= x + w and abs(coin.y - (y - 18)) < 1e-6
            for x, y, w, _ in engine.static_platforms
        )


def test_high_coins_require_platform_at_its_top():
    engine = _engine(1)
    highest = engine.moving_specs[1]
    top = highest.y                 # 145
    bottom = highest.y + highest.amplitude  # 185
    for _, coin_y in get_fixed_coins(1):
        assert top - coin_y < 100       # alcançável no topo
        assert bottom - coin_y > 110    # fora do alcance na posição baixa


def test_checkpoint_config_applies_level():
    cfg = ArenaConfig(level=0)
    apply_env_config(cfg, {"config": {"level": 1}})
    assert cfg.level == 1


def test_train_cli_accepts_level():
    assert parse_args(["--level", "1"]).level == 1

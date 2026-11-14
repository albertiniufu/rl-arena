"""Testes do sistema de níveis e das plataformas móveis (nível 1)."""
from arenas.algs.alg_registry import apply_env_config
from arenas.seek_coin.arena.actions import STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine, Hazard
from arenas.seek_coin.arena.levels import (
    STATIC_PLATFORMS, get_fixed_coins, get_hazard_kinds,
    get_moving_coins)
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.observations import render_grid
from arenas.seek_coin.arena.scenario import Scenario
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
    assert len(engine.coins) == (
        engine.cfg.coin_count + len(get_moving_coins(1)))
    for x, y in fixed:
        assert any(abs(coin.x - x) < 1e-9 and abs(coin.y - y) < 1e-9
                   for coin in engine.coins)
    for coin in engine.coins:
        if coin.platform_index is not None:
            continue
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


def test_moving_coins_follow_highest_platform():
    engine = _engine(1)
    moving = [coin for coin in engine.coins
              if coin.platform_index is not None]
    assert len(moving) == 3
    highest_index = len(engine.static_platforms) + 1
    assert all(coin.platform_index == highest_index for coin in moving)
    y0 = [coin.y for coin in moving]
    engine.step([STOP], 0.1)
    assert all(coin.y > y0[i] for i, coin in enumerate(moving))


def test_level_two_has_hazards():
    engine = _engine(2)
    assert get_hazard_kinds(2) == ("sand", "ice")
    assert len(engine.hazards) == 2 * engine.cfg.hazard_count
    assert {h.kind for h in engine.hazards} == {"sand", "ice"}
    for hazard in engine.hazards:
        assert engine.cfg.hazard_min_seconds <= hazard.timer
        assert hazard.timer <= engine.cfg.hazard_max_seconds
        assert 0 <= hazard.platform_index < len(engine.platforms)


def test_sand_reduces_speed():
    engine = _engine(2)
    engine.hazards = [Hazard("sand", 350.0, 354.0, 120.0, 6.0, 1, 10.0)]
    c = engine.characters[0]
    c.x, c.y = 400.0, 360.0
    c.vx = c.vy = 0.0
    c.on_ground = True
    c.platform_index = 1
    engine.step([2], 1.0 / engine.cfg.fps)  # RIGHT
    sand_vx = abs(c.vx)

    free = _engine(2)
    free.hazards = []
    c = free.characters[0]
    c.x, c.y = 400.0, 360.0
    c.vx = c.vy = 0.0
    c.on_ground = True
    c.platform_index = 1
    free.step([2], 1.0 / free.cfg.fps)
    assert sand_vx < abs(c.vx)


def test_ice_slides_about_one_fifth_of_screen():
    engine = _engine(2)
    engine.hazards = [Hazard("ice", 350.0, 354.0, 120.0, 6.0, 1, 10.0)]
    c = engine.characters[0]
    c.x, c.y = 400.0, 360.0
    c.vx = c.vy = 0.0
    c.on_ground = True
    c.platform_index = 1
    x0 = c.x
    steps = int(engine.cfg.ice_slide_seconds * engine.cfg.fps) + 2
    for _ in range(steps):
        engine.step([0], 1.0 / engine.cfg.fps)
    distance = abs(engine.characters[0].x - x0)
    target = engine.cfg.width * engine.cfg.ice_slide_fraction
    assert 0.5 * target < distance < 1.6 * target


def test_hazard_channels_and_env_shapes():
    engine = _engine(2)
    cfg = engine.cfg
    grid = render_grid(engine, cfg, channels=7)
    assert grid.shape == (7, 20, 32)
    assert grid[5].any() and grid[6].any()

    env = ArenaEnv(ArenaConfig(level=2, opponent_count=0))
    obs, _ = env.reset(seed=3)
    assert obs.shape == (7, 20, 32)   # 5 base + areia + gelo
    env.close()

    scenario = Scenario.all_vs_all(2)
    multi_cfg = ArenaConfig(
        level=2, n_agents=2, team_ids=scenario.team_ids)
    multi = ArenaMultiEnv(multi_cfg, scenario=scenario)
    obs, _ = multi.reset(seed=3)
    assert obs.shape == (2, 8, 20, 32)  # 6 base + areia + gelo
    multi.close()

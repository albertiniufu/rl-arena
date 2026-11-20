"""Testes do nível 5: mundo procedural infinito (chunks e rolagem)."""
import numpy as np
import pytest

from arenas.algs.alg_registry import apply_env_config
from arenas.seek_coin.arena.actions import RIGHT, STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.endless import ChunkGenerator
from arenas.seek_coin.arena.engine import ArenaEngine
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.arena.levels import (
    get_hazard_kinds,
    get_level_channel_names,
    uses_endless_world,
    uses_local_view,
)
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.observations import observation_grid_shape
from arenas.seek_coin.arena.scenario import Scenario
from arenas.seek_coin.evaluate import parse_args as parse_eval_args
from arenas.seek_coin.play import parse_args as parse_play_args
from arenas.seek_coin.train import (
    parse_args as parse_train_args,
    validate_train_args,
)

VIEW = 9
CENTER = VIEW // 2
N_CHANNELS = 5 + len(get_level_channel_names(5))  # areia, gelo, buraco
SAND, ICE, HOLE = 5, 6, 7


def _place(engine, x=100.0, y=347.5):
    character = engine.characters[0]
    character.x, character.y = x, y
    character.vx = character.vy = 0.0
    character.alive = True
    character.on_ground = True
    character.platform_index = None
    return character


def _drive(engine, steps=200, action=RIGHT, dt=1 / 30):
    for _ in range(steps):
        engine.step([action], dt)


def _coins_inside_holes(engine):
    return [
        coin for coin in engine.coins
        if any(x <= coin.x <= x + w and y <= coin.y <= y + h
               for x, y, w, h in engine.holes)
    ]


def test_level_five_metadata():
    assert uses_endless_world(5)
    assert uses_local_view(5)
    assert get_hazard_kinds(5) == ("sand", "ice")
    assert get_level_channel_names(5) == ("sand", "ice", "hole")
    assert observation_grid_shape(ArenaConfig(level=5)) == (VIEW, VIEW)


def test_chunk_generator_is_deterministic_by_seed_and_index():
    first = ChunkGenerator(seed=11)
    second = ChunkGenerator(seed=11)
    other = ChunkGenerator(seed=12)
    for index in range(6):
        a, b = first.generate(index), second.generate(index)
        assert a.platforms == b.platforms
        assert a.holes == b.holes
        assert a.coins == b.coins
        assert a.movers == b.movers
    assert first.generate(3).platforms != other.generate(3).platforms


def test_chunks_start_and_end_with_ground():
    generator = ChunkGenerator(seed=5)
    for index in range(8):
        content = generator.generate(index)
        start = index * generator.chunk_width
        end = start + generator.chunk_width
        ground = [p for p in content.platforms if p[1] == 360.0]
        assert ground[0][0] == pytest.approx(start)
        assert ground[-1][0] + ground[-1][2] == pytest.approx(end)
        # Nenhum buraco encosta nas bordas do chunk (emenda segura).
        for x, _, w, _ in content.holes:
            assert x > start and x + w < end


def test_env_observation_shape_and_center():
    env = ArenaEnv(ArenaConfig(level=5), opponent_policy="random")
    assert env.observation_space.shape == (N_CHANNELS, VIEW, VIEW)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert tuple(np.argwhere(obs[-N_CHANNELS:][1])[0]) == (CENTER, CENTER)
    env.close()


def test_world_scrolls_and_generates_new_chunks():
    engine = ArenaEngine(ArenaConfig(level=5))
    engine.reset(seed=3)
    first_chunk = engine.next_chunk
    assert engine.world_offset == 0.0
    _place(engine, x=600.0)
    engine.step([RIGHT], 1 / 30)
    assert engine.world_offset > 0.0
    assert engine.characters[0].x <= engine.cfg.width * 0.55 + 1e-6
    assert engine.next_chunk > first_chunk
    assert engine.distance >= 599.0


def test_memory_stays_bounded_in_a_long_scroll():
    engine = ArenaEngine(ArenaConfig(level=5))
    engine.reset(seed=4)
    _drive(engine, steps=1500)
    assert engine.world_offset > 1000.0
    assert len(engine.platforms) < 40
    assert len(engine.holes) < 20
    assert len(engine.coins) < 40
    assert len(engine.hazards) <= 2 * engine.cfg.hazard_count
    # Nada ativo ficou muito para trás da tela.
    assert all(x + w > -100 for x, _, w, _ in engine.platforms)


def test_coins_never_spawn_inside_holes():
    engine = ArenaEngine(ArenaConfig(level=5))
    engine.reset(seed=7)
    assert _coins_inside_holes(engine) == []
    _drive(engine, steps=300)
    assert _coins_inside_holes(engine) == []


def test_hole_channel_marks_visible_holes_only():
    env = ArenaEnv(ArenaConfig(level=5), opponent_policy="random")
    env.reset(seed=2)
    env.engine.holes = [(90.0, 360.0, 60.0, 40.0)]
    _place(env.engine, x=100.0)
    assert env._observation()[HOLE].sum() > 0
    _place(env.engine, x=600.0)
    assert env._observation()[HOLE].sum() == 0
    env.close()


def test_falling_into_a_hole_kills_and_respawns_nearby():
    engine = ArenaEngine(ArenaConfig(level=5))
    engine.reset(seed=8)
    x, _, w, _ = engine.holes[0]
    character = engine.characters[0]
    character.x = x + w / 2
    character.y = 300.0
    character.vx = character.vy = 0.0
    character.alive = True
    character.on_ground = False
    character.platform_index = None
    for _ in range(400):
        engine.step([STOP], 1 / 30)
        if character.deaths > 0:
            break
    assert character.deaths == 1
    for _ in range(400):
        engine.step([STOP], 1 / 30)
        if character.alive:
            break
    assert character.alive
    assert character.on_ground
    assert 0 <= character.x < engine.cfg.width


def test_hazards_are_present_and_spawned_on_active_platforms():
    engine = ArenaEngine(ArenaConfig(level=5))
    engine.reset(seed=9)
    kinds = sorted(h.kind for h in engine.hazards)
    assert kinds == ["ice", "ice", "sand", "sand"]
    for hazard in engine.hazards:
        x, y, w, _ = engine.platforms[hazard.platform_index]
        assert x <= hazard.x <= x + w


def test_checkpoint_config_applies_level_five():
    cfg = ArenaConfig()
    apply_env_config(cfg, {"config": {"level": 5, "view_size": 7}})
    assert cfg.level == 5 and cfg.view_size == 7
    env = ArenaEnv(cfg, opponent_policy="random")
    assert env.observation_space.shape == (N_CHANNELS, 7, 7)
    env.close()


def test_cli_accepts_level_five():
    assert parse_train_args(["--level", "5"]).level == 5
    assert parse_eval_args(["--level", "5"]).level == 5
    assert parse_play_args(["--level", "5"]).level == 5


def test_multiagent_rejects_endless_world():
    with pytest.raises(ValueError):
        ArenaMultiEnv(ArenaConfig(level=5), scenario=Scenario.all_vs_all(2))
    args = parse_train_args(
        ["--level", "5", "--alg", "ippo", "--agents", "2"])
    with pytest.raises(SystemExit):
        validate_train_args(args)


def test_gpu_env_rejects_level_five():
    pytest.importorskip("torch")
    from arenas.seek_coin.arena.gpu_env import GpuArenaVecEnv

    with pytest.raises(NotImplementedError):
        GpuArenaVecEnv(ArenaConfig(level=5, n_agents=2, team_ids=(0, 1)),
                       n_envs=1, n_agents=2, team_ids=(0, 1),
                       device="cpu")


def test_single_agent_smoke_reports_distance():
    cfg = ArenaConfig(level=5, episode_seconds=1.0, frame_skip=2)
    env = ArenaEnv(cfg, opponent_policy="rule")
    obs, _ = env.reset(seed=3)
    steps = 0
    while True:
        obs, _, terminated, truncated, info = env.step(RIGHT)
        steps += 1
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break
    assert steps > 0 and truncated
    assert info["distance"] > 0
    assert info["chunks"] >= 2
    env.close()

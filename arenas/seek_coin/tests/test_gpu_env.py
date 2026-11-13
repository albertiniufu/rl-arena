"""Testes do ambiente vetorizado completo em GPU/CPU."""
import numpy as np
import torch

from arenas.seek_coin.arena.actions import PUSH_RIGHT, STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine
from arenas.seek_coin.arena.gpu_engine import GpuSeekCoin
from arenas.seek_coin.arena.gpu_env import GpuArenaVecEnv
from arenas.seek_coin.arena.observations import render_grid


def test_gpu_env_multiagent_shapes_and_truncation():
    cfg = ArenaConfig(
        level=1, n_agents=4, team_ids=(0, 1, 2, 3),
        frame_skip=2, frame_stack=2, episode_seconds=0.2,
        coin_count=5)
    env = GpuArenaVecEnv(
        config=cfg, n_envs=2, n_agents=4,
        team_ids=(0, 1, 2, 3), device="cpu", seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (2, 4, 12, 20, 32)
    actions = torch.zeros((2, 4), dtype=torch.int64)
    terminated = truncated = None
    for _ in range(5):
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert rewards.shape == (2, 4)
        assert terminated.shape == (2, 4)
        assert truncated.shape == (2, 4)
    assert bool(truncated.all())
    assert obs.shape == (2, 4, 12, 20, 32)
    assert torch.isfinite(obs).all()


def test_gpu_engine_push_transfers_coins():
    engine = GpuSeekCoin(
        n_envs=1, n_chars=2, level=0, device="cpu", seed=0)
    engine.x[0, 0] = 100.0
    engine.x[0, 1] = 120.0
    engine.y[0, 0] = engine.y[0, 1] = 360.0
    engine.alive[:] = True
    engine.coins[0, 0] = 0
    engine.coins[0, 1] = 5
    engine.on_ground[:] = True
    actions = torch.tensor([[PUSH_RIGHT, STOP]], dtype=torch.int64)
    out = engine.step(actions)
    assert out["coins_transferred"][0, 0].item() == 5
    assert out["coins_transferred"][0, 1].item() == -5
    assert out["push_missed"][0, 0].item() is False


def test_gpu_observation_matches_cpu_render_grid():
    cfg = ArenaConfig(
        level=1, n_agents=4, team_ids=(0, 1, 2, 3), coin_count=10)
    cpu = ArenaEngine(cfg)
    cpu.reset(seed=0, n_characters=4)
    env = GpuArenaVecEnv(
        config=cfg, n_envs=1, n_agents=4, team_ids=(0, 1, 2, 3),
        device="cpu", seed=0)
    env.x = torch.tensor(
        [[c.x for c in cpu.characters]], dtype=torch.float32)
    env.y = torch.tensor(
        [[c.y for c in cpu.characters]], dtype=torch.float32)
    env.alive = torch.ones((1, 4), dtype=torch.bool)
    env.coin_x = torch.tensor(
        [[c.x for c in cpu.coins]], dtype=torch.float32)
    env.coin_y = torch.tensor(
        [[c.y for c in cpu.coins]], dtype=torch.float32)
    expected = render_grid(
        cpu, cfg, viewer=0, channels=6, team_ids=(0, 1, 2, 3))
    got = env._observation_all()[0, 0].numpy()
    assert np.allclose(got, expected)


def test_gpu_env_uint8_observation_dtype():
    cfg = ArenaConfig(
        level=1, n_agents=4, team_ids=(0, 1, 2, 3),
        frame_skip=1, frame_stack=2, episode_seconds=1.0)
    env = GpuArenaVecEnv(
        config=cfg, n_envs=2, n_agents=4, team_ids=(0, 1, 2, 3),
        device="cpu", seed=0, obs_dtype="uint8")
    obs, _ = env.reset(seed=0)
    assert obs.dtype == torch.uint8
    assert obs.shape == (2, 4, 12, 20, 32)
    obs, *_ = env.step(torch.zeros((2, 4), dtype=torch.int64))
    assert obs.dtype == torch.uint8

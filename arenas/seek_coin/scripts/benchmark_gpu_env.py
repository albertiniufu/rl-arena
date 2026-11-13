#!/usr/bin/env python3
"""Benchmark CPU x GPU do seek_coin: física e passo completo.

- `--mode physics`: mede só a física (ArenaEngine por env vs GpuSeekCoin);
- `--mode full`: mede o passo completo multiagente, incluindo observação
  egocêntrica 6 canais, recompensas, penalidades, frame_skip e frame_stack
  (`ArenaMultiEnv` por env vs `GpuArenaVecEnv`).

Uso:
    python arenas/seek_coin/scripts/benchmark_gpu_env.py --mode physics --envs 64,256,1024
    python arenas/seek_coin/scripts/benchmark_gpu_env.py --mode full --envs 32,64,128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine
from arenas.seek_coin.arena.gpu_engine import GpuSeekCoin
from arenas.seek_coin.arena.gpu_env import GpuArenaVecEnv
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["physics", "full"],
                        default="physics")
    parser.add_argument("--envs", default="64,256,1024",
                        help="lista de números de ambientes")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--chars", type=int, default=2,
                        help="personagens no modo physics")
    parser.add_argument("--agents", type=int, default=4,
                        help="agentes no modo full")
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


# --------------------------------------------------------------------- #
# Física
# --------------------------------------------------------------------- #
def bench_cpu_physics(n_envs, steps, level, n_chars, seed):
    cfg = ArenaConfig(level=level, opponent_count=n_chars - 1)
    engines = [ArenaEngine(cfg) for _ in range(n_envs)]
    for i, engine in enumerate(engines):
        engine.reset(seed=seed + i, n_characters=n_chars)
    rng = np.random.default_rng(seed)
    actions = [rng.integers(0, 6, size=n_chars) for _ in range(n_envs)]
    dt = 1.0 / cfg.fps
    start = time.perf_counter()
    for _ in range(steps):
        for engine, action in zip(engines, actions):
            engine.step(action.tolist(), dt)
    return n_envs * steps / (time.perf_counter() - start)


def bench_gpu_physics(n_envs, steps, level, n_chars, seed, device, warmup):
    engine = GpuSeekCoin(
        n_envs=n_envs, n_chars=n_chars, level=level,
        device=device, seed=seed)
    actions = torch.randint(
        0, 6, (n_envs, n_chars), device=engine.device)
    for _ in range(warmup):
        engine.step(actions)
    if engine.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        engine.step(actions)
    if engine.device.type == "cuda":
        torch.cuda.synchronize()
    return n_envs * steps / (time.perf_counter() - start)


# --------------------------------------------------------------------- #
# Passo completo
# --------------------------------------------------------------------- #
def full_config(level, agents, frame_skip, frame_stack):
    return ArenaConfig(
        level=level, n_agents=agents,
        team_ids=tuple(range(agents)),
        frame_skip=frame_skip, frame_stack=frame_stack,
        coin_count=10)


def bench_cpu_full(n_envs, steps, level, agents, frame_skip, frame_stack,
                   seed):
    cfg = full_config(level, agents, frame_skip, frame_stack)
    scenario = Scenario.all_vs_all(agents)
    envs = [ArenaMultiEnv(cfg, scenario=scenario) for _ in range(n_envs)]
    for i, env in enumerate(envs):
        env.reset(seed=seed + i)
    rng = np.random.default_rng(seed)
    actions = [rng.integers(0, 6, size=agents) for _ in range(n_envs)]
    start = time.perf_counter()
    for _ in range(steps):
        for env, action in zip(envs, actions):
            env.step(action)
    return n_envs * steps / (time.perf_counter() - start)


def bench_gpu_full(n_envs, steps, level, agents, frame_skip, frame_stack,
                   seed, device, warmup):
    cfg = full_config(level, agents, frame_skip, frame_stack)
    env = GpuArenaVecEnv(
        config=cfg, n_envs=n_envs, n_agents=agents,
        team_ids=tuple(range(agents)), device=device, seed=seed)
    env.reset(seed=seed)
    actions = torch.randint(
        0, 6, (n_envs, agents), device=env.device)
    for _ in range(warmup):
        env.step(actions)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    return n_envs * steps / (time.perf_counter() - start)


def main(argv=None) -> None:
    args = parse_args(argv)
    envs = [int(item) for item in args.envs.split(",") if item.strip()]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponível; usando CPU no engine vetorizado.")
        device = "cpu"
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    mode = args.mode
    print(f"mode={mode} level={args.level} steps={args.steps} "
          f"device={device}")
    if mode == "full":
        print(f"agents={args.agents} frame_skip={args.frame_skip} "
              f"frame_stack={args.frame_stack}")
    else:
        print(f"chars={args.chars}")
    print(f"{'envs':>6} | {'CPU env-steps/s':>18} | "
          f"{'GPU env-steps/s':>18} | {'speedup':>8}")
    print("-" * 62)
    for n in envs:
        if mode == "full":
            cpu_rate = bench_cpu_full(
                n, args.steps, args.level, args.agents,
                args.frame_skip, args.frame_stack, args.seed)
            gpu_rate = bench_gpu_full(
                n, args.steps, args.level, args.agents,
                args.frame_skip, args.frame_stack, args.seed,
                device, args.warmup)
        else:
            cpu_rate = bench_cpu_physics(
                n, args.steps, args.level, args.chars, args.seed)
            gpu_rate = bench_gpu_physics(
                n, args.steps, args.level, args.chars, args.seed,
                device, args.warmup)
        print(f"{n:6d} | {cpu_rate:18,.0f} | {gpu_rate:18,.0f} | "
              f"{gpu_rate / cpu_rate:8.2f}x")


if __name__ == "__main__":
    main()

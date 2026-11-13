#!/usr/bin/env python3
"""Benchmark CPU x GPU do engine vetorizado do seek_coin.

O baseline de CPU usa `ArenaEngine` com um engine por ambiente (o custo
real de rodar N ambientes em Python). O lado GPU usa `GpuSeekCoin`, que
executa a física em lote. O benchmark mede **passos de ambiente por
segundo** (E × passos / tempo) e é uma comparação de throughput, não de
equivalência bit a bit (o engine GPU simplifica empurrões).

Uso:
    python arenas/seek_coin/scripts/benchmark_gpu_env.py --envs 64,256,1024
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default="64,256,1024",
                        help="lista de números de ambientes")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--chars", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def bench_cpu(n_envs: int, steps: int, level: int, n_chars: int,
              seed: int) -> float:
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
    elapsed = time.perf_counter() - start
    return n_envs * steps / elapsed


def bench_gpu(n_envs: int, steps: int, level: int, n_chars: int,
              seed: int, device: str, warmup: int) -> float:
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
    elapsed = time.perf_counter() - start
    return n_envs * steps / elapsed


def main(argv=None) -> None:
    args = parse_args(argv)
    envs = [int(item) for item in args.envs.split(",") if item.strip()]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA indisponível; usando CPU no engine vetorizado.")
        device = "cpu"
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"level={args.level} chars={args.chars} steps={args.steps} "
          f"device={device}")
    print(f"{'envs':>6} | {'CPU env-steps/s':>18} | "
          f"{'GPU env-steps/s':>18} | {'speedup':>8}")
    print("-" * 62)
    for n in envs:
        cpu_rate = bench_cpu(n, args.steps, args.level, args.chars, args.seed)
        gpu_rate = bench_gpu(
            n, args.steps, args.level, args.chars, args.seed,
            device, args.warmup)
        print(f"{n:6d} | {cpu_rate:18,.0f} | {gpu_rate:18,.0f} | "
              f"{gpu_rate / cpu_rate:8.2f}x")


if __name__ == "__main__":
    main()

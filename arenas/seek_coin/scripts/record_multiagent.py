#!/usr/bin/env python3
"""Grava um episódio guloso de um checkpoint multiagente em MP4.

Usado pelo watcher de vídeos horários para IPPO/MAPPO/HAPPO. O checkpoint
identifica o algoritmo e o cenário; o agente age em lote com `act_batch`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from arenas.algs.alg_registry import (
    algorithm_from_checkpoint,
    apply_env_config,
    load_checkpoint,
)
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skin", default=None)
    return parser.parse_args(argv)


def record_frame(writer):
    import pygame
    import pygame.surfarray  # noqa: F401
    frame = np.transpose(
        pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
    writer.append_data(frame)


def main(argv=None) -> None:
    args = parse_args(argv)
    alg, meta = algorithm_from_checkpoint(args.checkpoint)
    if not getattr(alg.Agent, "multi_agent", False):
        raise SystemExit("checkpoint não é multiagente")
    cfg = ArenaConfig()
    apply_env_config(cfg, meta)
    scenario = Scenario.from_config(cfg)
    env = ArenaMultiEnv(
        cfg, scenario=scenario, render_mode="human", skin=args.skin)
    agent = alg.Agent(
        env.observation_space.shape, env.action_space.n,
        n_agents=scenario.n_agents,
        team_ids=tuple(scenario.team_ids), device=args.device)
    load_checkpoint(args.checkpoint, agent, resume=False)

    import imageio
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.record, fps=args.fps or cfg.fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "23",
                       "-movflags", "+faststart"])
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            env.render()
            record_frame(writer)
            while True:
                actions = agent.act_batch(obs, greedy=True)
                obs, rewards, terminated, truncated, info = env.step(actions)
                env.render()
                record_frame(writer)
                if bool(np.all(truncated)):
                    break
            teams = info.get("teams", [])
            print("episódio", episode + 1,
                  [(t["team"], t["coins"]) for t in teams])
    finally:
        writer.close()
        env.close()
        print("vídeo salvo em", args.record)


if __name__ == "__main__":
    main()

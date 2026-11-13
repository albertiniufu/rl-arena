#!/usr/bin/env python3
"""Duelo curto no nível 1: PPO (J1) vs HAPPO (J2).

O PPO foi treinado no ambiente single-agent (observação de 5 canais) e o
HAPPO no multiagente (6 canais). Aqui os dois jogam no `ArenaMultiEnv`: o PPO
controla J1 (a observação multiagente é convertida para 5 canais, descartando
o canal de aliados) e o HAPPO controla J2 com o ator do agente 1. A partida
é gulosa e gravada em MP4.

Uso:
    python arenas/seek_coin/scripts/duel_ppo_happo.py \
        --ppo-checkpoint checkpoints/compare15/ppo/best.pt \
        --happo-checkpoint checkpoints/compare15/happo/best.pt \
        --record checkpoints/compare15/videos/ppo_vs_happo.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Permite rodar o arquivo diretamente (python arenas/seek_coin/scripts/...).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from arenas.algs.alg_registry import (
    algorithm_from_checkpoint,
    apply_env_config,
    checkpoint_meta,
    load_checkpoint,
)
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-checkpoint", required=True)
    parser.add_argument("--happo-checkpoint", required=True)
    parser.add_argument("--record", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes deve ser >= 1")
    return args


def load_agent(alg, meta, n_agents, team_ids, device):
    kwargs = {"device": device}
    if getattr(alg.Agent, "multi_agent", False):
        kwargs.update(n_agents=int(n_agents), team_ids=tuple(team_ids))
    agent = alg.Agent(
        tuple(meta["obs_shape"]), int(meta["n_actions"]), **kwargs)
    load_checkpoint(meta["path"], agent, resume=False)
    return agent


def ppo_view(observation):
    """Converte a obs multiagente (6 canais) para a do PPO (5 canais)."""
    stack, height, width = observation.shape
    frames = observation.reshape(stack // 6, 6, height, width)
    return frames[:, :5].reshape(-1, height, width)


def happo_act(agent, observation, actor_index):
    """Ação gulosa do ator `actor_index` do HAPPO para uma observação."""
    x = torch.as_tensor(
        np.asarray(observation), dtype=torch.float32,
        device=agent.device).unsqueeze(0)
    logits = agent._masked_logits(
        agent.online.actor_logits(x, actor_index), actor_index)
    return int(logits.argmax(dim=1).item())


def record_frame(writer):
    import pygame
    import pygame.surfarray  # noqa: F401
    frame = np.transpose(
        pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
    writer.append_data(frame)


def main(argv=None) -> None:
    args = parse_args(argv)
    ppo_alg, ppo_meta = algorithm_from_checkpoint(args.ppo_checkpoint)
    happo_alg, happo_meta = algorithm_from_checkpoint(args.happo_checkpoint)
    if ppo_meta["algorithm"] != "ppo":
        raise SystemExit("--ppo-checkpoint não é PPO")
    if happo_meta["algorithm"] != "happo":
        raise SystemExit("--happo-checkpoint não é HAPPO")

    ppo_cfg = ppo_meta.get("config") or {}
    happo_cfg = happo_meta.get("config") or {}
    for key in ("frame_stack", "frame_skip", "level"):
        if ppo_cfg.get(key) != happo_cfg.get(key):
            raise SystemExit(
                f"checkpoints incompatíveis em {key}: "
                f"PPO={ppo_cfg.get(key)!r} HAPPO={happo_cfg.get(key)!r}")

    cfg = ArenaConfig()
    apply_env_config(cfg, happo_meta)
    scenario = Scenario.all_vs_all(2)
    cfg.n_agents = scenario.n_agents
    cfg.team_ids = tuple(scenario.team_ids)
    env = ArenaMultiEnv(cfg, scenario=scenario, render_mode="human")

    ppo_meta["path"] = args.ppo_checkpoint
    ppo_agent = load_agent(
        ppo_alg, ppo_meta, n_agents=scenario.n_agents,
        team_ids=scenario.team_ids, device=args.device)
    happo_meta["path"] = args.happo_checkpoint
    happo_agent = load_agent(
        happo_alg, happo_meta,
        n_agents=int(happo_cfg.get("n_agents", 4)),
        team_ids=tuple(happo_cfg.get("team_ids") or ()),
        device=args.device)

    writer = None
    if args.record:
        import imageio
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            args.record, fps=args.fps or cfg.fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", "23",
                           "-movflags", "+faststart"])

    print(f"PPO vs HAPPO no nível {cfg.level}: "
          f"fs={cfg.frame_stack} fk={cfg.frame_skip}")
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            if writer is not None:
                env.render()
                record_frame(writer)
            totals = np.zeros(2)
            steps = 0
            while True:
                actions = np.array([
                    ppo_agent.act(ppo_view(obs[0]), greedy=True),
                    happo_act(happo_agent, obs[1], 1),
                ], dtype=np.int64)
                obs, rewards, terminated, truncated, info = env.step(
                    actions)
                totals += rewards
                steps += 1
                if writer is not None:
                    env.render()
                    record_frame(writer)
                if bool(np.all(truncated)):
                    break
            coins = [a["coins"] for a in info["agents"]]
            winner = ("PPO" if coins[0] > coins[1] else
                      "HAPPO" if coins[1] > coins[0] else "empate")
            print(f"episódio {episode + 1} ({steps} passos): "
                  f"PPO coins={coins[0]} reward={totals[0]:.2f} | "
                  f"HAPPO coins={coins[1]} reward={totals[1]:.2f} -> {winner}")
    finally:
        if writer is not None:
            writer.close()
            print(f"vídeo salvo em {args.record}")
        env.close()


if __name__ == "__main__":
    main()

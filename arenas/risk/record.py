"""Gravação de um episódio guloso da arena risk em MP4."""
from __future__ import annotations

import inspect
from pathlib import Path

import imageio.v2 as imageio

from .config import RiskConfig
from .env import RiskEnv
from .renderer import RiskRenderer


def record_episode(agent, config: RiskConfig, path, *, seed: int = 0,
                   fps: int = 4, max_frames: int = 160) -> dict:
    """Roda um episódio guloso e grava o tabuleiro quadro a quadro."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    env = RiskEnv(config=config)
    renderer = RiskRenderer()
    obs, _ = env.reset(seed=seed)

    accepts_mask = "action_mask" in inspect.signature(agent.act).parameters
    writer = imageio.get_writer(
        str(path), fps=fps, macro_block_size=None)
    frames = 0
    try:
        writer.append_data(renderer.frame(env.game))
        frames += 1
        done = False
        while not done and frames < max_frames:
            if accepts_mask:
                action = agent.act(
                    obs, greedy=True, action_mask=env.action_masks())
            else:
                action = agent.act(obs, greedy=True)
            obs, _, terminated, truncated, _ = env.step(action)
            writer.append_data(renderer.frame(env.game))
            frames += 1
            done = bool(terminated or truncated)
    finally:
        writer.close()
        env.close()

    return {
        "path": str(path),
        "frames": frames,
        "winner": env.game.winner,
        "turns": env.game.turn,
    }

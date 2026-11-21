"""Avaliação gulosa e gravação de vídeo de episódios."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def greedy_evaluation(agent, env, episodes, seed):
    """Roda `episodes` episódios gulosos e devolve médias/desvios.

    O ambiente é dedicado e as seeds são fixas (`seed + i`, como no
    `evaluate.py`), então a avaliação é comparável entre algoritmos, seeds de
    treino e retomadas. `act(..., greedy=True)` não usa ε nem atualiza pesos;
    `reset(seed=...)` semeia os RNGs do próprio ambiente, de modo que os RNGs
    do treino não são consumidos.
    """
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, info = 0.0, {}
        while True:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rows.append({
            "reward": total,
            "coins": info.get("coins", 0),
            "deaths": info.get("deaths", 0),
        })
    rewards = np.array([r["reward"] for r in rows], dtype=float)
    coins = np.array([r["coins"] for r in rows], dtype=float)
    deaths = np.array([r["deaths"] for r in rows], dtype=float)
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "coins_mean": float(coins.mean()),
        "coins_std": float(coins.std()),
        "deaths_mean": float(deaths.mean()),
        "deaths_std": float(deaths.std()),
    }


def multi_greedy_evaluation(agent, env, episodes, seed):
    """Avaliação gulosa multiagente com seeds fixas (`seed + i`).

    Devolve médias/desvios globais e médias por time, no mesmo espírito de
    `greedy_evaluation`, mas consumindo ações conjuntas.
    """
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        totals = np.zeros(env.n_agents, dtype=np.float64)
        agents = []
        while True:
            actions = agent.act_batch(obs, greedy=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            totals += rewards
            agents = info.get("agents", [])
            if bool(np.all(truncated)):
                break
        rows.append({
            "reward": totals,
            "coins": np.array([a["coins"] for a in agents], dtype=float),
            "deaths": np.array([a["deaths"] for a in agents], dtype=float),
        })
    rewards = np.stack([r["reward"] for r in rows])
    coins = np.stack([r["coins"] for r in rows])
    deaths = np.stack([r["deaths"] for r in rows])
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "coins_mean": float(coins.mean()),
        "coins_std": float(coins.std()),
        "deaths_mean": float(deaths.mean()),
        "deaths_std": float(deaths.std()),
    }
    for team in sorted(set(env.scenario.team_ids)):
        idx = [i for i, t in enumerate(env.scenario.team_ids) if t == team]
        metrics[f"team{team}_reward_mean"] = float(rewards[:, idx].mean())
        metrics[f"team{team}_coins_mean"] = float(coins[:, idx].mean())
        metrics[f"team{team}_deaths_mean"] = float(deaths[:, idx].mean())
    return metrics


def record_multi_episode(agent, env, path, seed):
    """Grava um episódio guloso em MP4 e devolve o caminho.

    Usa um ambiente dedicado e `act_batch(greedy=True)`: não consome o RNG do
    treino. Headless com `SDL_VIDEODRIVER=dummy` (definido pelo chamador).
    """
    try:
        import imageio
        import pygame
        import pygame.surfarray  # noqa: F401 — submódulo precisa de import explícito
    except ImportError as exc:
        raise SystemExit(
            "Gravar vídeo requer: pip install imageio imageio-ffmpeg") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), fps=env.cfg.fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "23",
                       "-movflags", "+faststart"])

    def append_frame():
        frame = np.transpose(
            pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
        writer.append_data(frame)

    try:
        obs, _ = env.reset(seed=seed)
        env.render()
        append_frame()
        while True:
            actions = agent.act_batch(obs, greedy=True)
            obs, _, _, truncated, _ = env.step(actions)
            append_frame()
            if bool(np.all(truncated)):
                break
    finally:
        writer.close()
    return path

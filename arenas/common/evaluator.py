"""Avaliação gulosa de um agente em qualquer arena."""
from __future__ import annotations

import inspect

import numpy as np


def evaluate_agent(agent, env, episodes: int, seed: int,
                   metric_fn=None, print_fn=print) -> dict:
    """Roda `episodes` episódios gulosos e agrega retorno e métricas."""
    returns: list[float] = []
    rows: list[dict] = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        total = 0.0
        done = (isinstance(info, dict) and info.get("winner") is not None)
        accepts_mask = False
        try:
            accepts_mask = "action_mask" in inspect.signature(
                agent.act).parameters
        except (TypeError, ValueError):
            accepts_mask = False
        while not done:
            if accepts_mask and hasattr(env, "action_masks"):
                action = agent.act(
                    obs, greedy=True, action_mask=env.action_masks())
            else:
                action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
        if metric_fn is not None:
            rows.append(metric_fn(info))
        print_fn(f"episode={episode + 1} reward={total:.2f}")

    result = {
        "episodes": episodes,
        "reward_mean": float(np.mean(returns)) if returns else 0.0,
        "reward_std": float(np.std(returns)) if returns else 0.0,
    }
    if rows:
        for key in rows[0]:
            values = [row[key] for row in rows
                      if key in row and row[key] is not None]
            if not values:
                continue
            numeric = all(isinstance(value, (int, float, np.integer,
                                             np.floating))
                          for value in values)
            if numeric:
                result[f"{key}_mean"] = float(np.mean(values))
                result[f"{key}_std"] = float(np.std(values))
    return result

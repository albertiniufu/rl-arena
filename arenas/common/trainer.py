"""Loop de treino single-agent comum a risk e economy.

O seek_coin mantém o seu trainer histórico (currículo de oponentes,
multiagente, vetorização, TensorBoard). Este módulo cobre as arenas novas:
um ambiente Gymnasium, um agente do contrato `arenas.algs` e checkpoints
pesos-only no mesmo formato.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch

from arenas.algs.alg_registry import load_checkpoint, save_checkpoint


def schedule_epsilon(agent, episode: int, ramp: int, epsilon_min: float):
    """Rampa linear por episódio (só para algoritmos off-policy)."""
    if getattr(agent, "on_policy", False):
        return
    if ramp <= 0:
        return  # mantém o decay por passo do próprio agente
    fraction = min(1.0, max(0.0, (episode - 1) / ramp))
    agent.epsilon = 1.0 - fraction * (1.0 - epsilon_min)


def _resolve_resume(resume, checkpoint_dir: Path) -> Path:
    path = Path(resume)
    if not path.exists():
        path = checkpoint_dir / path
    if not path.exists():
        raise SystemExit(f"Checkpoint de retomada não encontrado: {resume}")
    return path


def train_agent(env, agent, *, config, episodes: int, seed: int,
                checkpoint_dir="checkpoints", log_path="logs/training.csv",
                metric_fn=None, metric_keys=(), learn_every: int = 4,
                save_every: int = 10, epsilon_min: float = 0.1,
                epsilon_ramp_episodes: int = 150, resume=None,
                print_fn=print) -> dict:
    """Treina `agent` em `env` e grava `best.pt`/`latest.pt`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_episode, best = 1, -float("inf")
    if resume is not None:
        payload = load_checkpoint(
            _resolve_resume(resume, checkpoint_dir), agent, resume=True)
        done = payload.get("episodes")
        if done is None:
            done = payload.get("extra", {}).get("episodes_done", 0)
        start_episode = int(done) + 1
        best = float(payload.get("extra", {}).get(
            "best_reward", -float("inf")))

    if getattr(agent, "on_policy", False):
        learn_every = 1
    metric_keys = tuple(metric_keys)
    fieldnames = ["episode", "steps", "reward", "mean_loss", "epsilon",
                  *metric_keys]
    new_file = not log_path.exists() or resume is None

    last_episode = start_episode - 1
    last_reward = 0.0
    with log_path.open("w" if new_file else "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for episode in range(start_episode, start_episode + episodes):
            schedule_epsilon(
                agent, episode, epsilon_ramp_episodes, epsilon_min)
            obs, info = env.reset(seed=seed + episode)
            total = 0.0
            losses: list[float] = []
            steps = 0
            learn_counter = 0
            done = (isinstance(info, dict)
                    and info.get("winner") is not None)
            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                agent.buffer.add(obs, action, reward, next_obs, terminated)
                learn_counter += 1
                if learn_counter % learn_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(float(loss))
                obs = next_obs
                total += float(reward)
                steps += 1
                done = bool(terminated or truncated)

            if hasattr(agent, "end_episode"):
                end_loss = agent.end_episode(info)
                if end_loss is not None:
                    losses.append(float(end_loss))

            metrics = metric_fn(info) if metric_fn is not None else {}
            row = {
                "episode": episode,
                "steps": steps,
                "reward": total,
                "mean_loss": (float(np.mean(losses)) if losses else ""),
                "epsilon": getattr(agent, "epsilon", 0.0),
            }
            row.update({key: metrics.get(key, "") for key in metric_keys})
            writer.writerow(row)
            handle.flush()

            last_episode, last_reward = episode, total
            if total > best:
                best = total
                save_checkpoint(
                    checkpoint_dir / "best.pt", agent, config,
                    {"episode": episode, "reward": total,
                     "episodes_done": episode, "best_reward": best},
                    episodes=episode)
            if save_every > 0 and episode % save_every == 0:
                save_checkpoint(
                    checkpoint_dir / "latest.pt", agent, config,
                    {"episode": episode, "reward": total,
                     "episodes_done": episode, "best_reward": best},
                    episodes=episode)

            extras = " ".join(
                f"{key}={metrics[key]}" for key in metric_keys
                if key in metrics)
            print_fn(f"episode={episode} reward={total:.3f} "
                     f"steps={steps} epsilon={row['epsilon']:.3f} "
                     f"loss={row['mean_loss']} {extras}".rstrip())

    if hasattr(agent, "finish_training"):
        tail = agent.finish_training()
        if tail is not None:
            print_fn(f"finish_training loss={float(tail):.5f}")
    save_checkpoint(
        checkpoint_dir / "latest.pt", agent, config,
        {"episode": last_episode, "reward": last_reward,
         "episodes_done": last_episode, "best_reward": best},
        episodes=last_episode)
    env.close()
    return {"episodes_done": last_episode, "best_reward": best}

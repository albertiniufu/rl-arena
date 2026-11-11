"""Loop de treino single-agent comum a risk e economy.

O seek_coin mantém o seu trainer histórico (currículo de oponentes,
multiagente, vetorização, TensorBoard). Este módulo cobre as arenas novas:
um ambiente Gymnasium, um agente do contrato `arenas.algs` e checkpoints
pesos-only no mesmo formato.

Suporta orçamento por tempo de parede (`max_hours`/`deadline_epoch`),
TensorBoard por episódio, gravação periódica de vídeo (via `video_fn`) e
snapshots periódicos de checkpoint.
"""
from __future__ import annotations

import csv
import inspect
import random
import time
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


def _make_tensorboard_writer(enabled, tensorboard_dir, run_name, log_path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard não está instalado. "
            "Instale com: pip install tensorboard") from exc
    name = run_name or Path(log_path).stem
    return SummaryWriter(str(Path(tensorboard_dir) / name))


def train_agent(env, agent, *, config, episodes: int, seed: int,
                checkpoint_dir="checkpoints", log_path="logs/training.csv",
                metric_fn=None, metric_keys=(), learn_every: int = 4,
                save_every: int = 10, epsilon_min: float = 0.1,
                epsilon_ramp_episodes: int = 150, resume=None,
                print_fn=print, tensorboard: bool = False,
                tensorboard_dir="logs/runs", run_name=None,
                max_hours: float | None = None,
                deadline_epoch: float | None = None,
                save_every_minutes: float = 0.0,
                video_every_minutes: float = 0.0, video_fn=None,
                video_dir=None) -> dict:
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
    accepts_mask = False
    try:
        accepts_mask = "action_mask" in inspect.signature(
            agent.act).parameters
    except (TypeError, ValueError):
        accepts_mask = False
    env_masks = hasattr(env, "action_masks")
    buffer_masks = bool(getattr(agent.buffer, "accepts_masks", False))
    metric_keys = tuple(metric_keys)
    fieldnames = ["episode", "steps", "reward", "mean_loss", "epsilon",
                  *metric_keys]
    new_file = not log_path.exists() or resume is None

    start_moment = time.monotonic()
    budget_seconds = None
    if max_hours is not None:
        budget_seconds = float(max_hours) * 3600.0
    if deadline_epoch is not None:
        remaining = float(deadline_epoch) - time.time()
        budget_seconds = (remaining if budget_seconds is None
                          else min(budget_seconds, remaining))
    if budget_seconds is not None and budget_seconds <= 0:
        print_fn("orçamento de tempo já esgotado; nada a treinar")
        episodes = 0

    writer = _make_tensorboard_writer(
        tensorboard, tensorboard_dir, run_name, log_path)

    video_root = None
    last_video = start_moment
    if video_every_minutes > 0 and video_fn is not None:
        video_root = (Path(video_dir) if video_dir is not None
                      else checkpoint_dir / "videos")
        video_root.mkdir(parents=True, exist_ok=True)

    last_timed_save = start_moment
    stopped_by_budget = False
    last_episode = start_episode - 1
    last_reward = 0.0

    with log_path.open("w" if new_file else "a", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer_csv.writeheader()

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
                mask = env.action_masks() if env_masks else None
                if accepts_mask and mask is not None:
                    action = agent.act(obs, action_mask=mask)
                else:
                    action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                next_mask = env.action_masks() if env_masks else None
                if buffer_masks and next_mask is not None:
                    agent.buffer.add(
                        obs, action, reward, next_obs, terminated,
                        next_mask=next_mask)
                else:
                    agent.buffer.add(
                        obs, action, reward, next_obs, terminated)
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
            mean_loss = float(np.mean(losses)) if losses else ""
            row = {
                "episode": episode,
                "steps": steps,
                "reward": total,
                "mean_loss": mean_loss,
                "epsilon": getattr(agent, "epsilon", 0.0),
            }
            row.update({key: metrics.get(key, "") for key in metric_keys})
            writer_csv.writerow(row)
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

            if writer is not None:
                writer.add_scalar("train/reward", total, episode)
                writer.add_scalar("train/steps", steps, episode)
                writer.add_scalar(
                    "train/epsilon", getattr(agent, "epsilon", 0.0), episode)
                writer.add_scalar(
                    "train/mean_loss",
                    float(mean_loss) if mean_loss != "" else 0.0, episode)
                writer.add_scalar("train/agent_steps", agent.steps, episode)
                for key in metric_keys:
                    value = metrics.get(key)
                    if isinstance(value, (int, float, np.integer,
                                          np.floating)):
                        writer.add_scalar(f"train/{key}", float(value),
                                          episode)
                writer.flush()

            now = time.monotonic()
            if (video_root is not None
                    and now - last_video >= video_every_minutes * 60.0):
                video_path = video_root / f"ep{episode:05d}.mp4"
                print_fn(f"gravando vídeo: {video_path}")
                try:
                    video_fn(episode, video_path)
                except Exception as exc:  # vídeo não pode derrubar o treino
                    print_fn(f"falha ao gravar vídeo: {exc}")
                last_video = time.monotonic()

            if (save_every_minutes > 0
                    and time.monotonic() - last_timed_save
                    >= save_every_minutes * 60.0):
                save_checkpoint(
                    checkpoint_dir / "latest.pt", agent, config,
                    {"episode": episode, "reward": total,
                     "episodes_done": episode, "best_reward": best},
                    episodes=episode)
                last_timed_save = time.monotonic()

            extras = " ".join(
                f"{key}={metrics[key]}" for key in metric_keys
                if key in metrics)
            print_fn(f"episode={episode} reward={total:.3f} "
                     f"steps={steps} epsilon={row['epsilon']:.3f} "
                     f"loss={row['mean_loss']} {extras}".rstrip())

            if budget_seconds is not None:
                elapsed = time.monotonic() - start_moment
                if elapsed >= budget_seconds:
                    print_fn(f"orçamento atingido após o episódio "
                             f"{episode}; encerrando")
                    stopped_by_budget = True
                    break

    if hasattr(agent, "finish_training"):
        tail = agent.finish_training()
        if tail is not None:
            print_fn(f"finish_training loss={float(tail):.5f}")
    save_checkpoint(
        checkpoint_dir / "latest.pt", agent, config,
        {"episode": last_episode, "reward": last_reward,
         "episodes_done": last_episode, "best_reward": best},
        episodes=last_episode)
    if writer is not None:
        writer.close()
    env.close()
    return {"episodes_done": last_episode, "best_reward": best,
            "stopped_by_budget": stopped_by_budget}

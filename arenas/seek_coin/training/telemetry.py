"""Telemetria do treino: CSV, estatísticas, checkpoints e imagem da observação."""
from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from arenas.algs.alg_registry import save_checkpoint


def obs_to_image(obs, n_channels=5):
    """Grade semântica (C, H, W) -> imagem RGB (H, W, 3) para o TensorBoard.

    Com frame stack a observação tem `C * frame_stack` canais; usa apenas os
    últimos `n_channels` (o frame atual). `n_channels=6` no modo multiagente
    (IPPO/MAPPO), em que o canal 5 marca os aliados.
    Canais: 0 plataformas, 1 jogador, 2 oponentes, 3 moedas, 4 mortos/respawn,
    5 aliados (multiagente).
    """
    frame = obs[-n_channels:]
    h, w = frame.shape[1], frame.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        0: (120, 130, 140),
        1: (70, 190, 250),
        2: (240, 90, 95),
        3: (245, 195, 55),
        4: (180, 120, 200),
        5: (120, 230, 140),
        # Canais extras: areia/gelo (nível 2) ou botões/super moeda (nível 3).
        6: (210, 175, 90),
        7: (130, 205, 235),
        8: (255, 215, 80),
    }
    for channel, color in colors.items():
        if channel < frame.shape[0]:
            img[frame[channel] > 0] = color
    return img


@contextmanager
def _training_csv(resume):
    """Abre `logs/training.csv` em modo append (retomada) ou truncado."""
    fieldnames = ["episode", "steps", "reward", "coins", "deaths",
                  "mean_loss", "epsilon"]
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not resume
    with log_path.open("w" if new_file else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        yield f, writer


def _finish_training(agent, cfg, ckpt_dir, total_episodes, last_total, best):
    """Consome rollout parcial (se houver) e grava o `latest.pt` final."""
    if hasattr(agent, "finish_training"):
        tail = agent.finish_training()
        if tail is not None:
            print(f"finish_training loss={float(tail):.5f}")
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": total_episodes, "reward": last_total,
                     "episodes_done": total_episodes, "best_reward": best},
                    episodes=total_episodes)


def _episode_stats(losses, grad_norms, q_means, q_maxes):
    """Agrega as estatísticas opcionais de um episódio (stats/TensorBoard)."""
    return {
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "min_loss": float(np.min(losses)) if losses else 0.0,
        "max_loss": float(np.max(losses)) if losses else 0.0,
        "grad_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "q_mean": float(np.mean(q_means)) if q_means else 0.0,
        "q_max": float(np.max(q_maxes)) if q_maxes else 0.0,
    }


def _record_episode(f, writer, args, agent, cfg, ckpt_dir, budgets,
                    fired_budgets, best, episode, reward, steps, coins,
                    deaths, mean_loss, ep_seconds):
    """Grava a linha do episódio, salva checkpoints e imprime o resumo.

    Usado pelos loops de coleta concorrentes (vetorizado e `--overlap`); os
    checkpoints são salvos sempre na thread principal, nunca na de coleta.
    """
    row = {
        "episode": episode,
        "steps": steps,
        "reward": reward,
        "coins": coins,
        "deaths": deaths,
        "mean_loss": mean_loss,
        "epsilon": agent.epsilon,
    }
    writer.writerow(row)
    f.flush()
    if reward > best:
        best = reward
        save_checkpoint(ckpt_dir / "best.pt", agent, cfg,
                        {"episode": episode, "reward": reward,
                         "episodes_done": episode, "best_reward": best},
                        episodes=episode)
    if args.save_every > 0 and episode % args.save_every == 0:
        save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                        {"episode": episode, "reward": reward,
                         "episodes_done": episode, "best_reward": best},
                        episodes=episode)
    for budget in sorted(budgets - fired_budgets):
        if episode >= budget:
            save_checkpoint(ckpt_dir / "budget" / f"ep{budget:03d}.pt",
                            agent, cfg, {"episode": budget}, episodes=budget)
            fired_budgets.add(budget)
    print(f"episode={episode} reward={reward:.2f} coins={coins} "
          f"deaths={deaths} epsilon={agent.epsilon:.3f} "
          f"time={ep_seconds:.2f}s")
    return best

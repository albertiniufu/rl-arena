"""Treino single-agent na arena risk (DQN, Double DQN, PPO ou A2C).

Suporta orçamento por tempo de parede (`--max-hours`/`--deadline`),
TensorBoard (`--tensorboard`) e vídeos periódicos do tabuleiro
(`--video-every-minutes`).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from arenas.common.agent_factory import build_agent
from arenas.common.trainer import train_agent
from arenas.risk.config import RiskConfig
from arenas.risk.env import RiskEnv

AGENTS = ("dqn", "doubledqn_per_dueling", "ppo", "a2c")


def risk_metrics(info: dict) -> dict:
    winner = info.get("winner")
    return {
        "winner": -1 if winner is None else int(winner),
        "won": int(winner == 0),
        "turn": int(info.get("turn", 0)),
        "invalid_action": int(bool(info.get("invalid_action", False))),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alg", choices=AGENTS, default="dqn")
    parser.add_argument("--episodes", type=int, default=300,
                        help="episódios adicionais ao retomar; com "
                             "--deadline/--max-hours, o tempo manda")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--armies", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learn-every", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--epsilon-ramp-episodes", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-every-minutes", type=float, default=0.0,
                        help="snapshot de latest.pt a cada N minutos")
    parser.add_argument("--checkpoint-dir", default="checkpoints/risk")
    parser.add_argument("--log-dir", default="logs/risk")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.1)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="encerra limpo após N horas de parede")
    parser.add_argument("--deadline", default=None,
                        help="encerra limpo em 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", default="logs/risk/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tensorboard-every", type=int, default=10,
                        help="grava escalares a cada N episódios")
    parser.add_argument("--video-every-minutes", type=float, default=0.0,
                        help="grava um MP4 guloso a cada N minutos")
    parser.add_argument("--video-dir", default=None,
                        help="pasta dos vídeos (padrão: <checkpoint>/videos)")
    parser.add_argument("--video-fps", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    deadline_epoch = None
    if args.deadline:
        try:
            deadline_epoch = datetime.strptime(
                args.deadline, "%Y-%m-%d %H:%M").timestamp()
        except ValueError as exc:
            raise SystemExit(
                "--deadline deve ser 'YYYY-MM-DD HH:MM'") from exc

    config = RiskConfig(
        n_players=args.players,
        initial_armies=args.armies,
        max_turns=args.max_turns,
        seed=args.seed,
        invalid_action_penalty=args.invalid_action_penalty,
    )
    env = RiskEnv(config=config)
    agent = build_agent(
        args.alg, env.observation_space.shape, env.action_space,
        lr=args.lr, batch_size=args.batch_size,
        epsilon_min=args.epsilon_min, warmup=args.warmup,
    )
    print(f"arena=risk alg={args.alg} device={agent.device} "
          f"obs={env.observation_space.shape} actions={env.action_space.n}")

    video_dir = (Path(args.video_dir) if args.video_dir
                 else Path(args.checkpoint_dir) / "videos")
    video_fn = None
    if args.video_every_minutes > 0:
        from arenas.risk.record import record_episode

        def video_fn(episode: int, path: Path) -> None:
            result = record_episode(
                agent, config, path,
                seed=args.seed + episode + 100_000,
                fps=args.video_fps,
            )
            print(f"vídeo ep{episode}: winner={result['winner']} "
                  f"turns={result['turns']} frames={result['frames']}")

    if args.max_hours is not None:
        print(f"orçamento: {args.max_hours:g} h")
    if deadline_epoch is not None:
        print("deadline: " + datetime.fromtimestamp(
            deadline_epoch).strftime("%Y-%m-%d %H:%M"))
    if args.video_every_minutes > 0:
        print(f"vídeos a cada {args.video_every_minutes:g} min em "
              f"{video_dir}")

    train_agent(
        env, agent,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_path=Path(args.log_dir) / "training.csv",
        metric_fn=risk_metrics,
        metric_keys=("winner", "won", "turn", "invalid_action"),
        learn_every=args.learn_every,
        save_every=args.save_every,
        epsilon_min=args.epsilon_min,
        epsilon_ramp_episodes=args.epsilon_ramp_episodes,
        resume=args.resume,
        tensorboard=args.tensorboard,
        tensorboard_dir=args.tensorboard_dir,
        run_name=args.run_name or f"risk_{args.alg}",
        tensorboard_every=args.tensorboard_every,
        max_hours=args.max_hours,
        deadline_epoch=deadline_epoch,
        save_every_minutes=args.save_every_minutes,
        video_every_minutes=args.video_every_minutes,
        video_fn=video_fn,
        video_dir=video_dir,
    )


if __name__ == "__main__":
    main()

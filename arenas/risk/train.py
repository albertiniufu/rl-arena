"""Treino single-agent na arena risk (DQN, Double DQN, PPO ou A2C)."""
from __future__ import annotations

import argparse
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
                        help="episódios adicionais ao retomar")
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
    parser.add_argument("--checkpoint-dir", default="checkpoints/risk")
    parser.add_argument("--log-dir", default="logs/risk")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
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
    )


if __name__ == "__main__":
    main()

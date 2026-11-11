"""Treino do banco-agente na arena economy (PPO contínuo)."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from arenas.common.agent_factory import build_agent
from arenas.common.trainer import train_agent
from arenas.economy.bank import BankConfig
from arenas.economy.bank_env import BankEnv
from arenas.economy.config import EconomyConfig

METRIC_KEYS = ("raroc", "default_rate", "capital", "net_income", "ead_total")


def bank_metrics(info: dict) -> dict:
    metrics = info.get("bank") or {}
    return {key: metrics.get(key) for key in METRIC_KEYS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alg", default="ppo_continuous",
                        choices=["ppo_continuous"])
    parser.add_argument("--episodes", type=int, default=200,
                        help="episódios adicionais ao retomar")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=120,
                        help="passos por episódio (meses simulados)")
    parser.add_argument("--n-customers", type=int, default=300)
    parser.add_argument("--n-firms", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints/economy")
    parser.add_argument("--log-dir", default="logs/economy")
    parser.add_argument("--resume", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    economy_cfg = EconomyConfig(
        n_customers=args.n_customers,
        n_firms=args.n_firms,
        max_steps=args.horizon,
        seed=args.seed,
    )
    bank_cfg = BankConfig(max_steps=args.horizon, seed=args.seed)
    env = BankEnv(economy_cfg, bank_cfg)
    agent = build_agent(
        "ppo_continuous", env.observation_space.shape, env.action_space,
        lr=args.lr, batch_size=args.batch_size,
        rollout_steps=args.rollout_steps, epochs=args.epochs,
    )
    config = SimpleNamespace(
        economy=vars(economy_cfg), bank=vars(bank_cfg))
    print(f"arena=economy alg=ppo_continuous device={agent.device} "
          f"obs={env.observation_space.shape} "
          f"actions={env.action_space.shape}")
    train_agent(
        env, agent,
        config=config,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_path=Path(args.log_dir) / "training.csv",
        metric_fn=bank_metrics,
        metric_keys=METRIC_KEYS,
        learn_every=1,
        save_every=args.save_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

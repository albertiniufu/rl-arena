"""Avaliação gulosa de um checkpoint do banco-agente."""
from __future__ import annotations

import argparse

from arenas.algs.alg_registry import checkpoint_meta
from arenas.common.checkpoint import agent_from_checkpoint
from arenas.common.evaluator import evaluate_agent
from arenas.economy.bank import BankConfig
from arenas.economy.bank_env import BankEnv
from arenas.economy.config import EconomyConfig
from arenas.economy.train import METRIC_KEYS, bank_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = checkpoint_meta(args.checkpoint)["config"] or {}
    economy_cfg = EconomyConfig(**config.get("economy", {}))
    bank_cfg = BankConfig(**config.get("bank", {}))
    env = BankEnv(economy_cfg, bank_cfg)
    agent, _ = agent_from_checkpoint(args.checkpoint, env)
    result = evaluate_agent(
        agent, env, args.episodes, args.seed, metric_fn=bank_metrics)
    print("resultado:")
    for key, value in result.items():
        print(f"  {key}={value:.4f}" if isinstance(value, float)
              else f"  {key}={value}")
    print("métricas do banco: " + ", ".join(METRIC_KEYS))


if __name__ == "__main__":
    main()

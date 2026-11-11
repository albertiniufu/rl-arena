"""Avaliação gulosa de um checkpoint na arena risk."""
from __future__ import annotations

import argparse

from arenas.algs.alg_registry import checkpoint_meta
from arenas.common.checkpoint import agent_from_checkpoint
from arenas.common.evaluator import evaluate_agent
from arenas.risk.config import RiskConfig
from arenas.risk.env import RiskEnv
from arenas.risk.train import risk_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_dict = checkpoint_meta(args.checkpoint)["config"] or {}
    config = RiskConfig(**config_dict)
    env = RiskEnv(config=config)
    agent, _ = agent_from_checkpoint(args.checkpoint, env)
    result = evaluate_agent(
        agent, env, args.episodes, args.seed, metric_fn=risk_metrics)
    print("resultado:")
    for key, value in result.items():
        print(f"  {key}={value:.4f}" if isinstance(value, float)
              else f"  {key}={value}")


if __name__ == "__main__":
    main()

"""Reprodução do banco-agente em texto.

Sem checkpoint, executa a política de exemplo (`bank_demo`). Com
`--checkpoint`, o agente treinado decide aprovação, limite e spread.
"""
from __future__ import annotations

import argparse

import numpy as np

from arenas.algs.alg_registry import checkpoint_meta
from arenas.common.checkpoint import agent_from_checkpoint
from arenas.economy import bank_demo
from arenas.economy.bank import BankConfig
from arenas.economy.bank_env import BankEnv
from arenas.economy.config import EconomyConfig

HEADER = (
    "passo |      EAD | receita líq | perda esp |   opex |   lucro |"
    "  capital |  RAROC | default | recompensa"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy", choices=["risk-aware", "naive", "random"],
                        default="risk-aware")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not args.checkpoint:
        bank_demo.main([
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--every", str(args.every),
            "--policy", args.policy,
        ])
        return

    config = checkpoint_meta(args.checkpoint)["config"] or {}
    economy_cfg = EconomyConfig(**config.get("economy", {}))
    bank_cfg = BankConfig(**config.get("bank", {}))
    env = BankEnv(economy_cfg, bank_cfg)
    agent, _ = agent_from_checkpoint(args.checkpoint, env)
    obs, _ = env.reset(seed=args.seed)

    print(HEADER)
    print("-" * len(HEADER))
    done = False
    step = 0
    while not done and step < args.steps:
        action = agent.act(obs, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        metrics = info["bank"]
        step += 1
        if step % max(1, args.every) == 0 or terminated or truncated:
            print(
                f"{step:5d} | {metrics['ead_total']:8.0f} |"
                f" {metrics['nii']:11.0f} | {metrics['expected_loss']:9.0f} |"
                f" {metrics['opex']:6.0f} | {metrics['net_income']:7.0f} |"
                f" {metrics['capital']:8.0f} | {metrics['raroc']:6.1%} |"
                f" {metrics['default_rate']:6.1%} | {reward:10.3f}")
        done = bool(terminated or truncated)
    print("-" * len(HEADER))
    print(f"RAROC final: {env.bank.last_raroc:.1%} | "
          f"capital: {env.bank.capital:.0f} | "
          f"default: {env.bank.last_default_rate:.2%}")
    env.close()


if __name__ == "__main__":
    main()

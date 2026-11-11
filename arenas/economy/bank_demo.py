"""Demonstração do banco-agente em texto.

Uso (a partir da raiz do repositório):

    python -m games.economy.bank_demo --steps 24 --policy risk-aware
    python -m games.economy.bank_demo --steps 24 --policy naive
"""
from __future__ import annotations

import argparse

import numpy as np

from .bank import BankConfig
from .bank_env import BankEnv
from .config import EconomyConfig

HEADER = (
    "passo |      EAD | receita líq | perda esp |   opex |   lucro |"
    "  capital |  RAROC | default | recompensa"
)


def make_policy(name: str, env: BankEnv, rng: np.random.Generator) -> np.ndarray:
    bank = env.bank
    n = env.bank_cfg.n_groups
    reference = np.array(
        [group.reference_spread for group in bank.groups])
    if name == "naive":
        return np.column_stack([
            np.ones(n), np.ones(n), reference])
    if name == "risk-aware":
        approval = np.array([1.0, 0.85, 0.55, 0.25])[:n]
        limit = np.array([1.0, 0.9, 0.7, 0.5])[:n]
        premium = np.array([0.0, 0.01, 0.03, 0.06])[:n]
        return np.column_stack([approval, limit, reference + premium])
    # random
    return np.column_stack([
        rng.uniform(0.0, 1.0, size=n),
        rng.uniform(0.1, 1.0, size=n),
        rng.uniform(0.0, 0.25, size=n),
    ])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--policy", choices=["risk-aware", "naive", "random"],
                        default="risk-aware")
    args = parser.parse_args(argv)

    env = BankEnv(EconomyConfig(seed=args.seed),
                  BankConfig(seed=args.seed, max_steps=args.steps))
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)

    print("GHs:", " | ".join(info["groups"]))
    print(f"policy: {args.policy} | obs: {obs.shape} | "
          f"ações: {env.bank_cfg.n_groups} GHs × (aprovação, limite, spread)")
    print(HEADER)
    print("-" * len(HEADER))

    rarocs = []
    default_rates = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = make_policy(args.policy, env, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        m = info["bank"]
        rarocs.append(m["raroc"])
        default_rates.append(m["default_rate"])
        step += 1
        if step % max(1, args.every) == 0 or terminated or truncated:
            print(
                f"{step:5d} | {m['ead_total']:8.0f} | {m['nii']:11.0f} |"
                f" {m['expected_loss']:9.0f} | {m['opex']:6.0f} |"
                f" {m['net_income']:7.0f} | {m['capital']:8.0f} |"
                f" {m['raroc']:6.1%} | {m['default_rate']:6.1%} |"
                f" {reward:10.3f}")

    print("-" * len(HEADER))
    print(
        f"RAROC médio: {np.mean(rarocs):.1%} | "
        f"default médio: {np.mean(default_rates):.2%} | "
        f"lucro acumulado: {env.bank.cumulative_net_income:.0f} | "
        f"capital final: {env.bank.capital:.0f}"
        + (" | BANCO FALIU" if env.bank.failed else ""))


if __name__ == "__main__":
    main()

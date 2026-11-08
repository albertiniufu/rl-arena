"""Demonstração em texto do simulador econômico.

Uso (a partir da raiz do repositório):

    python -m games.economy.demo --steps 24
    python -m games.economy.demo --steps 36 --policy intervention
"""
from __future__ import annotations

import argparse

from .config import EconomyConfig
from .model import Economy

HEADER = (
    "passo |     PIB | consume |  infl | desemp |  preço |  dívida |"
    " imposto | gasto |  lucro |  gini"
)


def _row(m: dict) -> str:
    return (
        f"{m['step']:5d} | {m['gdp']:7.1f} | {m['consumption']:7.1f} |"
        f" {m['inflation']:5.1%} | {m['unemployment']:6.1%} |"
        f" {m['price_index']:6.1f} | {m['government_debt']:7.1f} |"
        f" {m['income_tax']:7.1%} | {m['spending_ratio']:5.1%} |"
        f" {m['firm_profit']:6.1f} | {m['gini']:5.2f}"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24,
                        help="número de períodos simulados")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=1,
                        help="imprime a cada N passos")
    parser.add_argument("--policy", choices=["default", "intervention"],
                        default="default",
                        help="policy intervention: governo keynesiano simples")
    args = parser.parse_args(argv)

    cfg = EconomyConfig(seed=args.seed)
    economy = Economy(cfg)
    print(HEADER)
    print("-" * len(HEADER))
    last = None
    for step in range(args.steps):
        if args.policy == "intervention":
            m = economy.metrics()
            # Regra simples: aperta a política em alta de preços e afrouxa
            # quando o desemprego sobe.
            tax = 0.30 if m["inflation"] > cfg.inflation_target else 0.15
            spending = 0.28 if m["unemployment"] > 0.15 else 0.18
            economy.set_policy(income_tax=tax, spending_ratio=spending)
        last = economy.step()
        if step % max(1, args.every) == 0 or step == args.steps - 1:
            print(_row(last))
    if last is not None:
        print("-" * len(HEADER))
        print(
            f"Final: PIB={last['gdp']:.1f} consumo={last['consumption']:.1f} "
            f"inflação={last['inflation']:.1%} "
            f"desemprego={last['unemployment']:.1%} "
            f"dívida={last['government_debt']:.1f} "
            f"gini={last['gini']:.2f}")


if __name__ == "__main__":
    main()

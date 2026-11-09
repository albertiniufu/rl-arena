"""Modelo de banco com concessão de crédito e precificação por GH.

O banco empresta a grupos homogêneos de risco (GH). Para cada grupo, o
agente decide:

- **aprovação**: fração das propostas aprovadas (0 = nega tudo, 1 = aprova
  tudo);
- **limite**: multiplicador da exposição máxima por operação;
- **spread**: taxa cobrada acima do custo de funding.

A rentabilidade é medida por **RAROC líquido**:

    RAROC = (receita líquida de juros - perda esperada - opex) / capital econômico

O capital econômico usa a maior entre o capital regulatório e a perda
inesperada (UL = EAD · LGD · sqrt(PD(1-PD)) · z), com carga extra por
concentração. Perdas realizadas (defaults) reduzem o capital do banco.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RiskGroup:
    """Grupo homogêneo de risco (GH)."""

    name: str
    base_pd: float
    lgd: float
    base_exposure: float
    base_demand: float
    reference_spread: float


@dataclass
class BankConfig:
    """Parâmetros do banco e dos grupos homogêneos de risco."""

    group_names: tuple[str, ...] = (
        "A - baixo risco",
        "B - medio-baixo",
        "C - medio-alto",
        "D - alto risco",
    )
    group_pd: tuple[float, ...] = (0.01, 0.03, 0.08, 0.15)
    group_lgd: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65)
    group_exposure: tuple[float, ...] = (1000.0, 800.0, 600.0, 400.0)
    group_demand: tuple[float, ...] = (180.0, 150.0, 110.0, 70.0)
    group_reference_spread: tuple[float, ...] = (0.04, 0.07, 0.11, 0.16)

    spread_elasticity: float = 3.0
    funding_rate: float = 0.01
    opex_per_loan: float = 2.0
    opex_rate: float = 0.001
    regulatory_capital_ratio: float = 0.08
    unexpected_loss_z: float = 2.326
    concentration_threshold: float = 0.40
    concentration_capital_load: float = 1.0

    initial_capital: float = 100_000.0
    default_penalty: float = 0.5
    failure_penalty: float = 5.0
    max_steps: int = 120
    seed: int = 0

    def __post_init__(self) -> None:
        lengths = {
            len(self.group_names),
            len(self.group_pd),
            len(self.group_lgd),
            len(self.group_exposure),
            len(self.group_demand),
            len(self.group_reference_spread),
        }
        if len(lengths) != 1:
            raise ValueError("todos os campos de grupo devem ter o mesmo tamanho")
        if len(self.group_names) < 1:
            raise ValueError("é preciso ao menos um GH")

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    def groups(self) -> tuple[RiskGroup, ...]:
        return tuple(
            RiskGroup(
                name=name,
                base_pd=float(pd),
                lgd=float(lgd),
                base_exposure=float(exposure),
                base_demand=float(demand),
                reference_spread=float(spread),
            )
            for name, pd, lgd, exposure, demand, spread in zip(
                self.group_names, self.group_pd, self.group_lgd,
                self.group_exposure, self.group_demand,
                self.group_reference_spread,
            )
        )


@dataclass
class Bank:
    """Estado, decisões e métricas do banco."""

    config: BankConfig
    groups: tuple[RiskGroup, ...] = field(init=False)
    rng: np.random.Generator = field(init=False)
    capital: float = field(init=False)
    last_action: np.ndarray = field(init=False)
    last_pd: np.ndarray = field(init=False)
    last_demand_factor: np.ndarray = field(init=False)
    last_raroc: float = field(init=False)
    last_default_rate: float = field(init=False)
    cumulative_net_income: float = field(init=False)
    failed: bool = field(init=False)
    history: list[dict] = field(init=False)

    def __post_init__(self) -> None:
        self.groups = self.config.groups()
        self.rng = np.random.default_rng(self.config.seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.config.n_groups
        self.capital = float(self.config.initial_capital)
        self.last_action = np.array([
            [0.5, 1.0, group.reference_spread] for group in self.groups
        ], dtype=float)
        self.last_pd = np.array(
            [group.base_pd for group in self.groups], dtype=float)
        self.last_demand_factor = np.ones(n, dtype=float)
        self.last_raroc = 0.0
        self.last_default_rate = 0.0
        self.cumulative_net_income = 0.0
        self.failed = False
        self.history = []

    # ------------------------------------------------------------------ #
    # Ambiente macro -> risco e demanda
    # ------------------------------------------------------------------ #
    @staticmethod
    def stress_multiplier(macro: dict, inflation_target: float) -> float:
        """Multiplicador de PD a partir do estado macroeconômico."""
        growth = float(macro.get("growth", 0.0))
        inflation = float(macro.get("inflation", 0.0))
        unemployment = float(macro.get("unemployment", 0.0))
        stress = (3.0 * max(0.0, -growth)
                  + 2.0 * max(0.0, inflation - inflation_target)
                  + 1.5 * max(0.0, unemployment - 0.5))
        return 1.0 + min(stress, 2.0)

    def probability_of_default(self, group: RiskGroup, macro: dict,
                               inflation_target: float) -> float:
        multiplier = self.stress_multiplier(macro, inflation_target)
        return float(np.clip(group.base_pd * multiplier, 0.001, 0.6))

    def demand_factor(self, group: RiskGroup, spread: float,
                      macro: dict, inflation_target: float) -> float:
        growth = float(macro.get("growth", 0.0))
        inflation = float(macro.get("inflation", 0.0))
        unemployment = float(macro.get("unemployment", 0.0))
        price_factor = np.clip(
            1.0 - self.config.spread_elasticity
            * (spread - group.reference_spread), 0.05, 1.2)
        macro_factor = np.clip(
            1.0 + 4.0 * growth
            - 2.0 * max(0.0, inflation - inflation_target)
            - 1.0 * max(0.0, unemployment - 0.5), 0.4, 1.5)
        return float(price_factor * macro_factor)

    # ------------------------------------------------------------------ #
    # Passo
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray, macro: dict,
             inflation_target: float = 0.02) -> dict:
        """Aplica a decisão do banco e devolve as métricas do período.

        `action` tem shape `(n_groups, 3)`: aprovação, limite e spread.
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (self.config.n_groups, 3):
            raise ValueError(
                f"ação deve ter shape ({self.config.n_groups}, 3)")
        approval = np.clip(action[:, 0], 0.0, 1.0)
        limit = np.clip(action[:, 1], 0.1, 1.0)
        spread = np.clip(action[:, 2], 0.0, 0.25)

        approved = np.zeros(self.config.n_groups, dtype=int)
        ead = np.zeros(self.config.n_groups)
        nii = np.zeros(self.config.n_groups)
        expected_loss = np.zeros(self.config.n_groups)
        opex = np.zeros(self.config.n_groups)
        economic_capital = np.zeros(self.config.n_groups)
        realized_loss = np.zeros(self.config.n_groups)
        defaults = np.zeros(self.config.n_groups, dtype=int)
        pd = np.zeros(self.config.n_groups)
        demand_factor = np.zeros(self.config.n_groups)

        for g, group in enumerate(self.groups):
            pd[g] = self.probability_of_default(
                group, macro, inflation_target)
            demand_factor[g] = self.demand_factor(
                group, spread[g], macro, inflation_target)
            applications = max(0.0, group.base_demand * demand_factor[g])
            approved[g] = int(round(applications * approval[g]))
            exposure = group.base_exposure * limit[g]
            ead[g] = approved[g] * exposure

            nii[g] = ead[g] * spread[g]
            expected_loss[g] = ead[g] * pd[g] * group.lgd
            opex[g] = (approved[g] * self.config.opex_per_loan
                       + ead[g] * self.config.opex_rate)
            unexpected = (ead[g] * group.lgd
                          * np.sqrt(pd[g] * (1.0 - pd[g]))
                          * self.config.unexpected_loss_z)
            economic_capital[g] = max(
                ead[g] * self.config.regulatory_capital_ratio, unexpected)

            if approved[g] > 0:
                defaults[g] = int(self.rng.binomial(approved[g], pd[g]))
                realized_loss[g] = defaults[g] * exposure * group.lgd

        total_ead = float(ead.sum())
        shares = ead / total_ead if total_ead > 0 else np.zeros_like(ead)
        excess_concentration = float(np.sum(
            np.maximum(0.0, shares - self.config.concentration_threshold)))
        capital_load = 1.0 + (self.config.concentration_capital_load
                              * excess_concentration)
        economic_capital_total = float(
            economic_capital.sum() * capital_load)

        total_nii = float(nii.sum())
        total_expected_loss = float(expected_loss.sum())
        total_opex = float(opex.sum())
        net_income = total_nii - total_expected_loss - total_opex
        raroc = net_income / max(economic_capital_total, 1.0)

        total_approved = int(approved.sum())
        total_defaults = int(defaults.sum())
        default_rate = (total_defaults / total_approved
                        if total_approved else 0.0)
        total_realized_loss = float(realized_loss.sum())
        self.capital += total_nii - total_opex - total_realized_loss
        capital_ratio = self.capital / max(total_ead, 1.0)
        self.failed = self.capital <= 0.0

        self.last_action = np.column_stack([approval, limit, spread])
        self.last_pd = pd
        self.last_demand_factor = demand_factor
        self.last_raroc = float(raroc)
        self.last_default_rate = float(default_rate)
        self.cumulative_net_income += net_income

        metrics = {
            "approved": approved.tolist(),
            "approved_total": total_approved,
            "ead": ead.tolist(),
            "ead_total": total_ead,
            "pd": pd.tolist(),
            "lgd": [group.lgd for group in self.groups],
            "demand_factor": demand_factor.tolist(),
            "spread": spread.tolist(),
            "nii": total_nii,
            "expected_loss": total_expected_loss,
            "opex": total_opex,
            "net_income": net_income,
            "economic_capital": economic_capital_total,
            "raroc": float(raroc),
            "defaults": defaults.tolist(),
            "default_rate": float(default_rate),
            "realized_loss": total_realized_loss,
            "capital": self.capital,
            "capital_ratio": float(capital_ratio),
            "concentration": excess_concentration,
            "failed": self.failed,
        }
        self.history.append(metrics)
        return metrics

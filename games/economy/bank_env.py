"""Ambiente Gymnasium em que o agente é um banco de crédito.

O agente decide, para cada grupo homogêneo de risco (GH), a **fração de
aprovação**, o **limite** (multiplicador da exposição) e o **spread**.
A economia (`Economy`) roda em segundo plano e fornece o estado macro, que
altera probabilidade de default e demanda por crédito.

Recompensa: RAROC líquido do período (receita de juros - perda esperada -
opex, sobre o capital econômico), descontado por defaults realizados,
exigência de capital e falência do banco.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bank import Bank, BankConfig
from .config import EconomyConfig
from .model import Economy


class BankEnv(gym.Env):
    """Banco decidindo concessão e preço de crédito por GH."""

    metadata = {"render_modes": []}

    def __init__(self, config: EconomyConfig | None = None,
                 bank_config: BankConfig | None = None):
        super().__init__()
        self.cfg = config or EconomyConfig()
        self.bank_cfg = bank_config or BankConfig()
        self.economy = Economy(self.cfg)
        self.bank = Bank(self.bank_cfg)

        n = self.bank_cfg.n_groups
        self.action_space = spaces.Box(
            low=np.tile(np.array([0.0, 0.1, 0.0]), n).astype(np.float32),
            high=np.tile(np.array([1.0, 1.0, 0.25]), n).astype(np.float32),
            dtype=np.float32,
        )
        obs_dim = 5 + 6 * n + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self._episode_steps = 0

    # ------------------------------------------------------------------ #
    # Contrato Gymnasium
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.economy.reset(seed=seed)
        # RNG próprio para os defaults; a seed separa do RNG da economia.
        self.bank.reset(seed=seed)
        self._episode_steps = 0
        return self._observation(), {
            "macro": self.economy.metrics(),
            "bank": None,
            "groups": [group.name for group in self.bank.groups],
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.bank_cfg.n_groups, 3)
        action = np.clip(
            action,
            np.array([0.0, 0.1, 0.0]),
            np.array([1.0, 1.0, 0.25]),
        )
        macro = self.economy.metrics()
        metrics = self.bank.step(
            action, macro, inflation_target=self.cfg.inflation_target)
        self.economy.step()
        self._episode_steps += 1

        reward = self._reward(metrics)
        terminated = bool(metrics["failed"])
        truncated = self._episode_steps >= self.bank_cfg.max_steps
        return self._observation(), reward, terminated, truncated, {
            "bank": metrics,
            "macro": macro,
        }

    # ------------------------------------------------------------------ #
    # Observação e recompensa
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        macro = self.economy.metrics()
        scale = max(1.0, self.cfg.n_customers * self.cfg.base_wage)
        features = [
            macro["gdp"] / scale,
            macro["growth"] * 10.0,
            macro["inflation"],
            macro["unemployment"],
            macro["interest_rate"] * 10.0,
        ]
        for g, group in enumerate(self.bank.groups):
            features.extend([
                self.bank.last_pd[g],
                group.lgd,
                self.bank.last_demand_factor[g],
                self.bank.last_action[g, 0],
                self.bank.last_action[g, 1],
                self.bank.last_action[g, 2] / 0.25,
            ])
        features.extend([
            self.bank.capital / max(1.0, self.bank_cfg.initial_capital),
            self.bank.last_raroc,
            self.bank.last_default_rate,
        ])
        return np.clip(np.array(features, dtype=np.float32),
                       -10.0, 10.0)

    def _reward(self, metrics: dict) -> float:
        reward = float(np.clip(metrics["raroc"], -2.0, 2.0))
        reward -= self.bank_cfg.default_penalty * metrics["default_rate"]
        if metrics["capital_ratio"] < self.bank_cfg.regulatory_capital_ratio:
            reward -= 1.0
        if metrics["failed"]:
            reward -= self.bank_cfg.failure_penalty
        return float(reward)

    # ------------------------------------------------------------------ #
    # Utilitários
    # ------------------------------------------------------------------ #
    def action_labels(self) -> list[str]:
        """Nomes das dimensões do espaço de ação (para logs/depuração)."""
        labels = []
        for group in self.bank.groups:
            labels.extend([
                f"{group.name}: aprovação",
                f"{group.name}: limite",
                f"{group.name}: spread",
            ])
        return labels

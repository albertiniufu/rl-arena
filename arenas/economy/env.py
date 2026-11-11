"""Ambiente Gymnasium do simulador econômico.

O agente é o governo: a cada passo escolhe três instrumentos contínuos —
alíquota de imposto de renda, razão de gasto público/PIB e taxa de juros
sobre a dívida. A recompensa é um bem-estar social que combina PIB per
capita, inflação próxima da meta, desemprego, dívida e desigualdade.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EconomyConfig
from .model import Economy


class EconomyEnv(gym.Env):
    """Governo controlado por RL em uma economia de clientes e empresas."""

    metadata = {"render_modes": []}

    def __init__(self, config: EconomyConfig | None = None):
        super().__init__()
        self.cfg = config or EconomyConfig()
        self.economy = Economy(self.cfg)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.05, 0.0], dtype=np.float32),
            high=np.array([0.60, 0.50, 0.05], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(11,), dtype=np.float32)
        self._episode_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.economy.reset(seed=seed)
        self._episode_steps = 0
        return self._observation(), {"metrics": self.economy.metrics()}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low, self.action_space.high)
        self.economy.set_policy(
            income_tax=float(action[0]),
            spending_ratio=float(action[1]),
            interest_rate=float(action[2]),
        )
        metrics = self.economy.step()
        self._episode_steps += 1
        reward = self._reward(metrics)
        terminated = False
        truncated = self._episode_steps >= self.cfg.max_steps
        return self._observation(), float(reward), terminated, truncated, {
            "metrics": metrics,
        }

    def _observation(self) -> np.ndarray:
        m = self.economy.metrics()
        scale = max(1.0, m["n_customers"] * self.cfg.base_wage)
        obs = np.array([
            m["gdp"] / scale,
            m["inflation"],
            m["unemployment"],
            m["price_index"] / max(1.0, self.cfg.base_wage),
            m["government_debt"] / max(1.0, m["gdp"]),
            m["income_tax"],
            m["spending_ratio"],
            m["interest_rate"],
            m["firm_profit"] / scale,
            m["avg_savings"] / max(1.0, self.cfg.base_wage),
            self._episode_steps / max(1, self.cfg.max_steps),
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _reward(self, m: dict) -> float:
        gdp_per_capita = m["gdp"] / max(1, m["n_customers"])
        welfare = gdp_per_capita / (10.0 * self.cfg.base_wage)
        welfare -= 2.0 * abs(m["inflation"] - self.cfg.inflation_target)
        welfare -= 2.0 * m["unemployment"]
        welfare -= 0.5 * m["government_debt"] / max(1.0, m["gdp"])
        welfare -= m["gini"]
        return welfare

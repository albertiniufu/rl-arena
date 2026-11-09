"""Simulador de sistemas econômicos.

População de clientes, empresas concorrentes com produtos, governo com
política fiscal/monetária e variabilidade macroeconômica (choques, inflação,
desemprego). O motor puro está em `model.py`.

Dois agentes Gymnasium:

- `BankEnv` (principal): o agente é um banco que decide concessão de
  crédito e precificação por grupo homogêneo de risco (GH), com recompensa
  por RAROC líquido.
- `EconomyEnv`: o agente é o governo (imposto, gasto e juros).
"""
from .bank import Bank, BankConfig, RiskGroup
from .bank_env import BankEnv
from .config import EconomyConfig
from .model import Economy

__all__ = [
    "Bank",
    "BankConfig",
    "BankEnv",
    "Economy",
    "EconomyConfig",
    "RiskGroup",
]

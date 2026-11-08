"""Simulador de sistemas econômicos.

População de clientes, empresas concorrentes com produtos, governo com
política fiscal/monetária e variabilidade macroeconômica (choques, inflação,
desemprego). O motor puro está em `model.py`; `env.py` o expõe como um
ambiente Gymnasium para RL.
"""
from .config import EconomyConfig
from .model import Economy

__all__ = ["Economy", "EconomyConfig"]

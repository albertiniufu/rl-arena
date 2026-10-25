"""DQN padrão: rede que devolve Q por ação, replay uniforme e alvo fixo C=10.

O nome do algoritmo na CLI é o sufixo do diretório: `--alg dqn`.
"""
from .agent import DQNAgent, td_target

Agent = DQNAgent

__all__ = ["Agent", "DQNAgent", "td_target"]

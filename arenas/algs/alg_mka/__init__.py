"""DQN-mka: rede que devolve Q por ação, replay uniforme e alvo fixo C=10 com modelo custom.

O nome do algoritmo na CLI é o sufixo do diretório: `--alg mka`.
"""
from .agent import MKAAgent, td_target

Agent = MKAAgent

__all__ = ["Agent", "MKAAgent", "td_target"]

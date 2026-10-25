"""Double DQN + replay priorizado (rank-based) + cabeça dueling.

O nome do algoritmo na CLI é o sufixo do diretório:
`--alg doubledqn_per_dueling`.
"""
from .agent import DoubleDQNPerDuelingAgent, double_td_target

Agent = DoubleDQNPerDuelingAgent

__all__ = ["Agent", "DoubleDQNPerDuelingAgent", "double_td_target"]

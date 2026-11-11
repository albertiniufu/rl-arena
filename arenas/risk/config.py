"""Configuração da arena risk."""
from dataclasses import dataclass


@dataclass
class RiskConfig:
    n_players: int = 2
    initial_armies: int = 20
    max_turns: int = 200
    seed: int = 0
    invalid_action_penalty: float = 0.1

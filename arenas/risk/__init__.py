"""Jogo de conquista inspirado em War/Risk.

Territórios agrupados em continentes, reforços, fases (reforçar, atacar,
fortificar) e combate por dados. O motor puro (`game.py`) serve tanto ao
modo jogável em texto (`play.py`) quanto ao ambiente Gymnasium (`env.py`).
"""
from .board import (
    ADJACENCY,
    ADJACENCY_INDEX,
    CONTINENT_BONUS,
    CONTINENT_OF,
    CONTINENTS,
    TERRITORIES,
    TERRITORY_INDEX,
)
from .game import RiskGame

__all__ = [
    "ADJACENCY",
    "ADJACENCY_INDEX",
    "CONTINENT_BONUS",
    "CONTINENT_OF",
    "CONTINENTS",
    "TERRITORIES",
    "TERRITORY_INDEX",
    "RiskGame",
]

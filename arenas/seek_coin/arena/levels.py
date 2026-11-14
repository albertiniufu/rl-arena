"""Definições dos níveis do seek_coin.

O nível 0 é o mapa histórico (mapa fixo). O nível 1 acrescenta duas
plataformas móveis: a mais alta (topo esquerdo), que sobe e desce, e uma
central, que se desloca horizontalmente. Há também duas moedas fixas no alto,
alcançáveis apenas quando a plataforma mais alta está no ponto máximo
(y=145); na posição baixa (y=185) o pulo não chega. O nível 2 usa o mesmo
mapa do nível 1 e acrescenta hazards temporários: areia (reduz a velocidade)
e gelo (escorrega ~1/5 da largura da tela numa direção aleatória).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MovingPlatform:
    """Plataforma móvel; a posição é calculada a partir de `sim_time`."""

    x: float
    y: float
    w: int
    h: int
    axis: str = "x"  # "x" ou "y"
    amplitude: float = 0.0
    period: float = 6.0
    phase: float = 0.0

    def at(self, t: float):
        offset = self.amplitude * (
            0.5 - 0.5 * math.cos(2.0 * math.pi * t / self.period + self.phase))
        if self.axis == "x":
            return (self.x + offset, self.y, self.w, self.h)
        return (self.x, self.y + offset, self.w, self.h)


STATIC_PLATFORMS = {
    0: (
        (0, 360, 220, 20), (280, 360, 360, 20),
        (70, 285, 160, 14), (350, 285, 150, 14),
        (210, 215, 180, 14), (35, 145, 130, 14),
        (445, 155, 150, 14),
    ),
    1: (
        (0, 360, 220, 20), (280, 360, 360, 20),
        (70, 285, 160, 14), (350, 285, 150, 14),
        (210, 215, 180, 14),
        (445, 155, 150, 14),
    ),
}

# Moedas fixas (x, y) no alto do nível 1. Com a plataforma no topo (y=145),
# a subida até y=60 é ~85 px (dentro do pulo ~105 px); na posição baixa
# (y=185) a subida é ~125 px (fora do alcance).
FIXED_COIN_POSITIONS = {
    0: (),
    1: ((75.0, 60.0), (135.0, 60.0)),
}

# Moedas presas a uma plataforma móvel: (índice em MOVING_PLATFORMS, dx, dy)
# em relação ao centro/topo da plataforma. A moeda acompanha o movimento.
# No nível 1, o índice 1 é a plataforma mais alta.
MOVING_COIN_SPECS = {
    0: (),
    1: (
        (1, -40.0, -30.0),
        (1, 0.0, -30.0),
        (1, 40.0, -30.0),
    ),
}

MOVING_PLATFORMS = {
    0: (),
    # Entre as plataformas de y=215 e y=285; passa pelo corredor central.
    1: (
        # Plataforma central, desloca-se na horizontal.
        MovingPlatform(
            x=280, y=250, w=100, h=14,
            axis="x", amplitude=140.0, period=6.0,
        ),
        # Plataforma mais alta (topo esquerdo), sobe e desce.
        MovingPlatform(
            x=35, y=145, w=130, h=14,
            axis="y", amplitude=40.0, period=5.0,
        ),
    ),
}

# Nível 2: mesmo mapa/moedas do nível 1, com hazards temporários.
STATIC_PLATFORMS[2] = STATIC_PLATFORMS[1]
FIXED_COIN_POSITIONS[2] = FIXED_COIN_POSITIONS[1]
MOVING_COIN_SPECS[2] = MOVING_COIN_SPECS[1]
MOVING_PLATFORMS[2] = MOVING_PLATFORMS[1]

HAZARD_KINDS = {
    0: (),
    1: (),
    2: ("sand", "ice"),
}


def get_level(level: int):
    """Devolve `(plataformas_estáticas, plataformas_móveis)` do nível."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return STATIC_PLATFORMS[level], MOVING_PLATFORMS[level]


def get_fixed_coins(level: int):
    """Moedas fixas do nível (posições no mapa, não presas a plataformas)."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return FIXED_COIN_POSITIONS[level]


def get_moving_coins(level: int):
    """Moedas presas a plataformas móveis: `(moving_index, dx, dy)`."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return MOVING_COIN_SPECS[level]


def get_hazard_kinds(level: int):
    """Tipos de hazard que aparecem no nível (vazio nos níveis 0/1)."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return HAZARD_KINDS[level]

"""Definições dos níveis do seek_coin.

O nível 0 é o mapa histórico (mapa fixo). O nível 1 acrescenta duas
plataformas móveis: a mais alta (topo esquerdo), que sobe e desce, e uma
central, que se desloca horizontalmente. A posição é uma função do tempo de
simulação, então o nível é reproduzível por seed.
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


def get_level(level: int):
    """Devolve `(plataformas_estáticas, plataformas_móveis)` do nível."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return STATIC_PLATFORMS[level], MOVING_PLATFORMS[level]

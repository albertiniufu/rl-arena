"""Mundo procedural infinito do nível 5 (rolagem contínua).

O mundo é gerado em **chunks determinísticos** por `(seed, índice do chunk)`:
a mesma seed produz a mesma sequência de chãos, buracos, plataformas, moedas e
plataformas móveis, independentemente da ordem em que os chunks são pedidos.

As posições geradas usam **coordenadas de mundo** (x cresce sem limite). O
engine converte para coordenadas de tela subtraindo `world_offset`: quando o
agente passa de ~55% da largura, tudo é deslocado para a esquerda e o que
ficou para trás é descartado. Assim o custo por passo e a memória ficam
limitados ao que está na tela (mais uma margem de geração), mesmo num episódio
infinito.

O chão é feito de segmentos com buracos (gaps) entre eles; buracos são
regiões sem colisão — quem cai volta a nascer numa plataforma próxima e paga
`death_penalty`. Areia e gelo reutilizam os hazards do nível 2 (o engine
continua espalhando/renovando os patches sobre as plataformas ativas).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHUNK_WIDTH = 480.0
GROUND_Y = 360.0
GROUND_H = 20.0
# Gera chunks até `width + MARGIN` à frente da tela.
MARGIN = 240.0
# Descarta entidades que terminaram antes desta margem à esquerda.
DROP_MARGIN = 80.0
MAX_GAP = 110.0


@dataclass
class Mover:
    """Plataforma móvel do nível 5, ancorada em uma base que rola junto."""

    platform_index: int
    base_x: float
    base_y: float
    axis: str
    amplitude: float
    period: float
    phase: float

    def at(self, t: float):
        offset = self.amplitude * (
            0.5 - 0.5 * math.cos(2.0 * math.pi * t / self.period + self.phase))
        if self.axis == "x":
            return self.base_x + offset, self.base_y
        return self.base_x, self.base_y + offset


@dataclass(frozen=True)
class MoverSpec:
    """Definição de uma plataforma móvel dentro de um chunk."""

    platform: int  # índice em `ChunkContent.platforms`
    base_x: float
    base_y: float
    axis: str
    amplitude: float
    period: float
    phase: float


@dataclass
class ChunkContent:
    """Entidades do chunk em coordenadas de mundo."""

    platforms: list = field(default_factory=list)   # (x, y, w, h)
    holes: list = field(default_factory=list)       # (x, y, w, h)
    coins: list = field(default_factory=list)       # (x, y)
    movers: list = field(default_factory=list)      # MoverSpec


class ChunkGenerator:
    """Gerador determinístico de chunks por `(seed, índice)`."""

    def __init__(self, seed: int, chunk_width: float = CHUNK_WIDTH,
                 world_height: float = 400.0):
        self.seed = int(seed)
        self.chunk_width = float(chunk_width)
        self.world_height = float(world_height)

    def rng_for(self, index: int):
        # SeedSequence por (seed, índice): o chunk é reproduzível mesmo que
        # seja gerado em outra ordem (ex.: retomada/depuração).
        return np.random.default_rng([self.seed, int(index)])

    def generate(self, index: int) -> ChunkContent:
        rng = self.rng_for(index)
        start = index * self.chunk_width
        end = start + self.chunk_width
        difficulty = min(int(index), 8)
        content = ChunkContent()
        hole_h = max(40.0, self.world_height - GROUND_Y)

        # Chão em segmentos com buracos. Chunks começam e terminam com chão:
        # o gap nunca encosta na borda, então a emenda entre chunks é segura.
        pos = start
        while pos < end - 1e-6:
            remaining = end - pos
            if remaining < 120.0:
                content.platforms.append((pos, GROUND_Y, remaining, GROUND_H))
                break
            seg = float(rng.uniform(150.0, 260.0))
            if index == 0:
                # Área segura no início do episódio: sem buraco perto do spawn.
                seg = max(seg, min(240.0, remaining))
            seg = min(seg, remaining)
            content.platforms.append((pos, GROUND_Y, seg, GROUND_H))
            pos += seg
            if pos >= end - 1e-6:
                break
            max_gap = min(MAX_GAP, 42.0 + 7.0 * difficulty)
            gap = float(rng.uniform(32.0, max(33.0, max_gap)))
            # Deixa pelo menos 80 px de chão no fim do chunk.
            gap = min(gap, end - pos - 80.0)
            if gap >= 30.0:
                content.holes.append((pos, GROUND_Y, gap, hole_h))
                pos += gap

        # Plataformas elevadas (0..2 por chunk), algumas com moedas no topo.
        for _ in range(int(rng.integers(0, 3))):
            width = float(rng.uniform(90.0, 160.0))
            px = float(rng.uniform(start + 30.0,
                                   max(start + 31.0, end - width - 30.0)))
            py = float(rng.choice([285.0, 215.0, 155.0]))
            content.platforms.append((px, py, width, 14.0))

        surfaces = [p for p in content.platforms if p[2] >= 60.0]
        if surfaces:
            for _ in range(int(rng.integers(1, 4))):
                px, py, width, _ = surfaces[int(rng.integers(len(surfaces)))]
                cx = float(rng.uniform(px + 12.0, px + width - 12.0))
                content.coins.append((cx, py - 18.0))

        # Plataforma móvel a cada ~3 chunks (a partir do segundo).
        if index > 0 and index % 3 == 0:
            width = 100.0
            bx = float(rng.uniform(start + 60.0,
                                   end - width - 60.0))
            by = float(rng.choice([250.0, 215.0]))
            axis = "x" if rng.random() < 0.6 else "y"
            amplitude = (float(rng.uniform(50.0, 90.0)) if axis == "x"
                         else float(rng.uniform(25.0, 45.0)))
            content.platforms.append((bx, by, width, 14.0))
            content.movers.append(MoverSpec(
                platform=len(content.platforms) - 1,
                base_x=bx,
                base_y=by,
                axis=axis,
                amplitude=amplitude,
                period=float(rng.uniform(5.0, 8.0)),
                phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            ))
        return content

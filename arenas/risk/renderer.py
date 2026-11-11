"""Renderer simples do tabuleiro de risk para vídeo (pygame, headless).

Desenha os 24 territórios numa grade 6×4 (uma coluna por continente), com
cor do dono, número de exércitos e estado do turno. Não usa `display`:
cria uma `Surface` e converte para array, então funciona com
`SDL_VIDEODRIVER=dummy`.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from .board import (
    CONTINENTS,
    CONTINENT_BONUS,
    TERRITORY_INDEX,
)

WIDTH, HEIGHT = 900, 520
MARGIN = 20
HEADER_H = 60
CELL_W = (WIDTH - 2 * MARGIN) // 6
CELL_H = (HEIGHT - HEADER_H - 2 * MARGIN) // 4

BACKGROUND = (24, 26, 32)
BORDER = (210, 214, 222)
TEXT = (240, 242, 246)
OWNER_COLORS = (
    (66, 133, 200),   # J1 azul
    (200, 72, 72),    # J2 vermelho
    (74, 160, 94),    # J3 verde
    (206, 160, 54),   # J4 amarelo
)


class RiskRenderer:
    """Desenha o estado de um `RiskGame` em uma superfície pygame."""

    def __init__(self):
        pygame.font.init()
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.font = pygame.font.Font(None, 18)
        self.small = pygame.font.Font(None, 15)
        self.header_font = pygame.font.Font(None, 24)

    def draw(self, game) -> None:
        self.surface.fill(BACKGROUND)

        if game.winner is not None:
            status = f"Vencedor: Jogador {game.winner + 1}"
        else:
            status = (f"Turno {game.turn} | Jogador "
                      f"{game.current_player + 1} | fase: {game.phase} | "
                      f"reforcos: {game.reinforcements_left}")
        title = self.header_font.render(status, True, TEXT)
        self.surface.blit(title, (MARGIN, 18))

        for column, (continent, names) in enumerate(CONTINENTS.items()):
            x = MARGIN + column * CELL_W
            label = self.small.render(
                f"{continent} (+{CONTINENT_BONUS[continent]})",
                True, (185, 190, 200))
            self.surface.blit(label, (x + 4, HEADER_H - 8))

            for row, name in enumerate(names):
                territory = TERRITORY_INDEX[name]
                owner = int(game.owner[territory])
                color = (OWNER_COLORS[owner] if 0 <= owner < len(OWNER_COLORS)
                         else (70, 72, 78))
                rect = pygame.Rect(
                    x + 3,
                    HEADER_H + MARGIN + row * CELL_H + 3,
                    CELL_W - 6,
                    CELL_H - 6,
                )
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, BORDER, rect, 1)
                name_text = self.font.render(name, True, TEXT)
                army_text = self.font.render(
                    f"x{int(game.armies[territory])}", True, TEXT)
                self.surface.blit(
                    name_text,
                    (rect.x + 5, rect.y + rect.height // 2 - 18))
                self.surface.blit(
                    army_text,
                    (rect.x + 5, rect.y + rect.height // 2 + 2))

    def frame(self, game) -> np.ndarray:
        """Estado atual como array `(H, W, 3)` uint8."""
        self.draw(game)
        # surfarray devolve (W, H, 3); transpor para (H, W, 3).
        return np.transpose(
            pygame.surfarray.array3d(self.surface), (1, 0, 2))

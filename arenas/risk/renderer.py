"""Renderer do tabuleiro de risk em um mapa-múndi estilizado (pygame).

Os 24 territórios ficam em posições geográficas aproximadas sobre formas de
continentes desenhadas no próprio pygame (sem assets externos). O agente J1
aparece em azul e os bots J2–J4 em vermelho, verde e amarelo. Funciona
headless (`SDL_VIDEODRIVER=dummy`).
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from .board import ADJACENCY_INDEX, CONTINENTS, TERRITORY_INDEX

WIDTH, HEIGHT = 900, 520
OCEAN = (16, 30, 46)
LAND = (38, 52, 62)
LAND_BORDER = (58, 76, 88)
EDGE = (48, 66, 80)
TEXT = (238, 241, 245)
MUTED = (160, 170, 182)
BORDER = (215, 220, 228)
OWNER_COLORS = (
    (66, 133, 200),   # J1 azul (agente)
    (200, 72, 72),    # J2 vermelho
    (74, 160, 94),    # J3 verde
    (206, 160, 54),   # J4 amarelo
)

CONTINENT_SHAPES = {
    "AmericaNorte": [(70, 85), (205, 68), (295, 108), (288, 190),
                     (230, 252), (158, 236), (104, 190), (74, 140)],
    "AmericaSul": [(240, 266), (310, 256), (345, 330), (320, 422),
                   (274, 462), (240, 410), (225, 330)],
    "Europa": [(405, 106), (500, 96), (525, 160), (480, 212),
               (420, 190), (400, 145)],
    "Africa": [(455, 220), (545, 210), (580, 290), (545, 405),
               (490, 395), (460, 310)],
    "Asia": [(530, 85), (790, 75), (820, 175), (760, 245),
             (650, 260), (560, 200), (535, 140)],
    "Oceania": [(690, 285), (825, 280), (845, 360), (785, 425),
                (700, 400), (675, 340)],
}
CONTINENT_LABEL_POS = {
    "AmericaNorte": (150, 82),
    "AmericaSul": (255, 280),
    "Europa": (430, 112),
    "Africa": (480, 230),
    "Asia": (660, 100),
    "Oceania": (725, 300),
}

TERRITORY_POSITIONS = {
    "Alaska": (110, 140),
    "Canada": (205, 120),
    "California": (155, 205),
    "Mexico": (200, 240),
    "Colombia": (255, 300),
    "Brasil": (305, 355),
    "Peru": (255, 360),
    "Argentina": (285, 435),
    "ReinoUnido": (430, 130),
    "Franca": (455, 180),
    "Alemanha": (510, 155),
    "Italia": (515, 200),
    "Egito": (525, 240),
    "Nigeria": (500, 295),
    "Congo": (540, 330),
    "AfricaDoSul": (550, 410),
    "Siberia": (655, 125),
    "China": (690, 200),
    "India": (640, 255),
    "Japao": (775, 190),
    "Indonesia": (705, 310),
    "Australia": (775, 365),
    "NovaGuine": (800, 315),
    "NovaZelandia": (820, 420),
}
SHORT_NAMES = {
    "Alaska": "Alas", "Canada": "Can", "California": "Cal",
    "Mexico": "Mex", "Colombia": "Col", "Brasil": "Bra", "Peru": "Per",
    "Argentina": "Arg", "ReinoUnido": "UK", "Franca": "Fra",
    "Alemanha": "Ale", "Italia": "Ita", "Egito": "Egi",
    "Nigeria": "Nig", "Congo": "Con", "AfricaDoSul": "Afs",
    "Siberia": "Sib", "China": "Chi", "India": "Ind", "Japao": "Jap",
    "Indonesia": "Indo", "Australia": "Aus", "NovaGuine": "NG",
    "NovaZelandia": "NZ",
}


class RiskRenderer:
    """Desenha o estado de um `RiskGame` em um mapa-múndi estilizado."""

    def __init__(self):
        pygame.font.init()
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.title_font = pygame.font.Font(None, 24)
        self.label_font = pygame.font.Font(None, 14)
        self.small_font = pygame.font.Font(None, 12)
        self.army_font = pygame.font.Font(None, 18)

    # ------------------------------------------------------------------ #
    # Fundo
    # ------------------------------------------------------------------ #
    def _draw_background(self) -> None:
        self.surface.fill(OCEAN)
        for continent, points in CONTINENT_SHAPES.items():
            pygame.draw.polygon(self.surface, LAND, points)
            pygame.draw.polygon(self.surface, LAND_BORDER, points, 1)
            label = self.small_font.render(continent, True, MUTED)
            x, y = CONTINENT_LABEL_POS[continent]
            self.surface.blit(label, (x, y))

        # Arestas do grafo (uma vez por par), por baixo dos nós.
        for src, neighbors in ADJACENCY_INDEX.items():
            sx, sy = TERRITORY_POSITIONS[TERRITORY_INDEX_NAMES[src]]
            for dst in neighbors:
                if dst <= src:
                    continue
                dx, dy = TERRITORY_POSITIONS[TERRITORY_INDEX_NAMES[dst]]
                pygame.draw.line(self.surface, EDGE, (sx, sy), (dx, dy), 1)

    # ------------------------------------------------------------------ #
    # Estado
    # ------------------------------------------------------------------ #
    def draw(self, game) -> None:
        self._draw_background()

        if game.winner is not None:
            status = f"Vencedor: Jogador {game.winner + 1}"
        else:
            status = (f"Turno {game.turn} | Jogador "
                      f"{game.current_player + 1} | fase: {game.phase} | "
                      f"reforcos: {game.reinforcements_left}")
        title = self.title_font.render(status, True, TEXT)
        self.surface.blit(title, (14, 10))

        for player in range(game.n_players):
            x = 600 + player * 68
            pygame.draw.circle(self.surface, OWNER_COLORS[player],
                               (x, 20), 7)
            label = self.label_font.render(
                f"J{player + 1}", True, TEXT)
            self.surface.blit(label, (x + 12, 13))

        for name, position in TERRITORY_POSITIONS.items():
            territory = TERRITORY_INDEX[name]
            owner = int(game.owner[territory])
            color = (OWNER_COLORS[owner] if 0 <= owner < len(OWNER_COLORS)
                     else (70, 72, 78))
            pygame.draw.circle(self.surface, color, position, 18)
            pygame.draw.circle(self.surface, BORDER, position, 18, 2)
            army = self.army_font.render(
                str(int(game.armies[territory])), True, TEXT)
            self.surface.blit(
                army, army.get_rect(center=position))
            label = self.small_font.render(
                SHORT_NAMES.get(name, name), True, TEXT)
            self.surface.blit(
                label, label.get_rect(midtop=(position[0],
                                              position[1] + 21)))

        caption = self.small_font.render(
            "J1 = agente RL | J2-J4 = bots", True, MUTED)
        self.surface.blit(caption, (14, HEIGHT - 18))

    def frame(self, game) -> np.ndarray:
        """Estado atual como array `(H, W, 3)` uint8."""
        self.draw(game)
        return np.transpose(
            pygame.surfarray.array3d(self.surface), (1, 0, 2))


# Mapa índice -> nome, usado para posicionar as arestas.
TERRITORY_INDEX_NAMES = {
    index: name for name, index in TERRITORY_INDEX.items()
}

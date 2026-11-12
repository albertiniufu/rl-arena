"""Renderer do tabuleiro de risk em um mapa-múndi estilizado (pygame).

Mapa ampliado: 48 regiões posicionadas geograficamente sobre formas de
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

from .board import ADJACENCY_INDEX, TERRITORY_INDEX

WIDTH, HEIGHT = 1200, 700
OCEAN = (16, 30, 46)
LAND = (38, 52, 62)
LAND_BORDER = (58, 76, 88)
EDGE = (40, 58, 72)
TEXT = (238, 241, 245)
MUTED = (160, 170, 182)
BORDER = (215, 220, 228)
NODE_RADIUS = 13
OWNER_COLORS = (
    (66, 133, 200),   # J1 azul (agente)
    (200, 72, 72),    # J2 vermelho
    (74, 160, 94),    # J3 verde
    (206, 160, 54),   # J4 amarelo
)

CONTINENT_SHAPES = {
    "AmericaNorte": [(70, 90), (250, 70), (380, 110), (370, 260),
                     (300, 360), (190, 330), (100, 250), (70, 170)],
    "AmericaSul": [(320, 370), (450, 360), (490, 450), (460, 580),
                   (410, 660), (350, 610), (330, 500), (310, 430)],
    "Europa": [(570, 100), (760, 90), (830, 170), (790, 250),
               (700, 290), (590, 270), (560, 190)],
    "Africa": [(630, 300), (760, 290), (840, 390), (830, 560),
               (770, 650), (700, 620), (660, 500), (620, 400)],
    "Asia": [(820, 70), (1150, 80), (1180, 220), (1120, 330),
             (1000, 390), (880, 360), (800, 280), (800, 160)],
    "Oceania": [(960, 400), (1160, 390), (1180, 520), (1120, 640),
                (1000, 640), (940, 540)],
}
CONTINENT_LABEL_POS = {
    "AmericaNorte": (150, 80),
    "AmericaSul": (350, 380),
    "Europa": (590, 110),
    "Africa": (650, 310),
    "Asia": (850, 80),
    "Oceania": (1000, 410),
}

TERRITORY_POSITIONS = {
    # América do Norte
    "Alaska": (110, 170),
    "Yukon": (180, 130),
    "Noroeste": (260, 120),
    "Ontario": (300, 180),
    "Quebec": (350, 150),
    "California": (140, 260),
    "Rockies": (220, 230),
    "Texas": (270, 290),
    "Leste": (360, 240),
    "Mexico": (230, 340),
    # América do Sul
    "Colombia": (330, 400),
    "Venezuela": (380, 380),
    "Amazonia": (420, 450),
    "Nordeste": (470, 490),
    "Sudeste": (440, 550),
    "Peru": (350, 500),
    "Argentina": (400, 620),
    # Europa
    "ReinoUnido": (610, 160),
    "Franca": (640, 210),
    "Espanha": (590, 250),
    "Alemanha": (690, 190),
    "Italia": (690, 250),
    "Polonia": (740, 170),
    "Escandinavia": (720, 110),
    "Balcas": (760, 230),
    "Ucrania": (800, 190),
    # África
    "Egito": (720, 330),
    "Libia": (690, 370),
    "Nigeria": (680, 430),
    "AfricaOcidental": (640, 450),
    "Congo": (720, 500),
    "AfricaOriental": (780, 480),
    "Angola": (720, 570),
    "AfricaDoSul": (760, 630),
    "Madagascar": (840, 590),
    # Ásia
    "Dudinka": (880, 110),
    "Siberia": (950, 130),
    "Vladivostok": (1040, 200),
    "China": (950, 260),
    "India": (880, 330),
    "Japao": (1100, 250),
    "OrienteMedio": (800, 310),
    "AsiaCentral": (850, 220),
    "SudesteAsiatico": (980, 350),
    # Oceania
    "Indonesia": (1000, 430),
    "NovaGuine": (1060, 470),
    "Australia": (1080, 550),
    "NovaZelandia": (1140, 630),
}
SHORT_NAMES = {
    "Alaska": "Alas", "Yukon": "Yuk", "Noroeste": "Noro",
    "Ontario": "Ont", "Quebec": "Que", "California": "Cal",
    "Rockies": "Roc", "Texas": "Tex", "Leste": "Les", "Mexico": "Mex",
    "Colombia": "Col", "Venezuela": "Ven", "Amazonia": "Ama",
    "Nordeste": "Nord", "Sudeste": "Sud", "Peru": "Per",
    "Argentina": "Arg",
    "ReinoUnido": "UK", "Franca": "Fra", "Espanha": "Esp",
    "Alemanha": "Ale", "Italia": "Ita", "Polonia": "Pol",
    "Escandinavia": "Esc", "Balcas": "Bal", "Ucrania": "Ucr",
    "Egito": "Egi", "Libia": "Lib", "Nigeria": "Nig",
    "AfricaOcidental": "AOC", "Congo": "Con",
    "AfricaOriental": "AOR", "Angola": "Ang", "AfricaDoSul": "AFS",
    "Madagascar": "Mad",
    "Dudinka": "Dud", "Siberia": "Sib", "Vladivostok": "Vla",
    "China": "Chi", "India": "Ind", "Japao": "Jap",
    "OrienteMedio": "OMe", "AsiaCentral": "ACe",
    "SudesteAsiatico": "SEA",
    "Indonesia": "Indo", "NovaGuine": "NG", "Australia": "Aus",
    "NovaZelandia": "NZ",
}

TERRITORY_INDEX_NAMES = {
    index: name for name, index in TERRITORY_INDEX.items()
}


class RiskRenderer:
    """Desenha o estado de um `RiskGame` em um mapa-múndi estilizado."""

    def __init__(self):
        pygame.font.init()
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.title_font = pygame.font.Font(None, 22)
        self.label_font = pygame.font.Font(None, 14)
        self.small_font = pygame.font.Font(None, 11)
        self.army_font = pygame.font.Font(None, 14)

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
            x = 900 + player * 68
            pygame.draw.circle(self.surface, OWNER_COLORS[player],
                               (x, 20), 7)
            label = self.label_font.render(f"J{player + 1}", True, TEXT)
            self.surface.blit(label, (x + 12, 12))

        for name, position in TERRITORY_POSITIONS.items():
            territory = TERRITORY_INDEX[name]
            owner = int(game.owner[territory])
            color = (OWNER_COLORS[owner] if 0 <= owner < len(OWNER_COLORS)
                     else (70, 72, 78))
            pygame.draw.circle(self.surface, color, position, NODE_RADIUS)
            pygame.draw.circle(
                self.surface, BORDER, position, NODE_RADIUS, 2)
            army = self.army_font.render(
                str(int(game.armies[territory])), True, TEXT)
            self.surface.blit(army, army.get_rect(center=position))
            label = self.small_font.render(
                SHORT_NAMES.get(name, name), True, TEXT)
            self.surface.blit(
                label, label.get_rect(
                    midtop=(position[0], position[1] + NODE_RADIUS + 2)))

        caption = self.small_font.render(
            "J1 = agente RL | J2-J4 = bots", True, MUTED)
        self.surface.blit(caption, (14, HEIGHT - 18))

    def frame(self, game) -> np.ndarray:
        """Estado atual como array `(H, W, 3)` uint8."""
        self.draw(game)
        return np.transpose(
            pygame.surfarray.array3d(self.surface), (1, 0, 2))

import math
import pygame


class ArenaRenderer:
    """Renderer da arena com visual neon, HUD e elementos com profundidade."""

    BG_TOP = (12, 18, 35)
    BG_BOTTOM = (25, 34, 58)
    CYAN = (65, 220, 255)
    RED = (255, 92, 117)
    GOLD = (255, 205, 75)
    WHITE = (235, 243, 255)
    MUTED = (145, 164, 194)

    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("NEON ARENA  |  Reinforcement Learning")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("dejavusans", 17)
        self.small_font = pygame.font.SysFont("dejavusans", 13)
        self.title_font = pygame.font.SysFont("dejavusans", 19, bold=True)
        self.big_font = pygame.font.SysFont("dejavusans", 24, bold=True)

        self._background = self._make_background()
        self._grid = self._make_grid()

    def _make_background(self):
        """Gradiente vertical para evitar o fundo chapado."""
        w, h = self.cfg.width, self.cfg.height
        surface = pygame.Surface((w, h))
        for y in range(h):
            t = y / max(1, h - 1)
            color = tuple(
                int(self.BG_TOP[i] * (1 - t) + self.BG_BOTTOM[i] * t)
                for i in range(3)
            )
            pygame.draw.line(surface, color, (0, y), (w, y))
        return surface

    def _make_grid(self):
        surface = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
        spacing = 32
        for x in range(0, self.cfg.width, spacing):
            pygame.draw.line(surface, (92, 145, 195, 24), (x, 0), (x, self.cfg.height))
        for y in range(0, self.cfg.height, spacing):
            pygame.draw.line(surface, (92, 145, 195, 24), (0, y), (self.cfg.width, y))
        return surface

    def _text(self, text, font, color, pos):
        image = font.render(str(text), True, color)
        self.screen.blit(image, pos)
        return image.get_rect(topleft=pos)

    def _draw_panel(self, rect, fill=(13, 21, 40, 225), border=(74, 103, 145)):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, fill, panel.get_rect(), border_radius=12)
        pygame.draw.rect(panel, (*border, 180), panel.get_rect(), width=1, border_radius=12)
        self.screen.blit(panel, rect.topleft)

    def _draw_platform(self, x, y, w, h):
        rect = pygame.Rect(int(x), int(y), int(w), int(h))
        if rect.width <= 0 or rect.height <= 0:
            return
        # Sombra e corpo da plataforma.
        shadow = rect.move(0, 5)
        pygame.draw.rect(self.screen, (5, 9, 20), shadow, border_radius=5)
        pygame.draw.rect(self.screen, (45, 66, 98), rect, border_radius=5)
        pygame.draw.line(
            self.screen, (91, 207, 235),
            (rect.left + 5, rect.top + 2), (rect.right - 5, rect.top + 2), 2
        )
        pygame.draw.line(
            self.screen, (76, 100, 139),
            (rect.left + 7, rect.bottom - 3), (rect.right - 7, rect.bottom - 3), 1
        )

    def _draw_coin(self, coin, elapsed):
        x, y = int(coin.x), int(coin.y)
        pulse = 1.0 + 0.15 * math.sin(elapsed * 5 + x * 0.03)
        radius = max(5, int(7 * pulse))

        # Halo suave em camadas.
        halo = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
        center = (halo.get_width() // 2, halo.get_height() // 2)
        pygame.draw.circle(halo, (255, 190, 50, 22), center, radius * 2)
        pygame.draw.circle(halo, (255, 190, 50, 35), center, int(radius * 1.5))
        self.screen.blit(halo, (x - center[0], y - center[1]))

        pygame.draw.circle(self.screen, (173, 111, 32), (x + 1, y + 2), radius)
        pygame.draw.circle(self.screen, self.GOLD, (x, y), radius)
        pygame.draw.circle(self.screen, (255, 239, 155), (x - 2, y - 2), max(2, radius // 3))
        pygame.draw.circle(self.screen, (255, 245, 190), (x, y), radius, 1)

    def _draw_character(self, character, index, elapsed):
        color = self.CYAN if index == 0 else self.RED
        dark = (20, 65, 91) if index == 0 else (94, 31, 54)

        if not character.alive:
            label = f"P{index + 1}  respawn {max(0, character.respawn_timer):.1f}s"
            image = self.small_font.render(label, True, color)
            self.screen.blit(image, (12 + index * 150, self.cfg.height - 27))
            return

        size = int(self.cfg.player_size)
        x = int(character.x - size / 2)
        y = int(character.y - size)
        rect = pygame.Rect(x, y, size, size)

        # Sombra, corpo e contorno luminoso.
        pygame.draw.ellipse(
            self.screen, (5, 9, 20),
            (x - 3, y + size - 1, size + 6, max(5, size // 4))
        )
        glow = pygame.Surface((size + 20, size + 20), pygame.SRCALPHA)
        pygame.draw.rect(
            glow, (*color, 35), (5, 5, size + 10, size + 10), border_radius=8
        )
        self.screen.blit(glow, (x - 10, y - 10))

        pygame.draw.rect(self.screen, dark, rect.inflate(4, 4), border_radius=7)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        pygame.draw.rect(self.screen, (245, 252, 255), rect, width=1, border_radius=6)

        # Visor/olhos para dar personalidade aos agentes.
        eye_y = y + max(5, size // 3)
        eye_r = max(2, size // 10)
        eye_offset = max(4, size // 4)
        pygame.draw.circle(self.screen, (8, 24, 40), (int(character.x - eye_offset), eye_y), eye_r)
        pygame.draw.circle(self.screen, (8, 24, 40), (int(character.x + eye_offset), eye_y), eye_r)

        tag = self.small_font.render(f"P{index + 1}", True, color)
        self.screen.blit(tag, (int(character.x - tag.get_width() / 2), y - 19))

    def _draw_hud(self, engine, elapsed, info=None):
        p = engine.characters[0]
        panel_w = min(460, self.cfg.width - 24)
        rect = pygame.Rect(12, 12, panel_w, 76)
        self._draw_panel(rect)

        self._text("NEON ARENA", self.title_font, self.WHITE, (25, 20))
        self._text("RL TRAINING ENVIRONMENT", self.small_font, self.MUTED, (26, 47))

        # Separador vertical.
        pygame.draw.line(self.screen, (66, 88, 125), (190, 24), (190, 76), 1)

        self._text(f"{elapsed:07.1f}s", self.big_font, self.CYAN, (205, 20))
        self._text("TEMPO", self.small_font, self.MUTED, (207, 53))

        stats_x = 300
        stats = [
            ("MOEDAS", getattr(p, "coins", 0), self.GOLD),
            ("MORTES", getattr(p, "deaths", 0), self.RED),
            ("ELIM.", getattr(p, "eliminations", 0), self.CYAN),
        ]
        col_w = max(48, (panel_w - stats_x + 15) // 3)
        for i, (label, value, color) in enumerate(stats):
            x = stats_x + i * col_w
            self._text(str(value), self.font, color, (x, 24))
            self._text(label, self.small_font, self.MUTED, (x, 49))

        # Legenda dos agentes no canto inferior direito.
        legend_w = 174
        legend = pygame.Rect(self.cfg.width - legend_w - 12, self.cfg.height - 43, legend_w, 31)
        self._draw_panel(legend, fill=(13, 21, 40, 205), border=(60, 82, 119))
        pygame.draw.circle(self.screen, self.CYAN, (legend.x + 15, legend.y + 15), 5)
        self._text("Agente", self.small_font, self.WHITE, (legend.x + 25, legend.y + 9))
        pygame.draw.circle(self.screen, self.RED, (legend.x + 91, legend.y + 15), 5)
        self._text("Adversário", self.small_font, self.WHITE, (legend.x + 101, legend.y + 9))

    def render(self, engine, elapsed, info=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.blit(self._background, (0, 0))
        self.screen.blit(self._grid, (0, 0))

        # Linha discreta na base da arena.
        pygame.draw.line(
            self.screen, (48, 79, 117),
            (0, self.cfg.height - 1), (self.cfg.width, self.cfg.height - 1), 2
        )

        for platform in engine.platforms:
            self._draw_platform(*platform)

        for coin in engine.coins:
            self._draw_coin(coin, elapsed)

        for i, character in enumerate(engine.characters):
            self._draw_character(character, i, elapsed)

        self._draw_hud(engine, elapsed, info)
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

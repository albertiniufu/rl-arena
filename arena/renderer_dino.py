import math
import random

import pygame


class ArenaRenderer:
    """Renderizador temático de dinossauros para a arena 2D."""

    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Dino Arena — Sobrevivência Pré-Histórica")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("comicsansms", 22, bold=True)
        self.small_font = pygame.font.SysFont("comicsansms", 16, bold=True)
        self._stars = [
            (random.randrange(cfg.width), random.randrange(max(1, cfg.height // 2)), random.choice([1, 2]))
            for _ in range(max(25, cfg.width // 18))
        ]

    def _text(self, text, pos, color=(255, 248, 220), font=None):
        surface = (font or self.font).render(text, True, color)
        self.screen.blit(surface, pos)

    def _draw_background(self):
        # Céu crepuscular e silhuetas de montanhas.
        self.screen.fill((25, 45, 57))
        pygame.draw.rect(self.screen, (40, 68, 69),
                         (0, int(self.cfg.height * 0.56), self.cfg.width,
                          self.cfg.height))
        for x, y, radius in self._stars:
            pygame.draw.circle(self.screen, (195, 211, 170), (x, y), radius)

        # Lua/sol pré-histórico.
        pygame.draw.circle(self.screen, (246, 190, 105),
                           (self.cfg.width - 75, 75), 30)
        pygame.draw.circle(self.screen, (25, 45, 57),
                           (self.cfg.width - 62, 65), 25)

        w, h = self.cfg.width, self.cfg.height
        pygame.draw.polygon(self.screen, (49, 76, 73), [
            (0, int(h * .63)), (int(w * .16), int(h * .35)),
            (int(w * .31), int(h * .63)), (int(w * .45), int(h * .43)),
            (int(w * .63), int(h * .65)), (int(w * .79), int(h * .39)),
            (w, int(h * .64)), (w, h), (0, h)
        ])
        pygame.draw.polygon(self.screen, (70, 94, 73), [
            (0, int(h * .77)), (int(w * .2), int(h * .57)),
            (int(w * .37), int(h * .78)), (int(w * .58), int(h * .59)),
            (int(w * .78), int(h * .78)), (int(w), int(h * .6)),
            (w, h), (0, h)
        ])

        # Camadas de chão com pequenos tufos de vegetação.
        ground_y = int(h * .84)
        pygame.draw.rect(self.screen, (93, 75, 49),
                         (0, ground_y, w, h - ground_y))
        pygame.draw.line(self.screen, (139, 117, 65),
                         (0, ground_y), (w, ground_y), 4)
        for x in range(12, w, 42):
            y = ground_y + 9 + (x % 17)
            pygame.draw.line(self.screen, (57, 119, 73), (x, y + 12),
                             (x - 5, y), 3)
            pygame.draw.line(self.screen, (57, 119, 73), (x, y + 12),
                             (x + 5, y - 1), 3)

    def _draw_platform(self, rect):
        x, y, w, h = map(int, rect)
        pygame.draw.rect(self.screen, (76, 83, 55), (x, y, w, h), border_radius=5)
        pygame.draw.rect(self.screen, (111, 133, 72), (x, y, w, max(4, h // 4)),
                         border_radius=4)
        for px in range(x + 10, x + w - 5, 24):
            pygame.draw.line(self.screen, (139, 151, 83),
                             (px, y + 2), (px + 5, y + max(4, h // 4)), 2)

    def _draw_coin(self, coin, elapsed):
        x, y = int(coin.x), int(coin.y)
        pulse = int(2 * math.sin(elapsed * 5 + x))
        pygame.draw.circle(self.screen, (74, 51, 27), (x, y + 2), 10 + pulse)
        pygame.draw.circle(self.screen, (255, 203, 74), (x, y), 9 + pulse)
        pygame.draw.circle(self.screen, (255, 239, 151), (x - 2, y - 2), 4)
        pygame.draw.circle(self.screen, (194, 127, 40), (x, y), 9 + pulse, 2)

    def _draw_dinosaur(self, character, color, player_index):
        s = self.cfg.player_size
        x, y = int(character.x), int(character.y)
        # Desenho vetorial simples: corpo, cauda, pernas, braços, cabeça e espinhos.
        body = pygame.Rect(int(x - s * .42), int(y - s * .72),
                           int(s * .72), int(s * .48))
        belly = (174, 191, 124) if player_index == 0 else (223, 151, 112)
        dark = (29, 51, 42) if player_index == 0 else (71, 40, 39)

        # Cauda voltada para trás.
        pygame.draw.polygon(self.screen, color, [
            (body.left + 4, body.centery), (x - s * .82, y - s * .55),
            (body.left + 10, y - s * .30)
        ])
        pygame.draw.ellipse(self.screen, color, body)
        pygame.draw.ellipse(self.screen, belly, (
            int(x - s * .27), int(y - s * .48), int(s * .48), int(s * .20)
        ))

        # Patas.
        leg_w = max(4, int(s * .16))
        pygame.draw.rect(self.screen, dark,
                         (int(x - s * .28), int(y - s * .32), leg_w, int(s * .28)),
                         border_radius=3)
        pygame.draw.rect(self.screen, dark,
                         (int(x + s * .08), int(y - s * .32), leg_w, int(s * .28)),
                         border_radius=3)

        # Braços pequenos.
        pygame.draw.line(self.screen, dark, (int(x + s * .17), int(y - s * .54)),
                         (int(x + s * .34), int(y - s * .43)), max(2, int(s * .07)))

        # Pescoço e cabeça.
        pygame.draw.polygon(self.screen, color, [
            (int(x + s * .12), int(y - s * .55)),
            (int(x + s * .30), int(y - s * .91)),
            (int(x + s * .48), int(y - s * .83)),
            (int(x + s * .35), int(y - s * .53))
        ])
        head = pygame.Rect(int(x + s * .25), int(y - s * .99),
                           int(s * .43), int(s * .31))
        pygame.draw.ellipse(self.screen, color, head)
        pygame.draw.polygon(self.screen, color, [
            (head.left + 5, head.centery), (head.right + 5, head.centery + 4),
            (head.right - 2, head.bottom - 2), (head.left + 4, head.bottom - 2)
        ])
        pygame.draw.circle(self.screen, (255, 247, 213),
                           (head.right - 7, head.top + 9), max(3, int(s * .07)))
        pygame.draw.circle(self.screen, (27, 35, 29),
                           (head.right - 6, head.top + 9), max(2, int(s * .035)))

        # Placas/espinhos nas costas.
        for k in range(4):
            sx = int(x - s * .27 + k * s * .14)
            sy = int(y - s * .72)
            pygame.draw.polygon(self.screen, belly, [
                (sx, sy + 3), (sx + int(s * .07), sy - int(s * .16)),
                (sx + int(s * .14), sy + 3)
            ])

    def render(self, engine, elapsed, info=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self._draw_background()

        for platform in engine.platforms:
            self._draw_platform(platform)

        for coin in engine.coins:
            self._draw_coin(coin, elapsed)

        colors = [(105, 190, 111), (224, 104, 82)]
        for i, character in enumerate(engine.characters):
            color = colors[i % len(colors)]
            if character.alive:
                self._draw_dinosaur(character, color, i)
            else:
                label = f"Renasce em {max(0, character.respawn_timer):.1f}s"
                self._text(label, (int(character.x), self.cfg.height - 27),
                           color, self.small_font)

        # Painel superior legível sem esconder a arena.
        p = engine.characters[0]
        panel = pygame.Surface((min(self.cfg.width - 20, 510), 42), pygame.SRCALPHA)
        panel.fill((12, 25, 29, 205))
        self.screen.blit(panel, (10, 10))
        line = f"MOEDAS {p.coins}   |   MORTES {p.deaths}   |   ELIMINACOES {p.eliminations}"
        self._text(line, (21, 20), (255, 240, 196), self.small_font)
        self._text("DINO ARENA", (16, 58), (201, 224, 151), self.small_font)

        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

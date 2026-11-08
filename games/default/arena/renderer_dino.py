import math
import random

import pygame


class ArenaRenderer:
    """Renderizador temático de dinossauros para a arena 2D."""

    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        # RNG próprio com seed fixo: a decoração é determinística e não
        # desloca o fluxo global de `random` usado pela exploração/buffer.
        self.rng = random.Random(0)
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Dino Arena — Sobrevivência Pré-Histórica")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("comicsansms", 22, bold=True)
        self.small_font = pygame.font.SysFont("comicsansms", 16, bold=True)
        self._stars = [
            (self.rng.randrange(cfg.width), self.rng.randrange(max(1, cfg.height // 2)), self.rng.choice([1, 2]))
            for _ in range(max(25, cfg.width // 18))
        ]
        # Estado puramente visual: não interfere nas observações/recompensas do agente.
        self._last_positions = {}
        self._walk_phase = {}
        self._impact_particles = []
        self._last_deaths = {}

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

    def _spawn_impact(self, x, y, color, strength=1.0):
        count = int(9 + 7 * min(2.0, strength))
        for _ in range(count):
            angle = self.rng.uniform(0, math.tau)
            speed = self.rng.uniform(45, 145) * min(1.7, strength)
            self._impact_particles.append({
                "x": float(x), "y": float(y),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - self.rng.uniform(10, 55),
                "life": self.rng.uniform(0.22, 0.48),
                "max_life": 0.48,
                "color": self.rng.choice([color, (255, 220, 118), (238, 245, 183)]),
                "radius": self.rng.randint(2, 4),
            })

    def _update_particles(self, dt):
        alive = []
        for particle in self._impact_particles:
            particle["life"] -= dt
            if particle["life"] <= 0:
                continue
            particle["x"] += particle["vx"] * dt
            particle["y"] += particle["vy"] * dt
            particle["vy"] += 260 * dt
            alive.append(particle)
        self._impact_particles = alive

    def _draw_particles(self):
        for particle in self._impact_particles:
            ratio = max(0.0, particle["life"] / particle["max_life"])
            radius = max(1, int(particle["radius"] * ratio))
            pygame.draw.circle(
                self.screen, particle["color"],
                (int(particle["x"]), int(particle["y"])), radius
            )

    def _draw_dinosaur(self, character, color, player_index, elapsed, dt):
        s = self.cfg.player_size
        x, y = int(character.x), int(character.y)
        previous = self._last_positions.get(player_index, (x, y))
        dx = x - previous[0]
        moving = abs(dx) > 0.5
        phase = self._walk_phase.get(player_index, 0.0)
        if moving:
            phase += dt * 13.0
        self._walk_phase[player_index] = phase
        leg_swing = math.sin(phase) * s * 0.10 if moving else 0
        bob = abs(math.sin(phase * 2)) * s * 0.025 if moving else math.sin(elapsed * 2 + player_index) * s * 0.012
        y_draw = int(y - bob)
        facing = 1 if dx >= 0 else -1
        # Desenho vetorial simples: corpo, cauda, pernas, braços, cabeça e espinhos.
        body = pygame.Rect(int(x - s * .42), int(y - s * .72),
                           int(s * .72), int(s * .48))
        belly = (174, 191, 124) if player_index == 0 else (223, 151, 112)
        dark = (29, 51, 42) if player_index == 0 else (71, 40, 39)

        # Cauda voltada para trás.
        tail_wag = math.sin(elapsed * 10 + player_index) * s * 0.055
        tail_tip_x = x - facing * s * 0.82
        pygame.draw.polygon(self.screen, color, [
            (int(x - facing * s * .22), int(y_draw - s * .52)),
            (int(tail_tip_x), int(y_draw - s * .55 + tail_wag)),
            (int(x - facing * s * .18), int(y_draw - s * .30))
        ])
        pygame.draw.ellipse(self.screen, color, (
            body.x, int(body.y - bob), body.width, body.height
        ))
        pygame.draw.ellipse(self.screen, belly, (
            int(x - s * .27), int(y_draw - s * .48), int(s * .48), int(s * .20)
        ))

        # Patas.
        leg_w = max(4, int(s * .16))
        leg_y = int(y_draw - s * .32)
        pygame.draw.line(self.screen, dark,
                         (int(x - s * .20), leg_y),
                         (int(x - s * .20 + leg_swing), int(y_draw - s * .04)),
                         leg_w)
        pygame.draw.line(self.screen, dark,
                         (int(x + s * .16), leg_y),
                         (int(x + s * .16 - leg_swing), int(y_draw - s * .04)),
                         leg_w)

        # Braços pequenos.
        pygame.draw.line(self.screen, dark, (int(x + s * .17), int(y_draw - s * .54)),
                         (int(x + s * .34), int(y_draw - s * .43)), max(2, int(s * .07)))

        # Pescoço e cabeça.
        pygame.draw.polygon(self.screen, color, [
            (int(x + s * .12), int(y_draw - s * .55)),
            (int(x + s * .30), int(y_draw - s * .91)),
            (int(x + s * .48), int(y_draw - s * .83)),
            (int(x + s * .35), int(y_draw - s * .53))
        ])
        head = pygame.Rect(int(x + s * .25), int(y_draw - s * .99),
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
            sy = int(y_draw - s * .72)
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
        dt = 1.0 / max(1, self.cfg.fps)
        for i, character in enumerate(engine.characters):
            color = colors[i % len(colors)]
            old = self._last_positions.get(i)
            if old is not None and character.alive and abs(character.x - old[0]) > self.cfg.player_size * 0.7:
                self._spawn_impact(character.x, character.y - self.cfg.player_size * 0.45,
                                   color, abs(character.x - old[0]) / max(1, self.cfg.player_size))
            if not character.alive and self._last_deaths.get(i, False):
                self._spawn_impact(character.x, character.y - self.cfg.player_size * 0.5,
                                   color, 1.5)
            self._last_deaths[i] = bool(character.alive)
            if character.alive:
                self._draw_dinosaur(character, color, i, elapsed, dt)
            else:
                label = f"Renasce em {max(0, character.respawn_timer):.1f}s"
                self._text(label, (int(character.x), self.cfg.height - 27),
                           color, self.small_font)
            self._last_positions[i] = (character.x, character.y)

        self._update_particles(dt)
        self._draw_particles()

        # Painel superior legível sem esconder a arena.
        p = engine.characters[0]
        panel = pygame.Surface((min(self.cfg.width - 20, 510), 42), pygame.SRCALPHA)
        panel.fill((12, 25, 29, 205))
        self.screen.blit(panel, (10, 10))
        line = f"MOEDAS {p.coins}   |   MORTES {p.deaths}"
        self._text(line, (21, 20), (255, 240, 196), self.small_font)
        self._text("DINO ARENA", (16, 58), (201, 224, 151), self.small_font)

        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

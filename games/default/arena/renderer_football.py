import math
import pygame


class ArenaRenderer:
    """Renderizador da arena com visual de partida de futebol."""

    GRASS_DARK = (24, 112, 57)
    GRASS_LIGHT = (31, 132, 65)
    WHITE = (239, 247, 235)
    NAVY = (15, 34, 48)

    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("RL Arena | Futebol de Rua")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("dejavusans", 19, bold=True)
        self.small_font = pygame.font.SysFont("dejavusans", 14, bold=True)
        self.big_font = pygame.font.SysFont("dejavusans", 27, bold=True)
        self._field_surface = self._make_pitch()

    def _make_pitch(self):
        w, h = self.cfg.width, self.cfg.height
        surface = pygame.Surface((w, h))
        surface.fill(self.GRASS_DARK)
        # Faixas alternadas do gramado
        stripe_count = 10
        stripe_w = max(1, w // stripe_count)
        for i in range(stripe_count):
            if i % 2 == 0:
                pygame.draw.rect(surface, self.GRASS_LIGHT,
                                 (i * stripe_w, 0, stripe_w, h))

        # Linhas do campo, adaptadas ao tamanho da janela
        margin = max(18, min(w, h) // 22)
        line_w = max(2, min(w, h) // 260)
        pygame.draw.rect(surface, self.WHITE,
                         (margin, margin, w - 2 * margin, h - 2 * margin), line_w)
        cx, cy = w // 2, h // 2
        pygame.draw.line(surface, self.WHITE, (cx, margin), (cx, h - margin), line_w)
        pygame.draw.circle(surface, self.WHITE, (cx, cy), max(28, h // 9), line_w)
        pygame.draw.circle(surface, self.WHITE, (cx, cy), max(3, line_w + 1))

        # Grandes áreas e pequenas áreas
        box_w = max(70, w // 7)
        box_h = max(100, h // 2)
        small_w = max(30, w // 15)
        small_h = max(55, h // 3)
        for x in (margin, w - margin - box_w):
            pygame.draw.rect(surface, self.WHITE,
                             (x, cy - box_h // 2, box_w, box_h), line_w)
        for x in (margin, w - margin - small_w):
            pygame.draw.rect(surface, self.WHITE,
                             (x, cy - small_h // 2, small_w, small_h), line_w)
        return surface

    def _draw_goal(self, x, y, width, height, facing_right):
        # Trave com rede sutil; decorativa para não alterar a física do jogo.
        s = self.screen
        net = (205, 225, 218)
        pygame.draw.rect(s, net, (x, y, width, height), 2)
        for nx in range(x + 5, x + width, 9):
            pygame.draw.line(s, net, (nx, y), (nx, y + height), 1)
        for ny in range(y + 5, y + height, 9):
            pygame.draw.line(s, net, (x, ny), (x + width, ny), 1)
        pygame.draw.line(s, self.WHITE, (x, y), (x, y + height), 4)
        pygame.draw.line(s, self.WHITE, (x, y), (x + width, y), 4)
        pygame.draw.line(s, self.WHITE, (x, y + height), (x + width, y + height), 4)

    def _draw_football(self, x, y, radius=8):
        x, y = int(x), int(y)
        pygame.draw.circle(self.screen, (10, 18, 20), (x + 1, y + 2), radius + 1)
        pygame.draw.circle(self.screen, (248, 248, 239), (x, y), radius)
        # Painéis pretos estilizados
        pygame.draw.polygon(self.screen, (28, 35, 38), [
            (x, y - radius // 2), (x + radius // 2, y - radius // 5),
            (x + radius // 3, y + radius // 2), (x - radius // 3, y + radius // 2),
            (x - radius // 2, y - radius // 5)])
        pygame.draw.circle(self.screen, (255, 255, 250), (x - radius // 3, y - radius // 2), 2)

    def _draw_player(self, player, index):
        x, y = int(player.x), int(player.y)
        size = max(18, int(self.cfg.player_size))
        # Sombra no gramado
        pygame.draw.ellipse(self.screen, (12, 62, 34),
                            (x - size // 2, y - 5, size + 7, max(7, size // 3)))
        if not player.alive:
            txt = self.small_font.render(
                f"RESERVA {max(0, player.respawn_timer):.1f}s", True, self.WHITE)
            self.screen.blit(txt, txt.get_rect(center=(x, self.cfg.height - 19)))
            return

        # Dois uniformes bem distintos, com gola, mangas e número
        if index == 0:
            shirt, trim, shorts = (28, 119, 232), (250, 218, 55), (20, 49, 122)
            number = "10"
        else:
            shirt, trim, shorts = (225, 48, 54), (250, 250, 250), (128, 24, 35)
            number = "9"

        # Pernas/chuteiras
        pygame.draw.line(self.screen, (245, 231, 190), (x - size // 5, y - 2),
                         (x - size // 4, y + size // 4), max(3, size // 7))
        pygame.draw.line(self.screen, (245, 231, 190), (x + size // 5, y - 2),
                         (x + size // 4, y + size // 4), max(3, size // 7))
        pygame.draw.line(self.screen, shorts, (x - size // 5, y - 4),
                         (x + size // 5, y - 4), max(4, size // 4))
        # Tronco com formato de camisa
        body = pygame.Rect(x - size // 2, y - size, size, int(size * .67))
        pygame.draw.rect(self.screen, shirt, body, border_radius=max(3, size // 7))
        pygame.draw.rect(self.screen, trim, body, max(1, size // 12),
                         border_radius=max(3, size // 7))
        pygame.draw.line(self.screen, trim, (x, body.top + 3),
                         (x, body.bottom - 2), max(2, size // 12))
        # Braços
        pygame.draw.line(self.screen, shirt, (body.left + 2, body.top + 7),
                         (body.left - size // 5, body.top + size // 3), max(4, size // 6))
        pygame.draw.line(self.screen, shirt, (body.right - 2, body.top + 7),
                         (body.right + size // 5, body.top + size // 3), max(4, size // 6))
        # Cabeça e faixa de cabelo
        head_r = max(6, size // 4)
        pygame.draw.circle(self.screen, (224, 174, 126), (x, body.top - head_r + 3), head_r)
        pygame.draw.arc(self.screen, (62, 39, 28),
                        (x - head_r, body.top - 2 * head_r + 3,
                         2 * head_r, 2 * head_r), math.pi, 2 * math.pi, 3)
        n = self.small_font.render(number, True, self.WHITE)
        self.screen.blit(n, n.get_rect(center=(x, body.centery)))

    def _draw_hud(self, engine, elapsed):
        p = engine.characters[0]
        w = self.cfg.width
        # Placar compacto no topo
        panel = pygame.Surface((min(w - 24, 440), 62), pygame.SRCALPHA)
        panel.fill((7, 23, 34, 220))
        px = (w - panel.get_width()) // 2
        self.screen.blit(panel, (px, 10))
        title = self.small_font.render("RL FOOTBALL  •  PARTIDA DE TREINO", True, (171, 220, 188))
        self.screen.blit(title, title.get_rect(center=(w // 2, 24)))
        score = self.big_font.render(f"AZUL  {p.coins}  -  {engine.characters[1].coins}  VERMELHO",
                                     True, self.WHITE)
        self.screen.blit(score, score.get_rect(center=(w // 2, 49)))

        # Estatísticas no rodapé
        footer = pygame.Surface((min(w - 24, 520), 29), pygame.SRCALPHA)
        footer.fill((7, 23, 34, 205))
        fx = (w - footer.get_width()) // 2
        fy = self.cfg.height - 39
        self.screen.blit(footer, (fx, fy))
        stats = self.font.render(
            f"⏱ {elapsed:05.1f}s    ⚽ POSSE: AZUL    🪙 {p.coins}    ☠ {p.deaths}",
            True, self.WHITE)
        self.screen.blit(stats, stats.get_rect(center=(w // 2, fy + 15)))

    def render(self, engine, elapsed, info=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.blit(self._field_surface, (0, 0))

        # Gols ilustrativos nas extremidades
        h = self.cfg.height
        goal_h = max(70, h // 3)
        goal_w = max(13, self.cfg.width // 45)
        self._draw_goal(5, (h - goal_h) // 2, goal_w, goal_h, False)
        self._draw_goal(self.cfg.width - goal_w - 5, (h - goal_h) // 2,
                        goal_w, goal_h, True)

        # Plataformas mantêm sua posição e colisão; ganham aparência de barreiras de treino.
        for x, y, w, h in engine.platforms:
            rect = pygame.Rect(int(x), int(y), int(w), int(h))
            pygame.draw.rect(self.screen, (12, 65, 39), rect.move(0, 4), border_radius=5)
            pygame.draw.rect(self.screen, (231, 235, 218), rect, border_radius=5)
            pygame.draw.rect(self.screen, (48, 87, 104), rect.inflate(-4, -4), border_radius=4)
            for mark_x in range(rect.left + 7, rect.right - 3, 18):
                pygame.draw.line(self.screen, (241, 246, 236),
                                 (mark_x, rect.top + 3), (mark_x, rect.bottom - 3), 2)

        # As moedas são desenhadas como bolas para manter os colecionáveis do jogo.
        for coin in engine.coins:
            self._draw_football(coin.x, coin.y, 8)

        for i, character in enumerate(engine.characters):
            self._draw_player(character, i)

        self._draw_hud(engine, elapsed)
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

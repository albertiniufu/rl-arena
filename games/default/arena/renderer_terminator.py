import pygame
import math

class ArenaRenderer:
    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("SKYNET :: T-800 TARGETING SYSTEM")
        self.clock = pygame.time.Clock()
        
        # Fontes estilo militar/HUD
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 22, bold=True)
        
        # Cores Temáticas "Terminator"
        self.COLOR_BG = (10, 12, 16)
        self.COLOR_HUD_RED = (255, 30, 30)
        self.COLOR_HUD_DARK_RED = (80, 10, 10)
        self.COLOR_PLATFORM = (45, 50, 60)
        self.COLOR_PLATFORM_BORDER = (200, 30, 30)
        self.COLOR_RESISTANCE = (0, 210, 255)  # Humano / Resistência (Azul Neon)
        self.COLOR_TERMINATOR = (160, 165, 175) # T-800 (Metal Prata)
        self.COLOR_PLASMA_CORE = (180, 70, 255) # Célula de Energia/Bateria de Plasma
        self.COLOR_PLASMA_GLOW = (110, 20, 200)

    def _draw_hud_grid(self):
        """Desenha linhas de retícula e overlay vermelho da visão do T-800."""
        # Filtro avermelhado sutil no fundo
        overlay = pygame.Surface((self.cfg.width, self.cfg.height), pygame.SRCALPHA)
        overlay.fill((40, 0, 0, 25))
        self.screen.blit(overlay, (0, 0))

        # Linhas de mira centrais
        cx, cy = self.cfg.width // 2, self.cfg.height // 2
        pygame.draw.line(self.screen, self.COLOR_HUD_DARK_RED, (cx - 40, cy), (cx + 40, cy), 1)
        pygame.draw.line(self.screen, self.COLOR_HUD_DARK_RED, (cx, cy - 40), (cx, cy + 40), 1)
        
        # Borda de enquadramento da tela (HUD Frame)
        margin = 15
        rect = (margin, margin, self.cfg.width - 2 * margin, self.cfg.height - 2 * margin)
        pygame.draw.rect(self.screen, self.COLOR_HUD_DARK_RED, rect, 1)

    def render(self, engine, elapsed, info=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 1. Fundo e HUD de Fundo
        self.screen.fill(self.COLOR_BG)
        self._draw_hud_grid()

        # 2. Plataformas (Instalações da Skynet)
        for x, y, w, h in engine.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (x, y, w, h))
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_BORDER, (x, y, w, h), 1)
            # Linha de brilho superior
            pygame.draw.line(self.screen, (100, 110, 130), (x, y), (x + w, y), 2)

        # 3. Moedas -> Células de Energia de Plasma (Plasma Power Cells)
        pulse = (math.sin(elapsed * 8) + 1) / 2  # Pulsação visual
        for coin in engine.coins:
            cx, cy = int(coin.x), int(coin.y)
            # Brilho externo
            glow_radius = int(8 + pulse * 3)
            pygame.draw.circle(self.screen, self.COLOR_PLASMA_GLOW, (cx, cy), glow_radius)
            # Núcleo da energia
            pygame.draw.circle(self.screen, self.COLOR_PLASMA_CORE, (cx, cy), 5)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 2)

        # 4. Personagens (T-800 e Resistência)
        for i, c in enumerate(engine.characters):
            s = self.cfg.player_size
            px, py = int(c.x - s / 2), int(c.y - s)

            if c.alive:
                if i == 0:
                    # Player 0: Resistência (Alvo com caixa de travamento / Lock-on)
                    pygame.draw.rect(self.screen, self.COLOR_RESISTANCE, (px, py, s, s))
                    # Retícula de mira da Skynet sobre o P1
                    pygame.draw.rect(self.screen, self.COLOR_HUD_RED, (px - 4, py - 4, s + 8, s + 8), 1)
                    pygame.draw.line(self.screen, self.COLOR_HUD_RED, (px - 8, py + s // 2), (px + s + 8, py + s // 2), 1)
                    
                    # Tag de identificação do Alvo
                    tag = self.font.render("TARGET: HUMAN", True, self.COLOR_HUD_RED)
                    self.screen.blit(tag, (px - 10, py - 18))
                else:
                    # Player 1: Exterminador T-800 (Chassi Metálico com olho vermelho)
                    pygame.draw.rect(self.screen, self.COLOR_TERMINATOR, (px, py, s, s))
                    pygame.draw.rect(self.screen, (50, 50, 50), (px, py, s, s), 2)
                    
                    # Olho Vermelho Icônico do T-800 (Vermelho Brilhante)
                    eye_x = px + (s * 3 // 4)  # Olhando para a direita por padrão
                    eye_y = py + (s // 3)
                    pygame.draw.circle(self.screen, self.COLOR_HUD_RED, (eye_x, eye_y), 3)
                    pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 1)
            else:
                # Animação / Status de Reparo (Respawn)
                status_txt = f"SYSTEM REBOOT: {max(0, c.respawn_timer):.1f}s"
                color = self.COLOR_RESISTANCE if i == 0 else self.COLOR_HUD_RED
                txt = self.font.render(status_txt, True, color)
                self.screen.blit(txt, (int(c.x - 60), int(c.y - 10)))

        # 5. Painel HUD de Dados (Cyberdyne Systems Diagnostics)
        p = engine.characters[0]
        
        # Cabeçalho Skynet
        header = self.font_large.render("CYBERDYNE SYSTEMS // T-800 v2.4", True, self.COLOR_HUD_RED)
        self.screen.blit(header, (25, 25))

        # Estatísticas estilo Terminal
        stats_line1 = f"TIME: {elapsed:05.1f}s | ENERGY CELLS: {p.coins:02d}"
        stats_line2 = f"DAMAGE TAKEN: {p.deaths:02d}"

        txt1 = self.font.render(stats_line1, True, (255, 100, 100))
        txt2 = self.font.render(stats_line2, True, (255, 100, 100))
        
        self.screen.blit(txt1, (25, 52))
        self.screen.blit(txt2, (25, 70))

        # Atualização do Display
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()
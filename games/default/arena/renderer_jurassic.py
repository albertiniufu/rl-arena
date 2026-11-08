import pygame
import math

class ArenaRenderer:
    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Jurassic Park - Isla Nublar RL Arena")
        self.clock = pygame.time.Clock()
        
        # Fontes estilizadas
        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        self.title_font = pygame.font.SysFont("impact", 20)

        # Paleta de cores Jurassic refinada
        self.COLOR_BG = (10, 20, 14)           # Selva densa e escura
        self.COLOR_FENCE = (50, 58, 52)        # Cercas industriais
        self.COLOR_WIRE = (220, 180, 40)       # Cerca de alta voltagem
        self.COLOR_EGG = (245, 215, 130)       # Ovo de dinossauro / Âmbar
        self.COLOR_EGG_SPOT = (120, 80, 30)    # Manchas no ovo
        
        # Cores dos Dinossauros (Tons mais vivos e biológicos)
        self.COLOR_RAPTOR = (50, 205, 130)     # Velociraptor (Verde Esmeralda)
        self.COLOR_RAPTOR_ACCENT = (20, 110, 70) 
        
        self.COLOR_TREX = (225, 75, 50)        # T-Rex (Vermelho Vulcânico)
        self.COLOR_TREX_ACCENT = (140, 35, 20)
        
        self.COLOR_HUD_BG = (8, 12, 10)
        self.COLOR_WARNING = (255, 180, 0)

    def _draw_jungle_background(self):
        """Fundo de selva com silhuetas de vegetação em camadas."""
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.cfg.width, 50):
            pygame.draw.polygon(self.screen, (15, 28, 20), [
                (i, self.cfg.height),
                (i + 25, self.cfg.height - 120),
                (i + 50, self.cfg.height)
            ])

    def _draw_fence_platform(self, x, y, w, h):
        """Plataforma estilo cerca elétrica industrial."""
        pygame.draw.rect(self.screen, self.COLOR_FENCE, (x, y, w, h), border_radius=4)
        wire_y = y + h // 2
        pygame.draw.line(self.screen, self.COLOR_WIRE, (x, wire_y), (x + w, wire_y), 2)
        for sx in range(int(x), int(x + w), 24):
            pygame.draw.line(self.screen, (30, 35, 32), (sx, y), (sx, y + h), 2)

    def _draw_dino_egg(self, x, y, radius=8):
        """Ovos de dinossauro texturizados."""
        rect = pygame.Rect(int(x - radius), int(y - radius * 1.3), radius * 2, radius * 2.6)
        pygame.draw.ellipse(self.screen, self.COLOR_EGG, rect)
        pygame.draw.ellipse(self.screen, (200, 160, 90), rect, 1) # Borda suave
        # Manchas
        pygame.draw.circle(self.screen, self.COLOR_EGG_SPOT, (int(x - 2), int(y - 4)), 2)
        pygame.draw.circle(self.screen, self.COLOR_EGG_SPOT, (int(x + 3), int(y + 3)), 2)

    def _draw_dinosaur(self, x, y, size, is_player1):
        """Desenho detalhado e agressivo dos dinossauros (Raptor vs T-Rex)."""
        px, py = int(x - size / 2), int(y - size)
        
        if is_player1:
            # --- VELOCIRAPTOR (Ágil, esguio, crista dorsal) ---
            color = self.COLOR_RAPTOR
            accent = self.COLOR_RAPTOR_ACCENT
            
            # Corpo principal inclinado
            body_rect = pygame.Rect(px, py + 4, size, size - 6)
            pygame.draw.rect(self.screen, color, body_rect, border_radius=6)
            
            # Cabeça esticada para frente
            head_rect = pygame.Rect(px + size - 6, py, 16, 10)
            pygame.draw.rect(self.screen, color, head_rect, border_radius=3)
            
            # Olho reptiliano amarelo cortante
            pygame.draw.circle(self.screen, (255, 240, 0), (px + size + 2, py + 4), 3)
            pygame.draw.line(self.screen, (0, 0, 0), (px + size, py + 3), (px + size + 4, py + 5), 2)
            
            # Cauda longa e elegante para estabilidade
            pygame.draw.polygon(self.screen, accent, [
                (px, py + 10),
                (px - 16, py + 4),
                (px, py + size - 4)
            ])
            
            # Crista / Listras de camuflagem nas costas
            pygame.draw.line(self.screen, accent, (px + 6, py + 4), (px + 6, py), 3)
            pygame.draw.line(self.screen, accent, (px + 14, py + 4), (px + 14, py), 3)

        else:
            # --- T-REX (Robusto, mandíbula forte, imponente) ---
            color = self.COLOR_TREX
            accent = self.COLOR_TREX_ACCENT
            
            # Corpo massivo
            body_rect = pygame.Rect(px - 2, py + 2, size + 4, size - 4)
            pygame.draw.rect(self.screen, color, body_rect, border_radius=8)
            
            # Cabeça grande e quadrada com focinho proeminente
            head_rect = pygame.Rect(px - 10, py - 4, 18, 14)
            pygame.draw.rect(self.screen, color, head_rect, border_radius=4)
            
            # Mandíbula aberta com dentes afiados à mostra
            jaw_rect = pygame.Rect(px - 12, py + 8, 16, 6)
            pygame.draw.rect(self.screen, accent, jaw_rect, border_radius=2)
            # Dentes (triângulos brancos)
            for dt in range(px - 10, px + 2, 4):
                pygame.draw.polygon(self.screen, (255, 255, 255), [
                    (dt, py + 8), (dt + 2, py + 12), (dt + 4, py + 8)
                ])

            # Olho feroz
            pygame.draw.circle(self.screen, (255, 150, 0), (px - 4, py + 2), 3)
            pygame.draw.circle(self.screen, (0, 0, 0), (px - 4, py + 2), 1)

            # Cauda pesada contrabalançando o corpo
            pygame.draw.polygon(self.screen, accent, [
                (px + size, py + 8),
                (px + size + 16, py + size),
                (px + size, py + size)
            ])

    def render(self, engine, elapsed, info=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # 1. Renderiza Cenário de Selva
        self._draw_jungle_background()

        # 2. Renderiza Plataformas (Cercas Eletrificadas)
        for x, y, w, h in engine.platforms:
            self._draw_fence_platform(x, y, w, h)

        # 3. Renderiza Ovos de Dinossauro
        for coin in engine.coins:
            self._draw_dino_egg(coin.x, coin.y)

        # 4. Renderiza Personagens (Dinossauros Estilizados)
        for i, c in enumerate(engine.characters):
            is_p1 = (i == 0)
            color = self.COLOR_RAPTOR if is_p1 else self.COLOR_TREX
            
            if c.alive:
                s = self.cfg.player_size
                self._draw_dinosaur(c.x, c.y, s, is_player1=is_p1)
            else:
                # Alerta de Incubadora / Regen de DNA
                txt = self.font.render(f"[INCUBADORA] {max(0, c.respawn_timer):.1f}s", True, color)
                self.screen.blit(txt, (int(c.x - 50), self.cfg.height - 30))

        # 5. Painel de Controle (HUD) Isla Nublar
        hud_rect = pygame.Rect(0, 0, self.cfg.width, 36)
        pygame.draw.rect(self.screen, self.COLOR_HUD_BG, hud_rect)
        pygame.draw.line(self.screen, self.COLOR_WARNING, (0, 36), (self.cfg.width, 36), 2)

        # Título do Sistema
        title_txt = self.title_font.render("ISLA NUBLAR - BIO-TRACK", True, self.COLOR_WARNING)
        self.screen.blit(title_txt, (10, 5))

        # Estatísticas
        p = engine.characters[0]
        stats_str = f"TEMPO: {elapsed:.1f}s | OVOS: {p.coins} | MORTES: {p.deaths}"
        stats_txt = self.font.render(stats_str, True, (210, 220, 210))
        self.screen.blit(stats_txt, (220, 10))

        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()
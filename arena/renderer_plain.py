import pygame

class ArenaRenderer:
    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("RL Arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)

    def render(self, engine, elapsed, info=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        self.screen.fill((24, 28, 40))
        for x, y, w, h in engine.platforms:
            pygame.draw.rect(self.screen, (95, 105, 125), (x, y, w, h))
        for coin in engine.coins:
            pygame.draw.circle(self.screen, (245, 195, 55), (int(coin.x), int(coin.y)), 7)
        for i, c in enumerate(engine.characters):
            color = (70, 190, 250) if i == 0 else (240, 90, 95)
            if c.alive:
                s = self.cfg.player_size
                pygame.draw.rect(self.screen, color, (int(c.x-s/2), int(c.y-s), s, s))
            else:
                txt = self.font.render(f"respawn {max(0,c.respawn_timer):.1f}s", True, color)
                self.screen.blit(txt, (int(c.x), self.cfg.height - 25))
        p = engine.characters[0]
        line = f"t={elapsed:.1f}s | moedas={p.coins} | mortes={p.deaths} | eliminacoes={p.eliminations}"
        self.screen.blit(self.font.render(line, True, (240, 240, 245)), (10, 10))
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

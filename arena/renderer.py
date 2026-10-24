"""
Renderer animado para o RL Arena (pygame).

Mantém a mesma interface do renderer original:
    ArenaRenderer(cfg).render(engine, elapsed, info=None) / .close()

Tudo é inferido do estado do engine (posições, moedas, mortes, eliminações),
então NÃO é preciso alterar o engine. Efeitos visuais:

  * fundo em gradiente com estrelas piscando, lua, montanhas e névoa
  * personagens com "squash & stretch", olhos que seguem o movimento e rastro
  * partículas: poeira ao correr/pular/pousar, explosão na morte, anel no respawn
  * moedas que giram, flutuam e brilham; explosão + "+1" ao coletar
  * tremor de tela em mortes e eliminações
  * kill feed, coroa para o líder, painel de HUD, minigráfico de histórico
  * painel com o dicionário `info` (recompensa, época, epsilon... o que você passar)

Teclas:  ESPAÇO = pausa | T = rastro | G = brilho | H = ajuda | ESC = sair
"""
import math
import random
from collections import deque

import pygame

# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
PLAYER_COLORS = [(70, 190, 250), (240, 90, 95), (120, 230, 140), (250, 200, 90)]
PLAYER_NAMES = ["Agente", "Rival", "Bot 3", "Bot 4"]
COIN_COLOR = (250, 200, 60)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def shade(color, f):
    return tuple(int(clamp(c * f, 0, 255)) for c in color)


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size", "gravity", "drag")

    def __init__(self, x, y, vx, vy, life, color, size, gravity=0.0, drag=0.98):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color, self.size, self.gravity, self.drag = color, size, gravity, drag

    def update(self, dt):
        self.vx *= self.drag
        self.vy = self.vy * self.drag + self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt


# --------------------------------------------------------------------------- #
# renderer
# --------------------------------------------------------------------------- #
class ArenaRenderer:
    def __init__(self, cfg):
        pygame.init()
        self.cfg = cfg
        self.W, self.H = cfg.width, cfg.height
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("RL Arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.font_small = pygame.font.SysFont(None, 17)
        self.font_big = pygame.font.SysFont(None, 64)

        self.world = pygame.Surface((self.W, self.H))
        self.fx = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        self.dt = 1.0 / max(1, cfg.fps)
        self.t = 0.0
        self.paused = False
        self.show_trails = True
        self.show_glow = True
        self.show_help = False

        self.particles = []
        self.rings = []        # [x, y, r, r_max, life, max_life, color]
        self.floats = []       # [x, y, texto, cor, life]
        self.feed = []         # [texto, cor, ttl]
        self.shake = 0.0
        self.cs = {}           # estado por personagem
        self._coin_pos = None
        self._last_elapsed = 0.0
        self._glow_cache = {}
        self.history = deque(maxlen=160)
        self._hist_timer = 0.0

        rnd = random.Random(7)
        self.stars = [(rnd.randrange(self.W), rnd.randrange(int(self.H * 0.8)),
                       rnd.uniform(0.6, 2.0), rnd.uniform(0, 6.28)) for _ in range(90)]
        self.orbs = [[rnd.uniform(0, self.W), rnd.uniform(0, self.H * 0.7),
                      rnd.uniform(60, 140), rnd.uniform(4, 14)] for _ in range(5)]
        self.bg = self._build_background(rnd)

    # ------------------------------------------------------------------ fundo
    def _build_background(self, rnd):
        bg = pygame.Surface((self.W, self.H))
        top, bottom = (12, 14, 34), (44, 30, 70)
        for y in range(self.H):
            pygame.draw.line(bg, mix(top, bottom, y / self.H), (0, y), (self.W, y))
        pygame.draw.circle(bg, (235, 235, 255), (int(self.W * 0.82), int(self.H * 0.17)), 34)
        pygame.draw.circle(bg, (205, 205, 235), (int(self.W * 0.82) - 9, int(self.H * 0.17) - 7), 6)
        pygame.draw.circle(bg, (205, 205, 235), (int(self.W * 0.82) + 11, int(self.H * 0.17) + 10), 4)
        for base, col, amp in ((0.78, (30, 26, 58), 90), (0.88, (22, 20, 44), 60)):
            pts = [(0, self.H)]
            for x in range(0, self.W + 40, 40):
                pts.append((x, self.H * base - rnd.uniform(0, amp)))
            pts.append((self.W, self.H))
            pygame.draw.polygon(bg, col, pts)
        return bg

    def _glow(self, radius, color):
        key = (radius, color)
        if key not in self._glow_cache:
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for r in range(radius, 0, -2):
                a = int(70 * (1 - r / radius) ** 2)
                pygame.draw.circle(surf, (*color, a), (radius, radius), r)
            self._glow_cache[key] = surf
        return self._glow_cache[key]

    # --------------------------------------------------------------- eventos
    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                raise SystemExit
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif e.key == pygame.K_g:
                    self.show_glow = not self.show_glow
                elif e.key == pygame.K_h:
                    self.show_help = not self.show_help

    # ------------------------------------------------------- efeitos (spawn)
    def _burst(self, x, y, color, n=18, speed=220, life=0.7, size=4, gravity=500):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            v = random.uniform(0.3, 1.0) * speed
            self.particles.append(Particle(x, y, math.cos(a) * v, math.sin(a) * v - speed * 0.2,
                                           random.uniform(0.5, 1.0) * life, color,
                                           random.uniform(0.6, 1.0) * size, gravity))

    def _dust(self, x, y, n=5, spread=1.0):
        for _ in range(n):
            self.particles.append(Particle(x + random.uniform(-6, 6), y - 2,
                                           random.uniform(-60, 60) * spread, random.uniform(-40, -5),
                                           random.uniform(0.25, 0.5), (200, 200, 220), random.uniform(2, 4),
                                           gravity=60, drag=0.93))

    def _ring(self, x, y, color, r_max=60, life=0.5):
        self.rings.append([x, y, 4, r_max, life, life, color])

    def _float(self, x, y, text, color):
        self.floats.append([x, y, text, color, 1.0])

    def _name(self, i):
        return PLAYER_NAMES[i] if i < len(PLAYER_NAMES) else f"Bot {i + 1}"

    def _color(self, i):
        return PLAYER_COLORS[i % len(PLAYER_COLORS)]

    # ---------------------------------------------- observação do estado/engine
    def _new_state(self, c):
        return dict(px=c.x, py=c.y, vx=0.0, vy=0.0, pvy=0.0, alive=c.alive, coins=c.coins,
                    deaths=c.deaths, elims=c.eliminations, sq=1.0, face=1,
                    trail=deque(maxlen=12), flash=0.0, timer_max=1.0)

    def _observe(self, engine, elapsed):
        chars = engine.characters
        s = self.cfg.player_size
        thr = s * 0.15

        if elapsed < self._last_elapsed - 0.5 or len(self.cs) != len(chars):  # novo episódio
            self.cs = {i: self._new_state(c) for i, c in enumerate(chars)}
            self._coin_pos = None
            self.history.clear()
        self._last_elapsed = elapsed

        # moedas coletadas = sumiram perto de algum personagem
        cur = {(round(co.x), round(co.y)) for co in engine.coins}
        if self._coin_pos is not None:
            for (cx, cy) in self._coin_pos - cur:
                best, bd = None, 1e9
                for i, c in enumerate(chars):
                    d = math.hypot(c.x - cx, (c.y - s / 2) - cy)
                    if c.alive and d < bd:
                        best, bd = i, d
                if best is not None and bd < s * 2.5:
                    col = self._color(best)
                    self._burst(cx, cy, COIN_COLOR, n=14, speed=180, life=0.6, size=3.5, gravity=300)
                    self._ring(cx, cy, COIN_COLOR, r_max=34, life=0.35)
                    self._float(cx, cy - 10, "+1", mix(COIN_COLOR, col, 0.25))
        self._coin_pos = cur

        for i, c in enumerate(chars):
            st = self.cs[i]
            col = self._color(i)
            st["pvy"] = st["vy"]
            st["vx"] = c.x - st["px"]
            st["vy"] = c.y - st["py"]
            if abs(st["vx"]) > thr * 0.3:
                st["face"] = 1 if st["vx"] > 0 else -1

            if c.alive:
                grounded = abs(st["vy"]) < 0.5
                if grounded and st["pvy"] > thr:                      # pouso
                    st["sq"] = 0.62
                    self._dust(c.x, c.y, n=9, spread=2.0)
                elif st["vy"] < -thr and st["pvy"] >= -thr * 0.5:     # pulo
                    st["sq"] = 1.35
                    self._dust(c.x, c.y, n=6)
                elif grounded and abs(st["vx"]) > thr * 0.5 and random.random() < 0.35:
                    self._dust(c.x - st["face"] * s * 0.4, c.y, n=1)
                st["trail"].append((c.x, c.y))
            else:
                st["trail"].clear()
                st["timer_max"] = max(st["timer_max"], getattr(c, "respawn_timer", 0) or 0.01)

            if st["alive"] and not c.alive:                            # morreu
                self._burst(st["px"], st["py"] - s / 2, col, n=34, speed=320, life=0.9, size=5)
                self._ring(st["px"], st["py"] - s / 2, col, r_max=90, life=0.5)
                self.shake = max(self.shake, 9)
                st["timer_max"] = 0.01
                killer = next((j for j, o in enumerate(chars)
                               if j != i and o.eliminations > self.cs[j]["elims"]), None)
                if killer is None:
                    self.feed.append([f"{self._name(i)} morreu", col, 4.0])
            if not st["alive"] and c.alive:                            # renasceu
                self._ring(c.x, c.y - s / 2, col, r_max=70, life=0.6)
                self._burst(c.x, c.y - s / 2, col, n=16, speed=150, life=0.6, size=3, gravity=-80)
            if c.eliminations > st["elims"]:                           # eliminação
                self._float(c.x, c.y - s - 12, "ELIMINAÇÃO!", (255, 235, 140))
                self.feed.append([f"{self._name(i)} eliminou o oponente", col, 4.0])
                self.shake = max(self.shake, 12)
            if c.coins != st["coins"] or c.deaths != st["deaths"] or c.eliminations != st["elims"]:
                st["flash"] = 0.5

            st.update(px=c.x, py=c.y, alive=c.alive, coins=c.coins, deaths=c.deaths, elims=c.eliminations)

    # ------------------------------------------------------------ atualização
    def _update(self, dt, engine, info):
        self.t += dt
        for st in self.cs.values():
            st["sq"] = lerp(st["sq"], 1.0, min(1.0, dt * 12))
            st["flash"] = max(0.0, st["flash"] - dt)
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.life > 0][-900:]
        for r in self.rings:
            r[4] -= dt
            r[2] = lerp(4, r[3], 1 - clamp(r[4] / r[5], 0, 1))
        self.rings = [r for r in self.rings if r[4] > 0]
        for f in self.floats:
            f[1] -= 40 * dt
            f[4] -= dt
        self.floats = [f for f in self.floats if f[4] > 0]
        for f in self.feed:
            f[2] -= dt
        self.feed = [f for f in self.feed if f[2] > 0][-5:]
        self.shake *= 0.88
        for o in self.orbs:
            o[0] = (o[0] + o[3] * dt) % (self.W + 200)

        self._hist_timer += dt
        if self._hist_timer > 0.1:
            self._hist_timer = 0
            val = None
            if info and isinstance(info.get("reward"), (int, float)):
                val = float(info["reward"])
            elif engine.characters:
                val = float(engine.characters[0].coins)
            if val is not None:
                self.history.append(val)

    # ----------------------------------------------------------------- desenho
    def _draw_background(self, surf):
        surf.blit(self.bg, (0, 0))
        for x, y, size, ph in self.stars:
            a = 0.5 + 0.5 * math.sin(self.t * 2.0 + ph)
            v = int(110 + 140 * a)
            pygame.draw.circle(surf, (v, v, min(255, v + 20)), (x, y), max(1, int(size * (0.6 + 0.4 * a))))
        for ox, oy, r, _ in self.orbs:
            surf.blit(self._glow(int(r), (120, 90, 200)), (ox - 100 - r, oy - r))

    def _draw_platforms(self, surf, engine):
        for x, y, w, h in engine.platforms:
            pygame.draw.rect(surf, (10, 10, 22), (x + 4, y + 6, w, h), border_radius=5)       # sombra
            pygame.draw.rect(surf, (82, 92, 118), (x, y, w, h), border_radius=5)
            pygame.draw.rect(surf, (60, 68, 92), (x, y + h * 0.55, w, h * 0.45), border_radius=5)
            pygame.draw.rect(surf, (140, 215, 150), (x, y, w, min(5, h)), border_radius=3)    # "grama"
            pygame.draw.line(surf, (190, 245, 195), (x + 3, y + 1), (x + w - 4, y + 1), 1)

    def _draw_coins(self, surf, engine):
        for co in engine.coins:
            ph = (co.x * 0.05) % 6.28
            bob = math.sin(self.t * 3 + ph) * 3
            cy = co.y + bob
            if self.show_glow:
                g = self._glow(22, COIN_COLOR)
                surf.blit(g, (co.x - 22, cy - 22))
            w = max(2, int(abs(math.cos(self.t * 3 + ph)) * 7))
            pygame.draw.ellipse(surf, shade(COIN_COLOR, 0.7), (co.x - w - 1, cy - 8, 2 * w + 2, 16))
            pygame.draw.ellipse(surf, COIN_COLOR, (co.x - w, cy - 7, 2 * w, 14))
            if w > 3:
                pygame.draw.ellipse(surf, (255, 240, 170), (co.x - w * 0.5, cy - 4, w, 8), 1)
            if int(self.t * 4 + ph) % 6 == 0:
                pygame.draw.line(surf, (255, 255, 255), (co.x + 6, cy - 9), (co.x + 6, cy - 3), 1)
                pygame.draw.line(surf, (255, 255, 255), (co.x + 3, cy - 6), (co.x + 9, cy - 6), 1)

    def _draw_character(self, surf, fx, i, c, leader):
        s = self.cfg.player_size
        st = self.cs[i]
        col = self._color(i)

        if self.show_trails and c.alive:
            n = len(st["trail"])
            for k, (tx, ty) in enumerate(st["trail"]):
                a = int(90 * (k + 1) / max(1, n))
                sz = s * (0.5 + 0.5 * (k + 1) / max(1, n))
                pygame.draw.rect(fx, (*col, a), (tx - sz / 2, ty - sz, sz, sz), border_radius=4)

        if not c.alive:
            rt = max(0.0, c.respawn_timer)
            frac = clamp(rt / st["timer_max"], 0, 1)
            cx, cy = int(c.x), self.H - 40
            bob = math.sin(self.t * 4 + i) * 4
            pygame.draw.circle(fx, (*col, 50), (cx, int(cy + bob)), 18)
            rect = pygame.Rect(cx - 18, int(cy + bob) - 18, 36, 36)
            pygame.draw.arc(surf, col, rect, math.pi / 2, math.pi / 2 + math.tau * (1 - frac), 3)
            txt = self.font.render(f"{rt:.1f}s", True, col)
            surf.blit(txt, txt.get_rect(center=(cx, int(cy + bob))))
            return

        sy = clamp(st["sq"], 0.55, 1.45)
        sx = 1 / sy
        w, h = s * sx, s * sy
        rect = pygame.Rect(0, 0, int(w), int(h))
        rect.midbottom = (int(c.x), int(c.y))
        rad = max(3, int(s * 0.25))

        if self.show_glow:
            surf.blit(self._glow(int(s * 1.4), col), (rect.centerx - s * 1.4, rect.centery - s * 1.4))
        pygame.draw.rect(surf, shade(col, 0.6), rect.inflate(4, 4), border_radius=rad + 2)
        pygame.draw.rect(surf, col, rect, border_radius=rad)
        pygame.draw.rect(surf, shade(col, 1.3), (rect.x + 3, rect.y + 3, rect.w - 6, max(2, rect.h // 4)),
                         border_radius=rad)

        # olhos
        er = max(2, int(s * 0.14))
        ey = rect.top + h * 0.38
        for sign in (-1, 1):
            ex = rect.centerx + sign * w * 0.2 + st["face"] * w * 0.05
            pygame.draw.circle(surf, (250, 250, 255), (int(ex), int(ey)), er)
            px = ex + st["face"] * er * 0.45
            py = ey + clamp(st["vy"] * 0.15, -er * 0.4, er * 0.4)
            pygame.draw.circle(surf, (20, 22, 36), (int(px), int(py)), max(1, int(er * 0.55)))

        # nome + coroa do líder
        tag = self.font_small.render(self._name(i), True, (235, 235, 245))
        tr = tag.get_rect(midbottom=(rect.centerx, rect.top - 4))
        surf.blit(tag, tr)
        if leader:
            cx, cy = rect.centerx, tr.top - 2
            pts = [(cx - 9, cy), (cx - 9, cy - 8), (cx - 4, cy - 4), (cx, cy - 11),
                   (cx + 4, cy - 4), (cx + 9, cy - 8), (cx + 9, cy)]
            pygame.draw.polygon(surf, COIN_COLOR, pts)
            pygame.draw.polygon(surf, shade(COIN_COLOR, 0.6), pts, 1)

    def _draw_fx(self, surf, fx):
        for r in self.rings:
            a = int(220 * clamp(r[4] / r[5], 0, 1))
            pygame.draw.circle(fx, (*r[6], a), (int(r[0]), int(r[1])), max(1, int(r[2])), 3)
        for p in self.particles:
            k = clamp(p.life / p.max_life, 0, 1)
            pygame.draw.circle(fx, (*p.color, int(255 * k)), (int(p.x), int(p.y)), max(1, int(p.size * (0.4 + 0.6 * k))))
        surf.blit(fx, (0, 0))
        for x, y, text, color, life in self.floats:
            img = self.font.render(text, True, color)
            img.set_alpha(int(255 * clamp(life * 2, 0, 1)))
            surf.blit(img, img.get_rect(center=(int(x), int(y))))

    # ----------------------------------------------------------------- HUD
    def _panel(self, rect):
        p = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(p, (10, 12, 26, 175), p.get_rect(), border_radius=10)
        pygame.draw.rect(p, (255, 255, 255, 35), p.get_rect(), 1, border_radius=10)
        self.screen.blit(p, rect.topleft)

    def _icon(self, kind, x, y, color):
        if kind == "coin":
            pygame.draw.circle(self.screen, COIN_COLOR, (x, y), 6)
            pygame.draw.circle(self.screen, shade(COIN_COLOR, 0.6), (x, y), 6, 1)
        elif kind == "death":
            pygame.draw.line(self.screen, (240, 110, 120), (x - 5, y - 5), (x + 5, y + 5), 3)
            pygame.draw.line(self.screen, (240, 110, 120), (x - 5, y + 5), (x + 5, y - 5), 3)
        else:
            pts = [(x, y - 7), (x + 2, y - 2), (x + 7, y - 2), (x + 3, y + 2), (x + 5, y + 7),
                   (x, y + 4), (x - 5, y + 7), (x - 3, y + 2), (x - 7, y - 2), (x - 2, y - 2)]
            pygame.draw.polygon(self.screen, (255, 235, 140), pts)

    def _draw_hud(self, engine, elapsed, info):
        chars = engine.characters
        row_h = 30
        panel = pygame.Rect(10, 10, 300, 14 + row_h * len(chars) + 30)
        self._panel(panel)
        for i, c in enumerate(chars):
            st = self.cs[i]
            y = panel.y + 14 + i * row_h
            col = self._color(i)
            pygame.draw.rect(self.screen, col, (panel.x + 12, y - 8, 14, 14), border_radius=4)
            self.screen.blit(self.font.render(self._name(i), True, (235, 235, 245)), (panel.x + 32, y - 8))
            fl = clamp(st["flash"] / 0.5, 0, 1)
            numcol = mix((235, 235, 245), (255, 230, 90), fl)
            for k, (kind, val) in enumerate((("coin", c.coins), ("death", c.deaths), ("elim", c.eliminations))):
                x = panel.x + 118 + k * 62
                self._icon(kind, x, y, col)
                self.screen.blit(self.font.render(str(val), True, numcol), (x + 12, y - 8))

        # tempo + minigráfico
        ty = panel.bottom - 28
        self.screen.blit(self.font.render(f"t = {elapsed:5.1f}s", True, (200, 205, 225)), (panel.x + 12, ty))
        self._sparkline(pygame.Rect(panel.x + 112, ty - 2, panel.w - 124, 22))

        # painel do info (direita)
        items = []
        if info:
            for k, v in info.items():
                items.append((str(k), f"{v:.3f}" if isinstance(v, float) else str(v)))
        items.append(("fps", f"{self.clock.get_fps():.0f}"))
        pw = 190
        pr = pygame.Rect(self.W - pw - 10, 10, pw, 12 + 20 * len(items))
        self._panel(pr)
        for n, (k, v) in enumerate(items):
            self.screen.blit(self.font_small.render(k, True, (150, 158, 190)), (pr.x + 10, pr.y + 9 + n * 20))
            img = self.font_small.render(v, True, (235, 235, 245))
            self.screen.blit(img, (pr.right - 10 - img.get_width(), pr.y + 9 + n * 20))

        # kill feed
        y = pr.bottom + 10
        for text, col, ttl in self.feed:
            img = self.font.render(text, True, col)
            img.set_alpha(int(255 * clamp(ttl, 0, 1)))
            bg = pygame.Surface((img.get_width() + 16, 24), pygame.SRCALPHA)
            pygame.draw.rect(bg, (10, 12, 26, int(170 * clamp(ttl, 0, 1))), bg.get_rect(), border_radius=8)
            self.screen.blit(bg, (self.W - bg.get_width() - 10, y))
            self.screen.blit(img, (self.W - bg.get_width() - 2, y + 4))
            y += 28

        hint = "H: ajuda" if not self.show_help else \
            "ESPAÇO pausa  |  T rastro  |  G brilho  |  H ajuda  |  ESC sair"
        self.screen.blit(self.font_small.render(hint, True, (170, 175, 200)), (12, self.H - 20))

    def _sparkline(self, rect):
        pygame.draw.rect(self.screen, (255, 255, 255, 20), rect, 1, border_radius=4)
        if len(self.history) < 2:
            return
        lo, hi = min(self.history), max(self.history)
        span = (hi - lo) or 1.0
        n = len(self.history)
        pts = [(rect.x + 2 + (rect.w - 4) * k / (n - 1), rect.bottom - 3 - (rect.h - 6) * (v - lo) / span)
               for k, v in enumerate(self.history)]
        pygame.draw.lines(self.screen, (120, 220, 170), False, pts, 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(pts[-1][0]), int(pts[-1][1])), 3)

    # --------------------------------------------------------------- compose
    def _draw(self, engine, elapsed, info):
        self.fx.fill((0, 0, 0, 0))
        self._draw_background(self.world)
        self._draw_platforms(self.world, engine)
        self._draw_coins(self.world, engine)
        coins = [c.coins for c in engine.characters]
        leader = coins.index(max(coins)) if coins and max(coins) > 0 and coins.count(max(coins)) == 1 else None
        for i, c in enumerate(engine.characters):
            self._draw_character(self.world, self.fx, i, c, leader == i)
        self._draw_fx(self.world, self.fx)

        ox = oy = 0
        if self.shake > 0.5:
            ox = int(random.uniform(-self.shake, self.shake))
            oy = int(random.uniform(-self.shake, self.shake))
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.world, (ox, oy))
        self._draw_hud(engine, elapsed, info)

    def render(self, engine, elapsed, info=None):
        self._handle_events()
        while self.paused:  # pausa bloqueante: o loop de treino também para
            self._draw(engine, elapsed, info)
            over = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            over.fill((0, 0, 0, 90))
            self.screen.blit(over, (0, 0))
            img = self.font_big.render("PAUSADO", True, (255, 255, 255))
            self.screen.blit(img, img.get_rect(center=(self.W // 2, self.H // 2)))
            pygame.display.flip()
            self.clock.tick(30)
            self._handle_events()

        self._observe(engine, elapsed)
        self._update(self.dt, engine, info)
        self._draw(engine, elapsed, info)
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self):
        pygame.quit()

import math
import numpy as np
from dataclasses import dataclass
from .config import ArenaConfig

@dataclass
class Character:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    alive: bool = True
    respawn_timer: float = 0.0
    coins: int = 0
    deaths: int = 0
    eliminations: int = 0
    on_ground: bool = False

@dataclass
class Coin:
    x: float
    y: float
    timer: float

class ArenaEngine:
    """Small fixed-map platformer physics engine. Coordinates are pixels."""
    def __init__(self, config=None):
        self.cfg = config or ArenaConfig()
        self.rng = np.random.default_rng()
        self.platforms = [
            (0, 360, 220, 20), (280, 360, 360, 20),
            (70, 285, 160, 14), (350, 285, 150, 14),
            (210, 215, 180, 14), (35, 145, 130, 14),
            (445, 155, 150, 14),
        ]
        self.characters = []
        self.coins = []
        self.sim_time = 0.0

    def reset(self, seed=None):
        # Sem seed, continua o RNG existente em vez de criar outro com entropia:
        # o autoreset do VectorEnv chama reset() sem seed, e re-semear ali
        # quebraria a reprodutibilidade a partir do 2º episódio de cada worker.
        # (No __init__ o RNG já nasce aleatório, então um reset sem seed num
        # engine recém-criado continua aleatório como antes.)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim_time = 0.0
        self.characters = [
            Character(90, 320),
            *[Character(500 + i * 25, 320) for i in range(self.cfg.opponent_count)]
        ]
        self.coins = []
        for _ in range(self.cfg.coin_count):
            self._spawn_coin()
        return self

    def _spawn_coin(self):
        surfaces = [(x, y, w) for x, y, w, h in self.platforms if w > 50]
        x, y, width = surfaces[int(self.rng.integers(len(surfaces)))]
        coin_x = self.rng.uniform(x + 10, x + width - 10)
        self.coins.append(Coin(float(coin_x), float(y - 18), self.cfg.coin_lifetime))

    def _rect(self, c):
        s = self.cfg.player_size
        return (c.x - s / 2, c.y - s, s, s)

    @staticmethod
    def _overlap(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

    def touching(self, a, b):
        ra, rb = self._rect(a), self._rect(b)
        # Contact includes close horizontal adjacency while vertical bodies overlap.
        vertical = ra[1] < rb[1] + rb[3] and ra[1] + ra[3] > rb[1]
        horizontal_gap = max(rb[0] - (ra[0] + ra[2]), ra[0] - (rb[0] + rb[2]), 0)
        return vertical and horizontal_gap <= 3

    def push(self, actor_index, direction, distance):
        actor = self.characters[actor_index]
        if not actor.alive:
            return False
        candidates = [(i, c) for i, c in enumerate(self.characters)
                      if i != actor_index and c.alive and self.touching(actor, c)]
        if not candidates:
            return False
        i, target = min(candidates, key=lambda pair: abs(pair[1].x - actor.x))
        # Apply an impulse; no teleportation. Distance controls impulse strength.
        target.vx = float(np.clip(target.vx + direction * self.cfg.push_impulse * distance,
                                  -self.cfg.max_speed * 2, self.cfg.max_speed * 2))
        target.vy = min(target.vy, -45.0)
        return True

    def _move_character(self, c, action, dt, is_player=False):
        if not c.alive:
            c.respawn_timer -= dt
            if c.respawn_timer <= 0:
                # Respawn on a randomly selected platform, not outside the map.
                valid = [(x, y, w, h) for x, y, w, h in self.platforms if w > self.cfg.player_size * 2]
                px, py, pw, _ = valid[int(self.rng.integers(len(valid)))]
                c.x = float(self.rng.uniform(px + self.cfg.player_size, px + pw - self.cfg.player_size))
                c.y = float(py)
                c.vx = 0.0
                c.vy = 0.0
                c.alive = True
                c.on_ground = True
            return
        # Actions 1,2 move; 3,4,5 jump; push actions are resolved by environment.
        if action == 1:
            c.vx -= self.cfg.move_accel * dt
        elif action == 2:
            c.vx += self.cfg.move_accel * dt
        elif action in (3, 4, 5) and c.on_ground:
            c.vy = -self.cfg.jump_speed
            if action == 4:
                c.vx -= self.cfg.max_speed * 0.75
            elif action == 5:
                c.vx += self.cfg.max_speed * 0.75
            c.on_ground = False
        c.vx *= 0.88 ** (dt * self.cfg.fps)
        c.vx = float(np.clip(c.vx, -self.cfg.max_speed * 2, self.cfg.max_speed * 2))
        c.vy += self.cfg.gravity * dt
        old_bottom = c.y
        c.x += c.vx * dt
        c.y += c.vy * dt
        half = self.cfg.player_size / 2
        c.x = float(np.clip(c.x, half, self.cfg.width - half))
        c.on_ground = False
        # Land on platform tops while descending.
        for px, py, pw, ph in self.platforms:
            if c.vy >= 0 and old_bottom <= py and c.y >= py and c.x + half > px and c.x - half < px + pw:
                c.y = float(py)
                c.vy = 0.0
                c.on_ground = True
        # Fatal region below the lowest floor.
        if c.y > self.cfg.height + 30:
            c.alive = False
            c.respawn_timer = self.cfg.respawn_seconds
            c.deaths += 1
            # Death costs 10 in-game coins, never allowing a negative balance.
            c.coins = max(0, c.coins - 10)
            c.vx = c.vy = 0.0

    def step(self, actions, dt):
        self.sim_time += dt
        before_deaths = [c.deaths for c in self.characters]
        before_elims = [c.eliminations for c in self.characters]
        # Resolve pushes before movement, with player action codes 6..11.
        for i, action in enumerate(actions):
            if action >= 6 and action <= 11:
                if action <= 8:
                    self.push(i, -1, action - 5)
                else:
                    self.push(i, 1, action - 8)
        for i, (c, action) in enumerate(zip(self.characters, actions)):
            self._move_character(c, action if action < 6 else 0, dt, i == 0)
        # A character can be eliminated if a push impulse eventually sends it into the fatal region.
        # Coin pickup.
        collected = [0] * len(self.characters)
        for coin in self.coins:
            coin.timer -= dt
        self.coins = [coin for coin in self.coins if coin.timer > 0]
        for i, c in enumerate(self.characters):
            if not c.alive:
                continue
            for coin in list(self.coins):
                if math.hypot(c.x - coin.x, (c.y - self.cfg.player_size / 2) - coin.y) < 18:
                    self.coins.remove(coin)
                    c.coins += 1
                    collected[i] += 1
                    self._spawn_coin()
        # Attribute eliminations approximately to a recent contact/push: if an opponent dies
        # while player is adjacent, the player gets credit. This is intentionally simple.
        for i in range(1, len(self.characters)):
            if self.characters[i].deaths > before_deaths[i] and self.characters[0].alive:
                self.characters[0].eliminations += 1
        return {
            "collected": collected,
            "deaths_delta": [c.deaths - before_deaths[i] for i, c in enumerate(self.characters)],
            "eliminations_delta": [c.eliminations - before_elims[i] for i, c in enumerate(self.characters)],
        }

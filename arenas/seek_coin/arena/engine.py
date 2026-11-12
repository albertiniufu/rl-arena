import math
import numpy as np
from dataclasses import dataclass
from .actions import JUMP, LEFT, PUSH_INTENSITY, PUSH_LEFT, PUSH_RIGHT, RIGHT, STOP
from .config import ArenaConfig
from .levels import get_level

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
    on_ground: bool = False
    platform_index: int | None = None

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
        self.level = int(self.cfg.level)
        self.static_platforms, self.moving_specs = get_level(self.level)
        self._set_platforms_at(0.0)
        self.characters = []
        self.coins = []
        self.sim_time = 0.0

    def _platforms_at(self, t: float):
        return (list(self.static_platforms)
                + [spec.at(t) for spec in self.moving_specs])

    def _set_platforms_at(self, t: float) -> None:
        self.platforms = self._platforms_at(t)
        self.platform_deltas = [(0.0, 0.0)] * len(self.platforms)

    def _advance_platforms(self, t: float) -> None:
        old = self.platforms
        new = self._platforms_at(t)
        self.platform_deltas = [
            (nx - ox, ny - oy) for (ox, oy, _, _), (nx, ny, _, _)
            in zip(old, new)
        ]
        self.platforms = new

    def reset(self, seed=None, n_characters=None, spawn_positions=None):
        # Sem seed, continua o RNG existente em vez de criar outro com entropia:
        # o autoreset do VectorEnv chama reset() sem seed, e re-semear ali
        # quebraria a reprodutibilidade a partir do 2º episódio de cada worker.
        # (No __init__ o RNG já nasce aleatório, então um reset sem seed num
        # engine recém-criado continua aleatório como antes.)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim_time = 0.0
        self._set_platforms_at(0.0)
        count = (int(n_characters) if n_characters is not None
                 else 1 + self.cfg.opponent_count)
        if spawn_positions is not None:
            if len(spawn_positions) != count:
                raise ValueError(
                    f"spawn_positions deve ter {count} entradas "
                    f"(recebido {len(spawn_positions)})")
            positions = [(float(x), float(y)) for x, y in spawn_positions]
        else:
            # Posições históricas: jogador à esquerda, bots em fila à direita.
            positions = [(90.0, 320.0)] + [
                (500.0 + i * 25.0, 320.0) for i in range(count - 1)
            ]
        self.characters = [Character(x, y) for x, y in positions]
        self.coins = []
        for _ in range(self.cfg.coin_count):
            self._spawn_coin()
        return self

    def _spawn_coin(self):
        # Moedas só em plataformas estáticas: uma moeda presa a uma
        # plataforma móvel ficaria em posição inválida durante o movimento.
        surfaces = [(x, y, w) for x, y, w, h in self.static_platforms
                    if w > 50]
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

    def _eligible_targets(self, actor_index):
        """Índices que o agente pode empurrar; `None` = qualquer personagem.

        Em times (`cfg.team_ids` preenchido) só inimigos são elegíveis. No
        caminho legado devolve `None` para preservar a chamada de 3 argumentos
        (inclusive nos testes que fazem monkeypatch de `push`).
        """
        team_ids = getattr(self.cfg, "team_ids", ())
        if not team_ids or len(team_ids) != len(self.characters):
            return None
        return [j for j in range(len(self.characters))
                if j != actor_index and team_ids[j] != team_ids[actor_index]]

    def push(self, actor_index, direction, intensity, eligible=None):
        actor = self.characters[actor_index]
        if not actor.alive:
            return False
        allowed = None if eligible is None else set(eligible)
        candidates = [(i, c) for i, c in enumerate(self.characters)
                      if i != actor_index and c.alive and self.touching(actor, c)
                      and (allowed is None or i in allowed)]
        if not candidates:
            return False
        i, target = min(candidates, key=lambda pair: abs(pair[1].x - actor.x))
        # Apply an impulse; no teleportation. Intensity scales the impulse.
        target.vx = float(np.clip(target.vx + direction * self.cfg.push_impulse * intensity,
                                  -self.cfg.max_speed * 2, self.cfg.max_speed * 2))
        target.vy = min(target.vy, -45.0)
        # O empurrão transfere moedas do alvo para quem empurrou. Quem tem
        # menos que `push_coin_transfer` passa o que tem (nunca fica negativo).
        transfer = min(int(self.cfg.push_coin_transfer), int(target.coins))
        if transfer > 0:
            target.coins -= transfer
            actor.coins += transfer
        return True

    def _move_character(self, c, action, dt, is_player=False):
        if not c.alive:
            c.respawn_timer -= dt
            if c.respawn_timer <= 0:
                # Respawn on a randomly selected platform, not outside the map.
                valid = [(i, x, y, w, h)
                         for i, (x, y, w, h) in enumerate(self.platforms)
                         if w > self.cfg.player_size * 2]
                index, px, py, pw, _ = valid[
                    int(self.rng.integers(len(valid)))]
                c.x = float(self.rng.uniform(px + self.cfg.player_size, px + pw - self.cfg.player_size))
                c.y = float(py)
                c.vx = 0.0
                c.vy = 0.0
                c.alive = True
                c.on_ground = True
                c.platform_index = index
            return
        # Carrega o personagem junto com a plataforma em que ele estava.
        if c.on_ground and c.platform_index is not None:
            dx, dy = self.platform_deltas[c.platform_index]
            c.x += dx
            c.y += dy
        # Ações: LEFT/RIGHT andam, JUMP pula; empurrões são resolvidos no step.
        if action == LEFT:
            c.vx -= self.cfg.move_accel * dt
        elif action == RIGHT:
            c.vx += self.cfg.move_accel * dt
        elif action == JUMP and c.on_ground:
            c.vy = -self.cfg.jump_speed
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
        c.platform_index = None
        # Land on platform tops while descending.
        for index, (px, py, pw, ph) in enumerate(self.platforms):
            if c.vy >= 0 and old_bottom <= py and c.y >= py and c.x + half > px and c.x - half < px + pw:
                c.y = float(py)
                c.vy = 0.0
                c.on_ground = True
                c.platform_index = index
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
        self._advance_platforms(self.sim_time)
        before_deaths = [c.deaths for c in self.characters]
        # Saldo antes dos empurrões: o delta isola a transferência de moedas
        # (a morte também mexe no saldo, mas acontece depois).
        before_push_coins = [c.coins for c in self.characters]
        # Empurrões são resolvidos antes do movimento, sempre na intensidade
        # máxima: um código por direção.
        for i, action in enumerate(actions):
            if action not in (PUSH_LEFT, PUSH_RIGHT):
                continue
            direction = -1 if action == PUSH_LEFT else 1
            eligible = self._eligible_targets(i)
            if eligible is None:
                self.push(i, direction, PUSH_INTENSITY)
            else:
                self.push(i, direction, PUSH_INTENSITY, eligible)
        # Positivo para quem recebeu moedas, negativo para quem as perdeu.
        coins_transferred = [c.coins - before
                             for c, before in zip(self.characters, before_push_coins)]
        for i, (c, action) in enumerate(zip(self.characters, actions)):
            push = action in (PUSH_LEFT, PUSH_RIGHT)
            self._move_character(c, STOP if push else action, dt, i == 0)
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
        return {
            "collected": collected,
            "deaths_delta": [c.deaths - before_deaths[i] for i, c in enumerate(self.characters)],
            "coins_transferred": coins_transferred,
        }

import math
import numpy as np
from dataclasses import dataclass
from .actions import JUMP, LEFT, PUSH_INTENSITY, PUSH_LEFT, PUSH_RIGHT, RIGHT, STOP
from .config import ArenaConfig, COLLECTOR_ROLE, PUSHER_ROLE
from .levels import (
    get_buttons, get_fixed_coins, get_hazard_kinds, get_level,
    get_moving_coins, get_super_coin_position, uses_endless_world)
from .endless import (
    DROP_MARGIN, GROUND_Y, MARGIN, ChunkGenerator, Mover)

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
    slide_timer: float = 0.0
    slide_vx: float = 0.0
    ice_cooldown: float = 0.0
    sand_timer: float = 0.0
    push_cooldown: float = 0.0  # segundos restantes até poder empurrar de novo

@dataclass
class Coin:
    x: float
    y: float
    timer: float
    # Plataforma à qual a moeda está presa (None = moeda solta/estática).
    platform_index: int | None = None


@dataclass
class Hazard:
    """Área temporária de areia ou gelo sobre uma plataforma."""
    kind: str
    x: float
    y: float
    w: float
    h: float
    platform_index: int
    timer: float


@dataclass
class SuperCoin:
    """Moeda especial do nível 3, separada das moedas normais."""
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
        self.hazard_kinds = get_hazard_kinds(self.level)
        self.characters = []
        self.coins = []
        self.hazards = []
        self.buttons = list(get_buttons(self.level))
        self.super_coin_position = get_super_coin_position(self.level)
        self.super_coin: SuperCoin | None = None
        self.super_coin_cooldown = 0.0
        self._button_state: dict[str, bool] = {}
        self.sim_time = 0.0
        # Estado do mundo procedural (nível 5); `None` nos demais níveis.
        self.world_offset = 0.0
        self.next_chunk = 0
        self.distance = 0.0
        self.holes: list[tuple] = []
        self.movers: list[Mover] = []
        self.chunk_generator = None

    def _platforms_at(self, t: float):
        return (list(self.static_platforms)
                + [spec.at(t) for spec in self.moving_specs])

    def _set_platforms_at(self, t: float) -> None:
        self.platforms = self._platforms_at(t)
        self.platform_deltas = [(0.0, 0.0)] * len(self.platforms)

    def _advance_platforms(self, t: float) -> None:
        if uses_endless_world(self.level):
            self._advance_endless_movers(t)
            return
        old = self.platforms
        new = self._platforms_at(t)
        self.platform_deltas = [
            (nx - ox, ny - oy) for (ox, oy, _, _), (nx, ny, _, _)
            in zip(old, new)
        ]
        self.platforms = new

    # ---------------------------------------------------------------- nível 5
    def _init_endless_world(self, seed) -> None:
        """Zera o mundo procedural e gera os primeiros chunks."""
        self.world_offset = 0.0
        self.next_chunk = 0
        self.distance = 0.0
        self.platforms = []
        self.platform_deltas = []
        self.movers = []
        self.holes = []
        self.coins = []
        self.hazards = []
        self.buttons = []
        self.super_coin = None
        self.super_coin_cooldown = 0.0
        self._button_state = {}
        world_seed = (int(seed) if seed is not None
                      else int(self.rng.integers(0, 2**32 - 1)))
        self.chunk_generator = ChunkGenerator(
            world_seed, world_height=float(self.cfg.height))
        self._generate_ahead()

    def _generate_ahead(self) -> None:
        """Gera chunks até cobrir a tela + margem à frente do agente."""
        if self.chunk_generator is None:
            return
        limit = self.world_offset + self.cfg.width + MARGIN
        while (self.next_chunk * self.chunk_generator.chunk_width) < limit:
            self._append_chunk(self.next_chunk)
            self.next_chunk += 1

    def _append_chunk(self, index: int) -> None:
        content = self.chunk_generator.generate(index)
        offset = self.world_offset
        platform_map = {}
        for local, (x, y, w, h) in enumerate(content.platforms):
            platform_map[local] = len(self.platforms)
            self.platforms.append((x - offset, y, w, h))
            self.platform_deltas.append((0.0, 0.0))
        for spec in content.movers:
            index_p = platform_map.get(spec.platform)
            if index_p is None:
                continue
            mover = Mover(
                platform_index=index_p,
                base_x=spec.base_x - offset,
                base_y=spec.base_y,
                axis=spec.axis,
                amplitude=spec.amplitude,
                period=spec.period,
                phase=spec.phase,
            )
            # Nasce já na fase atual: gerar um chunk no meio do episódio não
            # pode fazer a plataforma "pular" da base até a posição de t=0.
            mx, my = mover.at(self.sim_time)
            _, _, w, h = self.platforms[index_p]
            self.platforms[index_p] = (mx, my, w, h)
            self.movers.append(mover)
        for x, y, w, h in content.holes:
            self.holes.append((x - offset, y, w, h))
        for x, y in content.coins:
            self.coins.append(Coin(x - offset, y, self.cfg.coin_lifetime))

    def _advance_endless_movers(self, t: float) -> None:
        deltas = [(0.0, 0.0)] * len(self.platforms)
        for mover in self.movers:
            i = mover.platform_index
            if not 0 <= i < len(self.platforms):
                continue
            x, y, w, h = self.platforms[i]
            nx, ny = mover.at(t)
            deltas[i] = (nx - x, ny - y)
            self.platforms[i] = (nx, ny, w, h)
        self.platform_deltas = deltas

    def _scroll_endless(self) -> None:
        """Rola o mundo quando o agente passa de ~55% da largura."""
        if self.chunk_generator is None or not self.characters:
            return
        trigger = self.cfg.width * 0.55
        anchor = self.characters[0].x
        if anchor > trigger:
            dx = anchor - trigger
            self._shift_world(-dx)
            self.world_offset += dx
            self._drop_behind()
        self.distance = max(
            self.distance, self.world_offset + self.characters[0].x)
        self._generate_ahead()
        for kind in self.hazard_kinds:
            have = sum(1 for hazard in self.hazards
                       if hazard.kind == kind)
            for _ in range(max(0, int(self.cfg.hazard_count) - have)):
                self._spawn_hazard(kind)

    def _shift_world(self, dx: float) -> None:
        self.platforms = [(x + dx, y, w, h)
                          for x, y, w, h in self.platforms]
        self.platform_deltas = [(0.0, 0.0)] * len(self.platforms)
        self.holes = [(x + dx, y, w, h) for x, y, w, h in self.holes]
        for coin in self.coins:
            coin.x += dx
        for hazard in self.hazards:
            hazard.x += dx
        for character in self.characters:
            character.x += dx
        for mover in self.movers:
            mover.base_x += dx

    def _drop_behind(self) -> None:
        """Descarta entidades atrás da tela e remapeia índices de plataforma."""
        limit = -DROP_MARGIN
        keep = [i for i, (x, _, w, _) in enumerate(self.platforms)
                if x + w > limit]
        if len(keep) != len(self.platforms):
            mapping = {old: new for new, old in enumerate(keep)}
            self.platforms = [self.platforms[i] for i in keep]
            self.platform_deltas = [(0.0, 0.0)] * len(self.platforms)
            for character in self.characters:
                if character.platform_index is not None:
                    character.platform_index = mapping.get(
                        character.platform_index)
                    if character.platform_index is None:
                        character.on_ground = False
            kept_hazards = []
            for hazard in self.hazards:
                new_index = mapping.get(hazard.platform_index)
                if new_index is not None:
                    hazard.platform_index = new_index
                    kept_hazards.append(hazard)
            self.hazards = kept_hazards
            kept_movers = []
            for mover in self.movers:
                new_index = mapping.get(mover.platform_index)
                if new_index is not None:
                    mover.platform_index = new_index
                    kept_movers.append(mover)
            self.movers = kept_movers
        self.holes = [h for h in self.holes if h[0] + h[2] > limit]
        self.coins = [c for c in self.coins if c.x > limit]

    def _nearby_respawn(self):
        """Plataforma mais próxima do agente para renascer (nível 5)."""
        anchor = self.characters[0].x if self.characters else 0.0
        valid = [(i, x, y, w)
                 for i, (x, y, w, h) in enumerate(self.platforms)
                 if w > self.cfg.player_size * 2
                 and x + w > 0 and x < self.cfg.width]
        if not valid:
            return None
        ground = [p for p in valid if p[2] >= GROUND_Y - 1.0]
        pool = ground or valid
        index, px, py, pw = min(
            pool, key=lambda p: abs(p[1] + p[3] / 2.0 - anchor))
        lo, hi = px + self.cfg.player_size, px + pw - self.cfg.player_size
        rx = (px + pw / 2.0 if hi <= lo
              else float(self.rng.uniform(lo, hi)))
        return index, rx, float(py)

    def _reset_endless(self, seed, n_characters, spawn_positions):
        count = (int(n_characters) if n_characters is not None
                 else 1 + self.cfg.opponent_count)
        if spawn_positions is not None:
            if len(spawn_positions) != count:
                raise ValueError(
                    f"spawn_positions deve ter {count} entradas "
                    f"(recebido {len(spawn_positions)})")
            positions = [(float(x), float(y)) for x, y in spawn_positions]
        else:
            positions = [(90.0, 320.0)] + [
                (500.0 + i * 25.0, 320.0) for i in range(count - 1)
            ]
        self.characters = [Character(x, y) for x, y in positions]
        self._init_endless_world(seed)
        for kind in self.hazard_kinds:
            for _ in range(max(0, int(self.cfg.hazard_count))):
                self._spawn_hazard(kind)
        return self

    def reset(self, seed=None, n_characters=None, spawn_positions=None):
        # Sem seed, continua o RNG existente em vez de criar outro com entropia:
        # o autoreset do VectorEnv chama reset() sem seed, e re-semear ali
        # quebraria a reprodutibilidade a partir do 2º episódio de cada worker.
        # (No __init__ o RNG já nasce aleatório, então um reset sem seed num
        # engine recém-criado continua aleatório como antes.)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim_time = 0.0
        if uses_endless_world(self.level):
            return self._reset_endless(seed, n_characters, spawn_positions)
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
        fixed = get_fixed_coins(self.level)
        for x, y in fixed:
            self.coins.append(Coin(float(x), float(y),
                                   self.cfg.coin_lifetime))
        static_count = len(self.static_platforms)
        for moving_index, dx, dy in get_moving_coins(self.level):
            platform_index = static_count + moving_index
            px, py, pw, _ = self.platforms[platform_index]
            self.coins.append(Coin(
                px + pw / 2.0 + dx, py + dy, self.cfg.coin_lifetime,
                platform_index=platform_index))
        for _ in range(max(0, self.cfg.coin_count - len(fixed))):
            self._spawn_coin()
        self.hazards = []
        for kind in self.hazard_kinds:
            for _ in range(max(0, int(self.cfg.hazard_count))):
                self._spawn_hazard(kind)
        self.super_coin = None
        self.super_coin_cooldown = 0.0
        self._button_state = {button.kind: False for button in self.buttons}
        return self

    def _spawn_coin(self):
        # Moedas aleatórias nas plataformas ativas; no nível 5 o mundo é
        # procedural, então as plataformas fixas do nível são vazias.
        surfaces = [(x, y, w) for x, y, w, h in
                    (self.platforms if uses_endless_world(self.level)
                     else self.static_platforms)
                    if w > 50]
        if not surfaces:
            return
        x, y, width = surfaces[int(self.rng.integers(len(surfaces)))]
        coin_x = self.rng.uniform(x + 10, x + width - 10)
        self.coins.append(Coin(float(coin_x), float(y - 18),
                               self.cfg.coin_lifetime))

    def _spawn_hazard(self, kind: str):
        """Cria um hazard em uma plataforma aleatória; devolve ou None."""
        min_w = self.cfg.hazard_width_min
        candidates = [(i, x, y, w)
                      for i, (x, y, w, h) in enumerate(self.platforms)
                      if w >= min_w + 4]
        if not candidates:
            return None
        index, x, y, width = candidates[
            int(self.rng.integers(len(candidates)))]
        max_w = min(self.cfg.hazard_width_max, width - 4)
        hazard_w = float(self.rng.uniform(min_w, max(min_w, max_w)))
        hazard_x = float(self.rng.uniform(
            x + 2, max(x + 2, x + width - 2 - hazard_w)))
        hazard = Hazard(
            kind=kind,
            x=hazard_x,
            y=float(y - self.cfg.hazard_height),
            w=hazard_w,
            h=float(self.cfg.hazard_height),
            platform_index=index,
            timer=float(self.rng.uniform(
                self.cfg.hazard_min_seconds, self.cfg.hazard_max_seconds)),
        )
        self.hazards.append(hazard)
        return hazard

    def _advance_hazards(self, dt: float) -> None:
        if not self.hazards:
            return
        for hazard in self.hazards:
            if 0 <= hazard.platform_index < len(self.platforms):
                dx, dy = self.platform_deltas[hazard.platform_index]
                hazard.x += dx
                hazard.y += dy
            hazard.timer -= dt
        for i in range(len(self.hazards) - 1, -1, -1):
            if self.hazards[i].timer <= 0.0:
                kind = self.hazards[i].kind
                self.hazards.pop(i)
                self._spawn_hazard(kind)

    def _hazard_under(self, c: Character, kind: str):
        if not c.alive or not c.on_ground or c.platform_index is None:
            return None
        half = self.cfg.player_size / 2
        for hazard in self.hazards:
            if (hazard.kind != kind
                    or hazard.platform_index != c.platform_index):
                continue
            if c.x + half > hazard.x and c.x - half < hazard.x + hazard.w:
                return hazard
        return None

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

    def _role(self, index):
        """Papel do personagem (`collector`/`pusher`) ou `None` no legado."""
        roles = getattr(self.cfg, "agent_roles", ())
        if roles and len(roles) == len(self.characters):
            return roles[index]
        return None

    def _vx_limit(self):
        """Teto de velocidade horizontal; o impulso forte pode elevá-lo."""
        cap = float(getattr(self.cfg, "push_speed_cap", 0.0) or 0.0)
        return max(float(self.cfg.max_speed) * 2.0, cap)

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
        limit = self._vx_limit()
        target.vx = float(np.clip(
            target.vx + direction * self.cfg.push_impulse * intensity,
            -limit, limit))
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
                # Respawn on a platform. No nível 5, perto do ponto onde o
                # agente está (o mundo infinito não tem "mapa" para sortear).
                spot = (self._nearby_respawn()
                        if uses_endless_world(self.level) else None)
                if spot is not None:
                    index, respawn_x, respawn_y = spot
                else:
                    valid = [(i, x, y, w, h)
                             for i, (x, y, w, h) in enumerate(self.platforms)
                             if w > self.cfg.player_size * 2]
                    index, px, py, pw, _ = valid[
                        int(self.rng.integers(len(valid)))]
                    respawn_x = float(self.rng.uniform(
                        px + self.cfg.player_size,
                        px + pw - self.cfg.player_size))
                    respawn_y = float(py)
                c.x = float(respawn_x)
                c.y = float(respawn_y)
                c.vx = 0.0
                c.vy = 0.0
                c.slide_timer = 0.0
                c.slide_vx = 0.0
                c.ice_cooldown = 0.0
                c.sand_timer = 0.0
                c.push_cooldown = 0.0
                c.alive = True
                c.on_ground = True
                c.platform_index = index
            return
        # Carrega o personagem junto com a plataforma em que ele estava.
        if c.on_ground and c.platform_index is not None:
            dx, dy = self.platform_deltas[c.platform_index]
            c.x += dx
            c.y += dy
        c.ice_cooldown = max(0.0, c.ice_cooldown - dt)
        sliding = c.slide_timer > 0.0
        if sliding:
            c.slide_timer = max(0.0, c.slide_timer - dt)
            c.vx = c.slide_vx
        else:
            ice = self._hazard_under(c, "ice")
            if ice is not None and c.ice_cooldown <= 0.0:
                direction = -1.0 if self.rng.random() < 0.5 else 1.0
                distance = self.cfg.width * self.cfg.ice_slide_fraction
                c.slide_vx = direction * distance / max(
                    self.cfg.ice_slide_seconds, 1e-6)
                c.slide_timer = self.cfg.ice_slide_seconds
                c.ice_cooldown = (self.cfg.ice_slide_seconds
                                  + self.cfg.ice_cooldown)
                c.vx = c.slide_vx
                sliding = True
        # Areia: ao tocar, o efeito dura cfg.sand_duration mesmo depois
        # de sair do patch; tocá-la de novo renova o timer.
        c.sand_timer = max(0.0, c.sand_timer - dt)
        if self._hazard_under(c, "sand") is not None:
            c.sand_timer = self.cfg.sand_duration
        sand_factor = (self.cfg.sand_speed_factor
                       if c.sand_timer > 0.0 else 1.0)
        # Ações: LEFT/RIGHT andam, JUMP pula; empurrões são resolvidos no step.
        if not sliding:
            if action == LEFT:
                c.vx -= self.cfg.move_accel * sand_factor * dt
            elif action == RIGHT:
                c.vx += self.cfg.move_accel * sand_factor * dt
            elif action == JUMP and c.on_ground:
                c.vy = -self.cfg.jump_speed
                c.on_ground = False
            c.vx *= 0.88 ** (dt * self.cfg.fps)
        limit = self._vx_limit()
        c.vx = float(np.clip(c.vx, -limit, limit))
        if sand_factor < 1.0:
            c.vx = float(np.clip(c.vx,
                                 -self.cfg.max_speed * sand_factor,
                                 self.cfg.max_speed * sand_factor))
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
            c.slide_timer = 0.0
            c.slide_vx = 0.0
            c.ice_cooldown = 0.0
            c.sand_timer = 0.0
            c.push_cooldown = 0.0

    def _button_touching(self, character, button):
        half = self.cfg.player_size / 2.0
        return (
            character.alive
            and character.x + half > button.x
            and character.x - half < button.x + button.w
            and character.y + half > button.y
            and character.y - half < button.y + button.h
        )

    def _update_super_coin(self, dt):
        """Atualiza a super moeda e os botões de contato do nível 3.

        Devolve `(collected, activated, cancelled, expired)`, em que
        `collected` tem uma entrada por personagem. A coleta tem prioridade
        sobre a expiração no mesmo substep.
        """
        collected = [0] * len(self.characters)
        activated = cancelled = expired = False
        self.super_coin_cooldown = max(0.0, self.super_coin_cooldown - dt)

        if self.super_coin is not None:
            for i, c in enumerate(self.characters):
                if not c.alive or self._role(i) == PUSHER_ROLE:
                    continue
                if math.hypot(
                        c.x - self.super_coin.x,
                        (c.y - self.cfg.player_size / 2.0) - self.super_coin.y
                ) < 18:
                    c.coins += int(self.cfg.super_coin_coins)
                    collected[i] = 1
                    self.super_coin = None
                    self.super_coin_cooldown = float(
                        self.cfg.super_coin_cooldown)
                    break
            if self.super_coin is not None:
                self.super_coin.timer -= dt
                if self.super_coin.timer <= 0:
                    self.super_coin = None
                    self.super_coin_cooldown = float(
                        self.cfg.super_coin_cooldown)
                    expired = True

        current = {
            button.kind: any(
                self._button_touching(c, button) for c in self.characters)
            for button in self.buttons
        }
        for button in self.buttons:
            pressed = current.get(button.kind, False)
            was = self._button_state.get(button.kind, False)
            self._button_state[button.kind] = pressed
            if not (pressed and not was):
                continue
            if button.kind == "activate":
                if (self.super_coin is None
                        and self.super_coin_cooldown <= 0
                        and self.super_coin_position is not None):
                    x, y = self.super_coin_position
                    self.super_coin = SuperCoin(
                        float(x), float(y), float(self.cfg.super_coin_seconds))
                    activated = True
            elif button.kind == "cancel" and self.super_coin is not None:
                self.super_coin = None
                self.super_coin_cooldown = float(self.cfg.super_coin_cooldown)
                cancelled = True
        return collected, activated, cancelled, expired

    def step(self, actions, dt):
        self.sim_time += dt
        self._advance_platforms(self.sim_time)
        # Moedas presas às plataformas móveis acompanham o deslocamento.
        for coin in self.coins:
            if coin.platform_index is not None:
                dx, dy = self.platform_deltas[coin.platform_index]
                coin.x += dx
                coin.y += dy
        self._advance_hazards(dt)
        # Recarga de empurrão: um empurrão bem-sucedido bloqueia o próximo
        # pelo tempo configurado (2 s no cenário coletores × empurradores).
        for c in self.characters:
            if c.push_cooldown > 0:
                c.push_cooldown = max(0.0, c.push_cooldown - dt)
        before_deaths = [c.deaths for c in self.characters]
        # Saldo antes dos empurrões: o delta isola a transferência de moedas
        # (a morte também mexe no saldo, mas acontece depois).
        before_push_coins = [c.coins for c in self.characters]
        # Empurrões são resolvidos antes do movimento, sempre na intensidade
        # máxima: um código por direção.
        for i, action in enumerate(actions):
            if action not in (PUSH_LEFT, PUSH_RIGHT):
                continue
            # Coletor não empurra: a ação é inerte (o ambiente a converte em
            # STOP antes daqui; o engine repete a regra para chamadas diretas).
            if self._role(i) == COLLECTOR_ROLE:
                continue
            actor = self.characters[i]
            if actor.push_cooldown > 0:
                continue
            direction = -1 if action == PUSH_LEFT else 1
            eligible = self._eligible_targets(i)
            if eligible is None:
                pushed = self.push(i, direction, PUSH_INTENSITY)
            else:
                pushed = self.push(i, direction, PUSH_INTENSITY, eligible)
            if pushed and self.cfg.push_cooldown > 0:
                actor.push_cooldown = float(self.cfg.push_cooldown)
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
            # Empurrador não coleta do mapa: a moeda dele vem da transferência
            # no empurrão (papel "só empurra").
            if not c.alive or self._role(i) == PUSHER_ROLE:
                continue
            for coin in list(self.coins):
                if math.hypot(c.x - coin.x, (c.y - self.cfg.player_size / 2) - coin.y) < 18:
                    self.coins.remove(coin)
                    c.coins += 1
                    collected[i] += 1
                    self._spawn_coin()
        super_collected, super_activated, super_cancelled, super_expired = (
            self._update_super_coin(dt))
        if uses_endless_world(self.level):
            self._scroll_endless()
        events = {
            "collected": collected,
            "deaths_delta": [c.deaths - before_deaths[i] for i, c in enumerate(self.characters)],
            "coins_transferred": coins_transferred,
            "super_coin_collected": super_collected,
            "super_coin_activated": super_activated,
            "super_coin_cancelled": super_cancelled,
            "super_coin_expired": super_expired,
        }
        if uses_endless_world(self.level):
            events["distance"] = float(self.distance)
        return events

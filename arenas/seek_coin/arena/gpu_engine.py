"""PoC de engine vetorizado do seek_coin rodando em GPU (PyTorch).

Este módulo **não** substitui o `ArenaEngine` da arena: ele é um experimento
de throughput. A física é reimplementada em tensores com lote de `n_envs`
ambientes, permitindo milhares de passos em paralelo na GPU. Simplificações
deliberadas em relação ao engine canônico:

- empurrões entre personagens não são simulados (ações de push viram STOP);
- a observação semântica é opcional e cobre personagens/moedas/plataformas
  de forma aproximada;
- o reset/respawn e as moedas seguem o nível 1 (plataformas móveis, moedas
  fixas, móveis e aleatórias), mas sem o RNG bit a bit do NumPy.

O objetivo é medir passos de ambiente por segundo contra o engine de CPU.
"""
from __future__ import annotations

import math

import torch

from .actions import JUMP, LEFT, PUSH_LEFT, PUSH_RIGHT, RIGHT, STOP
from .config import ArenaConfig
from .levels import get_fixed_coins, get_level, get_moving_coins


class GpuSeekCoin:
    """Lote de ambientes seek_coin com estado e física em tensores."""

    def __init__(self, n_envs: int = 256, n_chars: int = 2, level: int = 1,
                 device: str = "cuda", seed: int = 0, coin_count: int = 10):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.n_envs = int(n_envs)
        self.n_chars = int(n_chars)
        self.cfg = ArenaConfig(level=level, opponent_count=self.n_chars - 1)
        self.dt = 1.0 / self.cfg.fps
        self.level = int(level)
        self.static, self.moving_specs = get_level(self.level)
        self.static_t = torch.tensor(
            self.static, dtype=torch.float32, device=self.device)
        self.static_count = len(self.static)
        self.platform_count = self.static_count + len(self.moving_specs)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

        fixed = get_fixed_coins(self.level)
        moving = get_moving_coins(self.level)
        self.n_fixed = len(fixed)
        self.n_moving = len(moving)
        self.n_random = max(0, int(coin_count) - self.n_fixed)
        self.n_coins = self.n_fixed + self.n_moving + self.n_random
        self.fixed_positions = torch.tensor(
            fixed, dtype=torch.float32, device=self.device
        ).reshape(-1, 2)
        self.moving_specs_coins = tuple(moving)
        self.reset()

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def _zeros(self, *shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device=self.device)

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device,
                          generator=self.generator)

    def _randint(self, low, high, shape):
        return torch.randint(low, high, shape, device=self.device,
                             generator=self.generator)

    def _platforms_at(self, sim_time: torch.Tensor) -> torch.Tensor:
        parts = [self.static_t.unsqueeze(0).expand(
            self.n_envs, -1, -1)]
        for spec in self.moving_specs:
            offset = spec.amplitude * (
                0.5 - 0.5 * torch.cos(
                    2 * math.pi * sim_time / spec.period + spec.phase))
            if spec.axis == "x":
                rect = torch.stack([
                    spec.x + offset,
                    torch.full_like(offset, float(spec.y)),
                    torch.full_like(offset, float(spec.w)),
                    torch.full_like(offset, float(spec.h)),
                ], dim=1)
            else:
                rect = torch.stack([
                    torch.full_like(offset, float(spec.x)),
                    spec.y + offset,
                    torch.full_like(offset, float(spec.w)),
                    torch.full_like(offset, float(spec.h)),
                ], dim=1)
            parts.append(rect.unsqueeze(1))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self) -> dict:
        e, c = self.n_envs, self.n_chars
        self.sim_time = self._zeros(e)
        self.x = self._zeros(e, c)
        self.x[:, 0] = 90.0
        if c > 1:
            bots = 500.0 + torch.arange(c - 1, device=self.device) * 25.0
            self.x[:, 1:] = bots.unsqueeze(0)
        self.y = torch.full((e, c), 320.0, device=self.device)
        self.vx = self._zeros(e, c)
        self.vy = self._zeros(e, c)
        self.alive = torch.ones(e, c, dtype=torch.bool, device=self.device)
        self.respawn_timer = self._zeros(e, c)
        self.coins = self._zeros(e, c, dtype=torch.int64)
        self.deaths = self._zeros(e, c, dtype=torch.int64)
        self.on_ground = torch.zeros(
            e, c, dtype=torch.bool, device=self.device)
        self.platform_index = torch.full(
            (e, c), -1, dtype=torch.int64, device=self.device)
        self.platforms = self._platforms_at(self.sim_time)
        self.platform_deltas = self._zeros(e, self.platform_count, 2)

        # Moedas: fixas + móveis + aleatórias em plataformas estáticas.
        k = self.n_coins
        self.coin_x = self._zeros(e, k)
        self.coin_y = self._zeros(e, k)
        self.coin_timer = torch.full(
            (e, k), float(self.cfg.coin_lifetime), device=self.device)
        self.coin_platform = torch.full(
            (e, k), -1, dtype=torch.int64, device=self.device)

        for j, (x, y) in enumerate(self.fixed_positions):
            self.coin_x[:, j] = x
            self.coin_y[:, j] = y
        start = self.n_fixed
        for j, (moving_index, dx, dy) in enumerate(self.moving_specs_coins):
            pidx = self.static_count + moving_index
            px, py, pw, _ = self.platforms[:, pidx].unbind(dim=1)
            self.coin_x[:, start + j] = px + pw / 2.0 + dx
            self.coin_y[:, start + j] = py + dy
            self.coin_platform[:, start + j] = pidx
        self._respawn_random_coins(slice(self.n_fixed + self.n_moving, k))
        return self.readout()

    def readout(self) -> dict:
        return {
            "alive": self.alive,
            "coins": self.coins,
            "deaths": self.deaths,
            "sim_time": self.sim_time,
        }

    def _respawn_random_coins(self, slots) -> None:
        """Reposiciona as moedas `slots` em plataformas estáticas."""
        if isinstance(slots, slice):
            e, k = self.n_envs, self.n_coins
            index = torch.arange(k, device=self.device)[slots]
        else:
            index = slots
        if index.numel() == 0:
            return
        e = self.n_envs
        n = index.numel()
        s = self.static_count
        picks = self._randint(0, s, (e, n))
        px = torch.gather(
            self.static_t[:, 0].unsqueeze(0).expand(e, -1), 1, picks)
        py = torch.gather(
            self.static_t[:, 1].unsqueeze(0).expand(e, -1), 1, picks)
        pw = torch.gather(
            self.static_t[:, 2].unsqueeze(0).expand(e, -1), 1, picks)
        u = self._rand(e, n)
        self.coin_x[:, index] = px + 10.0 + u * (pw - 20.0)
        self.coin_y[:, index] = py - 18.0
        self.coin_timer[:, index] = float(self.cfg.coin_lifetime)
        self.coin_platform[:, index] = -1

    # ------------------------------------------------------------------ #
    # Passo
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> dict:
        e, c = self.n_envs, self.n_chars
        actions = actions.to(self.device, dtype=torch.int64)
        cfg = self.cfg

        self.sim_time += self.dt
        old_platforms = self.platforms
        self.platforms = self._platforms_at(self.sim_time)
        self.platform_deltas = (
            self.platforms[:, :, :2] - old_platforms[:, :, :2])

        # Moedas presas à plataforma móvel acompanham o deslocamento.
        if self.n_moving:
            slots = slice(self.n_fixed, self.n_fixed + self.n_moving)
            idx = self.coin_platform[:, slots].clamp(min=0)
            dx = torch.gather(self.platform_deltas[:, :, 0], 1, idx)
            dy = torch.gather(self.platform_deltas[:, :, 1], 1, idx)
            live = self.coin_platform[:, slots] >= 0
            self.coin_x[:, slots] += torch.where(
                live, dx, torch.zeros_like(dx))
            self.coin_y[:, slots] += torch.where(
                live, dy, torch.zeros_like(dy))

        # Carrega personagens em cima de plataformas móveis.
        idx = self.platform_index.clamp(min=0)
        dx = torch.gather(self.platform_deltas[:, :, 0], 1, idx)
        dy = torch.gather(self.platform_deltas[:, :, 1], 1, idx)
        carried = self.alive & self.on_ground & (self.platform_index >= 0)
        self.x += torch.where(carried, dx, torch.zeros_like(dx))
        self.y += torch.where(carried, dy, torch.zeros_like(dy))

        left = actions == LEFT
        right = actions == RIGHT
        jump = actions == JUMP
        self.vx += cfg.move_accel * self.dt * (
            right.float() - left.float())
        jump_now = jump & self.on_ground
        self.vy = torch.where(
            jump_now, torch.full_like(self.vy, -cfg.jump_speed), self.vy)
        self.on_ground = torch.where(
            jump_now, torch.zeros_like(self.on_ground), self.on_ground)
        self.vx *= 0.88 ** (self.dt * cfg.fps)
        self.vx = self.vx.clamp(-cfg.max_speed * 2, cfg.max_speed * 2)
        self.vy += cfg.gravity * self.dt
        old_bottom = self.y
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        half = cfg.player_size / 2.0
        self.x = self.x.clamp(half, cfg.width - half)

        on_ground = torch.zeros_like(self.on_ground)
        platform_index = torch.full_like(self.platform_index, -1)
        for p in range(self.platform_count):
            px = self.platforms[:, p, 0].unsqueeze(1)
            py = self.platforms[:, p, 1].unsqueeze(1)
            pw = self.platforms[:, p, 2].unsqueeze(1)
            land = ((self.vy >= 0) & (old_bottom <= py) & (self.y >= py)
                    & (self.x + half > px) & (self.x - half < px + pw)
                    & self.alive)
            self.y = torch.where(land, py, self.y)
            self.vy = torch.where(
                land, torch.zeros_like(self.vy), self.vy)
            on_ground = on_ground | land
            platform_index = torch.where(
                land, torch.full_like(platform_index, p), platform_index)
        self.on_ground = on_ground
        self.platform_index = platform_index

        # Queda fatal, respawn.
        dead = self.alive & (self.y > cfg.height + 30)
        self.alive = self.alive & ~dead
        self.respawn_timer = torch.where(
            dead, torch.full_like(self.respawn_timer, cfg.respawn_seconds),
            self.respawn_timer)
        self.deaths += dead.long()
        self.coins = torch.where(
            dead, torch.clamp(self.coins - 10, min=0), self.coins)
        self.vx = torch.where(dead, torch.zeros_like(self.vx), self.vx)
        self.vy = torch.where(dead, torch.zeros_like(self.vy), self.vy)
        self.respawn_timer = torch.where(
            self.alive, torch.zeros_like(self.respawn_timer),
            self.respawn_timer - self.dt)
        respawn = (~self.alive) & (self.respawn_timer <= 0)
        if bool(respawn.any()):
            picks = self._randint(0, self.platform_count, (e, c))
            px = torch.gather(self.platforms[:, :, 0], 1, picks)
            py = torch.gather(self.platforms[:, :, 1], 1, picks)
            pw = torch.gather(self.platforms[:, :, 2], 1, picks)
            u = self._rand(e, c)
            nx = px + cfg.player_size + u * torch.clamp(
                pw - 2 * cfg.player_size, min=1.0)
            self.x = torch.where(respawn, nx, self.x)
            self.y = torch.where(respawn, py, self.y)
            self.vx = torch.where(respawn, torch.zeros_like(self.vx), self.vx)
            self.vy = torch.where(respawn, torch.zeros_like(self.vy), self.vy)
            self.alive = self.alive | respawn
            self.on_ground = self.on_ground | respawn
            self.platform_index = torch.where(
                respawn, picks, self.platform_index)

        # Moedas: timers, coleta e reposicionamento.
        self.coin_timer -= self.dt
        cx = self.x.unsqueeze(2)
        cy = (self.y - half).unsqueeze(2)
        dist = torch.sqrt(
            (cx - self.coin_x.unsqueeze(1)) ** 2
            + (cy - self.coin_y.unsqueeze(1)) ** 2)
        picked = (self.alive.unsqueeze(2) & (dist < 18.0)
                  & (self.coin_timer.unsqueeze(1) > 0))
        any_pick = picked.any(dim=1)
        char_idx = picked.float().argmax(dim=1)
        inc = torch.zeros((e, c), dtype=torch.float32, device=self.device)
        inc.scatter_add_(1, char_idx, any_pick.float())
        collected = inc.long()
        self.coins += collected
        respawn_coins = any_pick | (self.coin_timer <= 0)
        if bool(respawn_coins.any()):
            flat = respawn_coins.nonzero(as_tuple=False)
            self._respawn_random_coins_flat(flat)

        return {
            "collected": collected,
            **self.readout(),
        }

    def _respawn_random_coins_flat(self, flat: torch.Tensor) -> None:
        """Reposiciona moedas em pares `(env, moeda)` em plataformas estáticas."""
        if flat.numel() == 0:
            return
        e = self.n_envs
        s = self.static_count
        n = flat.shape[0]
        picks = self._randint(0, s, (n,)).unsqueeze(1)
        px = torch.gather(
            self.static_t[:, 0].unsqueeze(0).expand(n, -1), 1, picks
        ).squeeze(1)
        py = torch.gather(
            self.static_t[:, 1].unsqueeze(0).expand(n, -1), 1, picks
        ).squeeze(1)
        pw = torch.gather(
            self.static_t[:, 2].unsqueeze(0).expand(n, -1), 1, picks
        ).squeeze(1)
        u = self._rand(n)
        self.coin_x[flat[:, 0], flat[:, 1]] = px + 10.0 + u * (pw - 20.0)
        self.coin_y[flat[:, 0], flat[:, 1]] = py - 18.0
        self.coin_timer[flat[:, 0], flat[:, 1]] = float(
            self.cfg.coin_lifetime)
        self.coin_platform[flat[:, 0], flat[:, 1]] = -1

    # ------------------------------------------------------------------ #
    # Observação opcional (aproximada)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def observation(self) -> torch.Tensor:
        """Grade `(E, 5, gh, gw)` com plataformas, personagens e moedas."""
        cfg = self.cfg
        gh, gw = cfg.height // cfg.cell, cfg.width // cfg.cell
        grid = self._zeros(self.n_envs, 5, gh, gw)
        rows = torch.arange(gh, device=self.device)[None, None, :, None]
        cols = torch.arange(gw, device=self.device)[None, None, None, :]
        for p in range(self.platform_count):
            px = self.platforms[:, p, 0].unsqueeze(1).unsqueeze(1)
            py = self.platforms[:, p, 1].unsqueeze(1).unsqueeze(1)
            pw = self.platforms[:, p, 2].unsqueeze(1).unsqueeze(1)
            ph = self.platforms[:, p, 3].unsqueeze(1).unsqueeze(1)
            inside = ((cols * cfg.cell >= px)
                      & (cols * cfg.cell < px + pw)
                      & (rows * cfg.cell >= py)
                      & (rows * cfg.cell < py + ph))
            grid[:, 0] = torch.where(inside.any(dim=1), torch.ones_like(
                grid[:, 0]), grid[:, 0])
        env = torch.arange(self.n_envs, device=self.device)
        for i in range(self.n_chars):
            ci = (self.x[:, i] / cfg.cell).long().clamp(0, gw - 1)
            ri = (self.y[:, i] / cfg.cell).long().clamp(0, gh - 1)
            channel = 1 if i == 0 else 2
            grid[env, channel, ri, ci] = 1.0
        ci = (self.coin_x / cfg.cell).long().clamp(0, gw - 1)
        ri = (self.coin_y / cfg.cell).long().clamp(0, gh - 1)
        grid[env.unsqueeze(1), 3, ri, ci] = 1.0
        return grid

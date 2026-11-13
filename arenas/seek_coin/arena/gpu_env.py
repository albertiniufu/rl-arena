"""Ambiente vetorizado completo do seek_coin na GPU (PyTorch).

`GpuArenaVecEnv` empacota o `GpuSeekCoin` em um contrato de ambiente
multiagente vetorizado:

- `reset(seed)` → `(obs, info)`;
- `step(actions (E, A))` → `(obs, rewards (E, A), terminated, truncated, info)`;
- observações egocêntricas `(E, A, C*frame_stack, gh, gw)` — 5 canais no modo
  legado (sem `team_ids`) ou 6 canais no multiagente (`self`, inimigos,
  aliados, mortos);
- `frame_skip` substeps por decisão, `frame_stack` frames empilhados;
- recompensas por agente (moedas, mortes, transferências de empurrão),
  penalidades de STOP/empurrão perdido uma vez por passo e PBRS opcional;
- em times, a recompensa de cada membro é a média do time, como no
  `ArenaMultiEnv` da CPU.

É a versão GPU do caminho `ArenaMultiEnv`; não há render nem políticas
baseline embutidas — todos os agentes são controlados externamente.
"""
from __future__ import annotations

from collections import deque

import torch

from .actions import PUSH_LEFT, PUSH_RIGHT, STOP
from .config import ArenaConfig
from .gpu_engine import GpuSeekCoin


class GpuArenaVecEnv(GpuSeekCoin):
    """Lote de ambientes multiagente com observação, recompensa e frame stack."""

    def __init__(self, config: ArenaConfig | None = None, n_envs: int = 256,
                 n_agents: int = 4, level: int = 1, team_ids=None,
                 device: str = "cuda", seed: int = 0):
        cfg = config or ArenaConfig(
            level=level, n_agents=n_agents,
            team_ids=tuple(team_ids or range(n_agents)))
        self.cfg = cfg
        self.n_agents = int(n_agents)
        self.team_ids = (None if team_ids is None
                         else tuple(int(t) for t in team_ids))
        if self.team_ids is None and getattr(cfg, "team_ids", ()):
            self.team_ids = tuple(int(t) for t in cfg.team_ids)
        self.frame_skip = int(cfg.frame_skip)
        self.frame_stack = int(cfg.frame_stack)
        self.channels = 6 if self.team_ids is not None else 5
        self.pbrs_coef = float(cfg.pbrs_coef)
        self.pbrs_gamma = float(cfg.pbrs_gamma)
        self.episode_seconds = float(cfg.episode_seconds)
        self.grid_h = cfg.height // cfg.cell
        self.grid_w = cfg.width // cfg.cell
        self._frames = deque(maxlen=self.frame_stack)
        super().__init__(
            n_envs=n_envs, n_chars=self.n_agents, level=int(cfg.level),
            device=device, seed=seed, coin_count=int(cfg.coin_count),
            team_ids=self.team_ids)
        self.cfg = cfg
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Gym-like
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.generator.manual_seed(int(seed))
        super().reset()
        frame = self._observation_all()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame.clone())
        return self._stacked(), {"sim_time": torch.zeros(
            self.n_envs, device=self.device)}

    def step(self, actions):
        actions = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        ).reshape(self.n_envs, self.n_agents)
        push_missed = self.push_missed(actions)
        phi_before = self._potential() if self.pbrs_coef else None

        raw = self._zeros(self.n_envs, self.n_agents)
        collected = torch.zeros(
            (self.n_envs, self.n_agents), dtype=torch.int64,
            device=self.device)
        deaths = torch.zeros_like(collected)
        transferred = torch.zeros_like(collected)
        for _ in range(self.frame_skip):
            event = super().step(actions)
            raw += (event["collected"].float() * self.cfg.coin_reward
                    + event["deaths_delta"].float()
                    * self.cfg.death_penalty
                    + event["coins_transferred"].float()
                    * self.cfg.push_coin_reward)
            collected += event["collected"]
            deaths += event["deaths_delta"]
            transferred += event["coins_transferred"]

        pbrs = self._zeros(self.n_envs, self.n_agents)
        if self.pbrs_coef:
            phi_after = self._potential()
            pbrs = self.pbrs_coef * (
                self.pbrs_gamma * phi_after - phi_before)
            raw += pbrs

        raw += torch.where(
            actions == STOP,
            torch.full_like(raw, self.cfg.stop_penalty),
            torch.zeros_like(raw))
        raw += torch.where(
            push_missed,
            torch.full_like(raw, self.cfg.push_miss_penalty),
            torch.zeros_like(raw))
        rewards = self._aggregate_rewards(raw)

        frame = self._observation_all()
        self._frames.append(frame)
        truncated_value = bool(
            self.sim_time[0] >= self.episode_seconds)
        truncated = torch.full(
            (self.n_envs, self.n_agents), truncated_value,
            dtype=torch.bool, device=self.device)
        terminated = torch.zeros_like(truncated)
        info = {
            "sim_time": self.sim_time,
            "coins": self.coins,
            "deaths": self.deaths,
            "collected_this_step": collected,
            "deaths_this_step": deaths,
            "coins_transferred_this_step": transferred,
            "pbrs_this_step": pbrs,
        }
        return self._stacked(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observação
    # ------------------------------------------------------------------ #
    def _platform_mask(self) -> torch.Tensor:
        cfg = self.cfg
        cell, gh, gw = cfg.cell, self.grid_h, self.grid_w
        mask = torch.zeros(
            (self.n_envs, gh, gw), dtype=torch.bool, device=self.device)
        rows = torch.arange(gh, device=self.device)[None, :, None]
        cols = torch.arange(gw, device=self.device)[None, None, :]
        for p in range(self.platform_count):
            px = self.platforms[:, p, 0]
            py = self.platforms[:, p, 1]
            pw = self.platforms[:, p, 2]
            ph = self.platforms[:, p, 3]
            c0 = torch.clamp((px // cell).long(), 0, gw)
            c1 = torch.clamp(((px + pw) // cell).long(), 0, gw)
            r0 = torch.clamp((py // cell).long(), 0, gh)
            r1 = torch.clamp(
                torch.maximum(((py + ph) // cell).long(),
                              (py // cell).long() + 1), 0, gh)
            col_ok = ((cols >= c0[:, None, None])
                      & (cols < c1[:, None, None]))
            row_ok = ((rows >= r0[:, None, None])
                      & (rows < r1[:, None, None]))
            mask |= (row_ok & col_ok)
        return mask

    def _observation_all(self) -> torch.Tensor:
        cfg = self.cfg
        e, a = self.n_envs, self.n_agents
        cell, gh, gw = cfg.cell, self.grid_h, self.grid_w
        grid = self._zeros(e, a, self.channels, gh, gw)
        platform_mask = self._platform_mask()
        env = torch.arange(e, device=self.device)
        coin_col = (self.coin_x / cell).long().clamp(0, gw - 1)
        coin_row = (self.coin_y / cell).long().clamp(0, gh - 1)

        for viewer in range(a):
            grid[:, viewer, 0] = platform_mask
            for i in range(a):
                if bool(self.alive[:, i].any()):
                    alive = self.alive[:, i]
                    if self.team_ids is None:
                        channel = 1 if i == viewer else 2
                    elif i == viewer:
                        channel = 1
                    elif self.team_ids[i] == self.team_ids[viewer]:
                        channel = 5 if self.channels > 5 else 2
                    else:
                        channel = 2
                    y = self.y[:, i] - cfg.player_size / 2.0
                    col = (self.x[:, i] / cell).long().clamp(0, gw - 1)
                    row = (y / cell).long().clamp(0, gh - 1)
                    grid[env[alive], viewer, channel,
                         row[alive], col[alive]] = 1.0
                    dead = ~alive
                    if bool(dead.any()):
                        dead_row = torch.full_like(
                            col, gh - 1, device=self.device)
                        grid[env[dead], viewer, 4,
                             dead_row[dead], col[dead]] = 1.0
                else:
                    col = (self.x[:, i] / cell).long().clamp(0, gw - 1)
                    dead_row = torch.full_like(col, gh - 1)
                    grid[env, viewer, 4, dead_row, col] = 1.0
            grid[env.unsqueeze(1), viewer, 3,
                 coin_row, coin_col] = 1.0
        return grid

    def _stacked(self) -> torch.Tensor:
        return torch.cat(list(self._frames), dim=2)

    # ------------------------------------------------------------------ #
    # Recompensas
    # ------------------------------------------------------------------ #
    def _potential(self) -> torch.Tensor:
        coins = self.coins.float()
        total = coins.sum(dim=1, keepdim=True)
        if self.team_ids is None:
            return 2.0 * coins - total
        potential = torch.zeros_like(coins)
        for team in sorted(set(self.team_ids)):
            mask = torch.tensor(
                [t == team for t in self.team_ids],
                dtype=torch.bool, device=self.device)
            own = (coins * mask).sum(dim=1, keepdim=True)
            other = total - own
            potential = torch.where(
                mask.unsqueeze(0), own - other, potential)
        return potential

    def _aggregate_rewards(self, raw: torch.Tensor) -> torch.Tensor:
        if self.team_ids is None:
            return raw
        rewards = raw.clone()
        for team in sorted(set(self.team_ids)):
            mask = torch.tensor(
                [t == team for t in self.team_ids],
                dtype=torch.bool, device=self.device)
            mean = raw[:, mask].mean(dim=1, keepdim=True)
            rewards[:, mask] = mean.expand(-1, int(mask.sum()))
        return rewards

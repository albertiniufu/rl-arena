"""Ambiente multiagente simultâneo para IPPO.

Diferente do `ArenaEnv` (um aprendiz + adversários baseline), aqui **todos**
os personagens são controlados externamente: `step` recebe o vetor de ações e
devolve observações, recompensas, terminadores e truncamentos por agente.

Cada agente vê uma grade egocêntrica de 6 canais (ver `arena/observations.py`)
e tem o seu próprio frame stack. As ações são decididas uma vez por step e
mantidas durante os `frame_skip` substeps de física — a mesma semântica do
jogador no ambiente single-agent. Episódios só terminam por truncamento
(tempo); morte custa moedas e respawna, mas não encerra nada.

Em times a recompensa de cada membro é a **média** das recompensas brutas do
time (escala estável independente do tamanho do time). O `info` mantém também
a recompensa bruta de cada agente e o agregado por time.
"""
from collections import deque

import numpy as np
from gymnasium import spaces

from .actions import N_ACTIONS, PUSH_LEFT, PUSH_RIGHT, STOP
from .config import ArenaConfig
from .engine import ArenaEngine
from .levels import get_hazard_kinds
from .observations import render_grid
from .reward_shaping import coin_potential
from .scenario import Scenario
from .skins import load_renderer_class

N_CHANNELS = 6


class ArenaMultiEnv:
    """Ambiente de ações conjuntas `(n_agents,)` e observações por agente."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None, scenario=None, render_mode=None, skin=None):
        self.cfg = config or ArenaConfig()
        if scenario is None:
            scenario = Scenario.from_config(self.cfg)
        self.scenario = scenario
        self.cfg.n_agents = scenario.n_agents
        self.cfg.team_ids = tuple(scenario.team_ids)
        self.render_mode = render_mode
        self.skin = skin
        self.engine = ArenaEngine(self.cfg)
        self.n_agents = scenario.n_agents
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.grid_h = self.cfg.height // self.cfg.cell
        self.grid_w = self.cfg.width // self.cfg.cell
        self.hazard_channels = len(get_hazard_kinds(self.cfg.level))
        self.observation_space = spaces.Box(
            0.0, 1.0,
            shape=((N_CHANNELS + self.hazard_channels)
                   * self.cfg.frame_stack, self.grid_h, self.grid_w),
            dtype=np.float32,
        )
        # Um frame stack por agente: a visão é egocêntrica, então cada um tem
        # a própria sequência de grades.
        self._frames = [deque(maxlen=self.cfg.frame_stack)
                        for _ in range(self.n_agents)]
        self.elapsed = 0.0
        self._renderer = None
        self._last_info = {}

    def _observation(self, viewer):
        return render_grid(
            self.engine, self.cfg, viewer=viewer,
            channels=N_CHANNELS + self.hazard_channels,
            team_ids=self.scenario.team_ids)

    def _stacked(self):
        return np.stack([
            np.concatenate(frames, axis=0) for frames in self._frames
        ])

    def _info(self, rewards=None, raw=None, collected=None, deaths=None,
              transferred=None, pbrs=None):
        agents = []
        for i, c in enumerate(self.engine.characters):
            agents.append({
                "coins": int(c.coins),
                "deaths": int(c.deaths),
                "team": int(self.scenario.team_of(i)),
                "reward": 0.0 if rewards is None else float(rewards[i]),
                "raw_reward": 0.0 if raw is None else float(raw[i]),
                "collected_this_step": 0 if collected is None else int(collected[i]),
                "deaths_this_step": 0 if deaths is None else int(deaths[i]),
                "coins_transferred_this_step": (
                    0 if transferred is None else int(transferred[i])),
                "pbrs_this_step": 0.0 if pbrs is None else float(pbrs[i]),
            })
        teams = []
        for team in sorted(set(self.scenario.team_ids)):
            members = [a for a in agents if a["team"] == team]
            teams.append({
                "team": int(team),
                "reward": float(np.mean([a["reward"] for a in members])),
                "coins": int(sum(a["coins"] for a in members)),
                "deaths": int(sum(a["deaths"] for a in members)),
                "pbrs": float(np.mean([a["pbrs_this_step"] for a in members])),
            })
        return {
            "sim_time": float(self.engine.sim_time),
            "team_ids": tuple(self.scenario.team_ids),
            "agents": agents,
            "teams": teams,
        }

    def reset(self, seed=None, options=None):
        self.engine.reset(
            seed=seed, n_characters=self.n_agents,
            spawn_positions=self.scenario.spawn_positions(self.cfg))
        self.elapsed = 0.0
        initial = []
        for i in range(self.n_agents):
            frame = self._observation(i)
            frames = self._frames[i]
            frames.clear()
            frames.extend([frame] * self.cfg.frame_stack)
            initial.append(np.concatenate(frames, axis=0))
        self._last_info = self._info()
        return np.stack(initial), self._last_info

    def _push_target_in_range(self, actor_index):
        """Há inimigo vivo em contato no início do step?"""
        actor = self.engine.characters[actor_index]
        if not actor.alive:
            return False
        return any(
            self.engine.characters[j].alive
            and self.engine.touching(actor, self.engine.characters[j])
            for j in self.scenario.enemies(actor_index)
        )

    def _aggregate_rewards(self, raw):
        """Em times, cada membro recebe a média das recompensas do time.

        Em todos-contra-todos cada grupo tem um único membro, então a operação
        é a identidade.
        """
        rewards = np.array(raw, dtype=np.float64, copy=True)
        groups = {}
        for i, team in enumerate(self.scenario.team_ids):
            groups.setdefault(team, []).append(i)
        for members in groups.values():
            rewards[members] = float(np.mean(rewards[members]))
        return rewards

    def step(self, actions, render_intermediate=False):
        """Avança a simulação; `render_intermediate` desenha cada substep.

        No modo humano isso mantém o render a `fps` (30) mesmo com
        `frame_skip > 1`: as decisões continuam uma vez por step do agente,
        mas a tela atualiza a cada substep de física.
        """
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape[0] != self.n_agents:
            raise ValueError(
                f"esperado {self.n_agents} ações, recebido {actions.shape[0]}")
        if np.any(actions < 0) or np.any(actions >= N_ACTIONS):
            raise ValueError(f"ações inválidas: {actions.tolist()}")
        # PBRS: potencial no estado em que a ação foi escolhida. Aplicado uma
        # vez por step do agente (não por substep), casando com o gamma do GAE.
        if self.cfg.pbrs_coef:
            phi_before = np.array([
                coin_potential(self.engine, i, self.scenario.team_ids)
                for i in range(self.n_agents)])
        push_missed = np.array([
            bool(a in (PUSH_LEFT, PUSH_RIGHT))
            and not self._push_target_in_range(i)
            for i, a in enumerate(actions)
        ], dtype=bool)

        dt = 1.0 / self.cfg.fps
        raw = np.zeros(self.n_agents, dtype=np.float64)
        collected = np.zeros(self.n_agents, dtype=np.int64)
        deaths = np.zeros(self.n_agents, dtype=np.int64)
        transferred = np.zeros(self.n_agents, dtype=np.int64)
        for _ in range(self.cfg.frame_skip):
            event = self.engine.step(actions.tolist(), dt)
            self.elapsed += dt
            step_collected = np.asarray(event["collected"], dtype=np.int64)
            step_deaths = np.asarray(event["deaths_delta"], dtype=np.int64)
            # Transferência assinada: ganha quem empurra, perde o alvo.
            step_transferred = np.asarray(event["coins_transferred"], dtype=np.int64)
            raw += (step_collected * self.cfg.coin_reward
                    + step_deaths * self.cfg.death_penalty
                    + step_transferred * self.cfg.push_coin_reward)
            collected += step_collected
            deaths += step_deaths
            transferred += step_transferred
            if render_intermediate and self.render_mode == "human":
                self.render()

        pbrs = np.zeros(self.n_agents, dtype=np.float64)
        if self.cfg.pbrs_coef:
            phi_after = np.array([
                coin_potential(self.engine, i, self.scenario.team_ids)
                for i in range(self.n_agents)])
            pbrs = self.cfg.pbrs_coef * (
                self.cfg.pbrs_gamma * phi_after - phi_before)
            raw += pbrs

        # Penalidades de modelagem: uma vez por step do agente, não por substep.
        raw += np.where(actions == STOP, self.cfg.stop_penalty, 0.0)
        raw += np.where(push_missed, self.cfg.push_miss_penalty, 0.0)
        rewards = self._aggregate_rewards(raw)

        for i in range(self.n_agents):
            self._frames[i].append(self._observation(i))
        obs = self._stacked()
        terminated = np.zeros(self.n_agents, dtype=bool)
        truncated = np.full(
            self.n_agents, self.elapsed >= self.cfg.episode_seconds, dtype=bool)
        info = self._info(rewards, raw, collected, deaths, transferred, pbrs)
        self._last_info = info
        if self.render_mode == "human" and not render_intermediate:
            self.render()
        return obs, rewards, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = load_renderer_class(self.skin)(self.cfg)
        self._renderer.render(self.engine, self.elapsed, self._last_info)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

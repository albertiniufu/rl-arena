from dataclasses import asdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.baseline import PolicyContext, make_policy
from .config import ArenaConfig
from .engine import ArenaEngine, Character, Coin

ACTIONS = 12

class ArenaEnv(gym.Env):
    """Gymnasium environment. Observation is a channels-first semantic grid.

    O adversário é controlado por uma política de `agents.baseline`,
    resolvida por nome ou recebida pronta em `opponent_policy`.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None, render_mode=None, opponent_policy="rule"):
        super().__init__()
        self.cfg = config or ArenaConfig()
        self.render_mode = render_mode
        self.engine = ArenaEngine(self.cfg)
        self.opponent_policy = make_policy(opponent_policy)
        self.action_space = spaces.Discrete(ACTIONS)
        self.grid_h = self.cfg.height // self.cfg.cell
        self.grid_w = self.cfg.width // self.cfg.cell
        self.observation_space = spaces.Box(0.0, 1.0, shape=(5, self.grid_h, self.grid_w), dtype=np.float32)
        self.elapsed = 0.0
        self._renderer = None
        self._last_info = {}

    def _draw_cell(self, grid, channel, x, y):
        col = int(np.clip(x // self.cfg.cell, 0, self.grid_w - 1))
        row = int(np.clip(y // self.cfg.cell, 0, self.grid_h - 1))
        grid[channel, row, col] = 1.0

    def _observation(self):
        g = np.zeros(self.observation_space.shape, dtype=np.float32)
        for x, y, w, h in self.engine.platforms:
            c0, c1 = max(0, int(x // self.cfg.cell)), min(self.grid_w, int((x+w) // self.cfg.cell))
            r0, r1 = max(0, int(y // self.cfg.cell)), min(self.grid_h, max(int((y+h) // self.cfg.cell), int(y // self.cfg.cell)+1))
            g[0, r0:r1, c0:c1] = 1
        for i, c in enumerate(self.engine.characters):
            if c.alive:
                self._draw_cell(g, 1 if i == 0 else 2, c.x, c.y - self.cfg.player_size / 2)
            else:
                self._draw_cell(g, 4, c.x, self.cfg.height - 5)
        for coin in self.engine.coins:
            self._draw_cell(g, 3, coin.x, coin.y)
        return g

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        self.elapsed = 0.0
        self._last_info = {}
        return self._observation(), {"sim_time": 0.0}

    def _opponent_action(self, index, obs):
        ctx = PolicyContext(
            obs=obs,
            rng=self.np_random,
            self_character=self.engine.characters[index],
            player_character=self.engine.characters[0],
        )
        return int(self.opponent_policy.act(ctx))

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        dt = 1.0 / self.cfg.fps
        # Observação pré-passo: é o estado sobre o qual as políticas decidem.
        policy_obs = self._observation()
        actions = [int(action)] + [self._opponent_action(i, policy_obs) for i in range(1, len(self.engine.characters))]
        event = self.engine.step(actions, dt)
        self.elapsed += dt
        p = self.engine.characters[0]
        reward = (
            event["collected"][0] * self.cfg.coin_reward
            + event["eliminations_delta"][0] * self.cfg.elimination_reward
            + event["deaths_delta"][0] * self.cfg.death_penalty
        )
        # Episodes end only when the configured simulation-time limit is reached.
        # Death does not end the episode: the character loses coins and respawns later.
        terminated = False
        truncated = self.elapsed >= self.cfg.episode_seconds
        info = {
            "coins": p.coins,
            "deaths": p.deaths,
            "eliminations": p.eliminations,
            "sim_time": self.engine.sim_time,
            "collected_this_step": event["collected"][0],
            "deaths_this_step": event["deaths_delta"][0],
        }
        self._last_info = info
        obs = self._observation()
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def get_state(self):
        """Snapshot serializável do ambiente, para retomada reproduzível."""
        return {
            "elapsed": self.elapsed,
            "sim_time": self.engine.sim_time,
            "characters": [asdict(c) for c in self.engine.characters],
            "coins": [asdict(c) for c in self.engine.coins],
        }

    def set_state(self, state):
        self.elapsed = float(state["elapsed"])
        self.engine.sim_time = float(state["sim_time"])
        self.engine.characters = [Character(**s) for s in state["characters"]]
        self.engine.coins = [Coin(**c) for c in state["coins"]]

    def render(self):
        if self._renderer is None:
            from .renderer import ArenaRenderer
            self._renderer = ArenaRenderer(self.cfg)
        self._renderer.render(self.engine, self.elapsed, self._last_info)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

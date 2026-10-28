from collections import deque
from dataclasses import asdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.baseline import PolicyContext, make_policy
from .actions import N_ACTIONS, PUSH_LEFT, PUSH_RIGHT, STOP
from .config import ArenaConfig
from .engine import ArenaEngine, Character, Coin
from .skins import load_renderer_class

class ArenaEnv(gym.Env):
    """Gymnasium environment. Observation is a channels-first semantic grid.

    A observação é a pilha dos últimos `frame_stack` frames (o mais antigo
    primeiro), o que dá ao agente acesso a velocidade na grade.

    O adversário é controlado por uma política de `agents.baseline`,
    resolvida por nome ou recebida pronta em `opponent_policy`.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None, render_mode=None, opponent_policy="rule", skin=None):
        super().__init__()
        self.cfg = config or ArenaConfig()
        self.render_mode = render_mode
        self.engine = ArenaEngine(self.cfg)
        self.opponent_policy = make_policy(opponent_policy)
        self.skin = skin
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.grid_h = self.cfg.height // self.cfg.cell
        self.grid_w = self.cfg.width // self.cfg.cell
        self.observation_space = spaces.Box(
            0.0, 1.0,
            shape=(5 * self.cfg.frame_stack, self.grid_h, self.grid_w),
            dtype=np.float32,
        )
        # Pilha de frames: cada elemento é um frame único (5, H, W).
        self._frames = deque(maxlen=self.cfg.frame_stack)
        self.elapsed = 0.0
        self._renderer = None
        self._last_info = {}

    def _observation(self):
        """Frame único atual: grade semântica (5, H, W).

        O clamp é feito em Python puro de propósito: `np.clip` sobre um
        escalar custa dezenas de microssegundos por célula, e esta função
        roda `frame_skip + 1` vezes por passo do agente (a maior parte do
        custo do `env.step` em CPU). O resultado é idêntico ao de antes.
        """
        cell, gh, gw = self.cfg.cell, self.grid_h, self.grid_w
        g = np.zeros((5, gh, gw), dtype=np.float32)
        for x, y, w, h in self.engine.platforms:
            c0, c1 = max(0, int(x // cell)), min(gw, int((x+w) // cell))
            r0, r1 = max(0, int(y // cell)), min(gh, max(int((y+h) // cell), int(y // cell)+1))
            g[0, r0:r1, c0:c1] = 1
        for i, c in enumerate(self.engine.characters):
            if c.alive:
                channel = 1 if i == 0 else 2
                y = c.y - self.cfg.player_size / 2
            else:
                channel = 4
                y = self.cfg.height - 5
            col, row = int(c.x // cell), int(y // cell)
            col = 0 if col < 0 else (gw - 1 if col >= gw else col)
            row = 0 if row < 0 else (gh - 1 if row >= gh else row)
            g[channel, row, col] = 1.0
        for coin in self.engine.coins:
            col, row = int(coin.x // cell), int(coin.y // cell)
            col = 0 if col < 0 else (gw - 1 if col >= gw else col)
            row = 0 if row < 0 else (gh - 1 if row >= gh else row)
            g[3, row, col] = 1.0
        return g

    def _stacked(self):
        """Observação do agente: pilha de frames (5 * frame_stack, H, W), do mais antigo ao mais novo."""
        return np.concatenate(self._frames, axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        self.elapsed = 0.0
        self._last_info = {}
        # A pilha começa cheia com cópias do primeiro frame: sem histórico
        # inventado, mas com o shape esperado desde o primeiro passo.
        frame = self._observation()
        self._frames.clear()
        self._frames.extend([frame] * self.cfg.frame_stack)
        return self._stacked(), {"sim_time": 0.0}

    def _opponent_action(self, index, obs):
        ctx = PolicyContext(
            obs=obs,
            rng=self.np_random,
            self_character=self.engine.characters[index],
            player_character=self.engine.characters[0],
            # Pilha do início do step: `_frames` só ganha o frame novo no fim
            # de `step()`, então aqui é exatamente o que o DQN do adversário
            # viu em treino (frames dos steps anteriores).
            stacked_obs=self._stacked(),
        )
        return int(self.opponent_policy.act(ctx))

    def _push_target_in_range(self):
        """Há oponente vivo em contato com o jogador no início do step?

        Usa a mesma regra `touching` do engine: se não há alvo, o empurrão
        não produz efeito e o step paga `push_miss_penalty`.
        """
        actor = self.engine.characters[0]
        return actor.alive and any(
            c.alive and self.engine.touching(actor, c)
            for c in self.engine.characters[1:]
        )

    def step(self, action):
        """Avança `frame_skip` substeps de física com a ação do jogador repetida.

        A cada substep o adversário decide de novo, sobre a observação corrente,
        e a recompensa acumula. O `info` traz os valores finais do personagem
        (não somas) e os totais do step em `*_this_step`.

        A penalidade de STOP (`stop_penalty`) é cobrada uma vez por step do
        agente, não por substep: assim ela não escala com `frame_skip`. Um
        empurrão sem oponente em contato no início do step também custa
        `push_miss_penalty` uma vez por step.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        action = int(action)
        push_missed = action in (PUSH_LEFT, PUSH_RIGHT) and not self._push_target_in_range()
        dt = 1.0 / self.cfg.fps
        reward = 0.0
        collected = deaths = 0
        for _ in range(self.cfg.frame_skip):
            # Observação pré-substep: é o estado sobre o qual as políticas decidem.
            policy_obs = self._observation()
            actions = [int(action)] + [self._opponent_action(i, policy_obs) for i in range(1, len(self.engine.characters))]
            event = self.engine.step(actions, dt)
            self.elapsed += dt
            reward += (
                event["collected"][0] * self.cfg.coin_reward
                + event["deaths_delta"][0] * self.cfg.death_penalty
            )
            collected += event["collected"][0]
            deaths += event["deaths_delta"][0]
        if action == STOP:
            reward += self.cfg.stop_penalty
        elif push_missed:
            reward += self.cfg.push_miss_penalty
        p = self.engine.characters[0]
        # Episodes end only when the configured simulation-time limit is reached.
        # Death does not end the episode: the character loses coins and respawns later.
        terminated = False
        truncated = self.elapsed >= self.cfg.episode_seconds
        info = {
            "coins": p.coins,
            "deaths": p.deaths,
            "sim_time": self.engine.sim_time,
            "collected_this_step": collected,
            "deaths_this_step": deaths,
        }
        self._last_info = info
        self._frames.append(self._observation())
        if self.render_mode == "human":
            self.render()
        return self._stacked(), float(reward), terminated, truncated, info

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
            self._renderer = load_renderer_class(self.skin)(self.cfg)
        self._renderer.render(self.engine, self.elapsed, self._last_info)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

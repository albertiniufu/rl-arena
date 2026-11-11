"""Ambiente Gymnasium do jogo de conquista.

O agente controla o jogador 0 contra bots nos demais jogadores. A cada
`step`, o agente executa uma ação legal e, em seguida, o ambiente roda os
turnos dos oponentes até voltar ao agente (ou até a partida terminar).

Ações são codificadas em `Discrete(action_size)` (ver `RiskGame.decode`):
0 encerra a fase, 1..n coloca um exército e n+1..n+n² move/ataca de uma
origem para um destino. Use `action_masks()` para mascarar ações inválidas.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agents import RiskBot
from .config import RiskConfig
from .game import RiskGame


class RiskEnv(gym.Env):
    """Conquista de territórios para um agente contra bots."""

    metadata = {"render_modes": []}

    def __init__(self, n_players: int = 2, initial_armies: int = 20,
                 seed: int = 0, max_turns: int = 200,
                 opponent: RiskBot | None = None,
                 config: RiskConfig | None = None,
                 invalid_action_penalty: float = 0.1):
        super().__init__()
        if config is not None:
            n_players = config.n_players
            initial_armies = config.initial_armies
            seed = config.seed
            max_turns = config.max_turns
            invalid_action_penalty = config.invalid_action_penalty
        self.invalid_action_penalty = float(invalid_action_penalty)
        self._last_invalid_action = False
        self.n_players = int(n_players)
        self.game = RiskGame(n_players=n_players,
                             initial_armies=initial_armies, seed=seed)
        self.opponent = opponent or RiskBot(seed=seed + 1)
        self.max_turns = int(max_turns)
        self.agent_player = 0
        self.action_space = spaces.Discrete(self.game.action_size)
        obs_dim = self.game.n_territories * (self.n_players + 1) + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._steps = 0

    # ------------------------------------------------------------------ #
    # Contrato Gymnasium
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None and hasattr(self.opponent, "rng"):
            self.opponent.rng = np.random.default_rng(seed + 1)
        self.game.reset(seed=seed)
        self._steps = 0
        self._last_invalid_action = False
        self._run_opponents()
        return self.game.observation(self.agent_player), self._info()

    def step(self, action):
        if self.game.winner is not None:
            raise RuntimeError("episódio terminado; chame reset()")
        if self.game.current_player != self.agent_player:
            raise RuntimeError("não é o turno do agente")

        legal = self.game.legal_actions()
        action = int(action)
        self._last_invalid_action = action not in legal
        if self._last_invalid_action:
            # O agente pode escolher uma ação estruturalmente inválida para a
            # fase; cai na primeira ação legal e paga uma pequena penalidade.
            if not legal:
                raise RuntimeError("não há ação legal para o agente")
            action = int(legal[0])
            reward = -self.invalid_action_penalty
        else:
            reward = 0.0

        before = self._territory_count(self.agent_player)
        self.game.apply_action(action)
        after = self._territory_count(self.agent_player)
        reward += 0.1 * (after - before)

        terminated = False
        if self.game.winner is not None:
            terminated = True
            reward += 10.0 if self.game.winner == self.agent_player else -10.0
        else:
            reward += self._run_opponents()
            terminated = self.game.winner is not None

        self._steps += 1
        truncated = (not terminated) and self._steps >= self.max_turns
        return (self.game.observation(self.agent_player), float(reward),
                terminated, truncated, self._info())

    # ------------------------------------------------------------------ #
    # Máscara de ações (estilo Stable-Baselines3)
    # ------------------------------------------------------------------ #
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if (self.game.winner is None
                and self.game.current_player == self.agent_player):
            mask[self.game.legal_actions()] = True
        return mask

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _run_opponents(self) -> float:
        reward = 0.0
        guard = 0
        while (self.game.winner is None
               and self.game.current_player != self.agent_player):
            action = self.opponent.act(self.game)
            if action is None:
                break
            before = self._territory_count(self.agent_player)
            self.game.apply_action(action)
            after = self._territory_count(self.agent_player)
            reward += 0.1 * (after - before)
            guard += 1
            if guard > 10000:
                raise RuntimeError("laço infinito nos turnos dos oponentes")
        if (self.game.winner is not None
                and self.game.winner != self.agent_player):
            reward -= 10.0
        return reward

    def _territory_count(self, player: int) -> int:
        return int(np.count_nonzero(self.game.owner == player))

    def _info(self) -> dict:
        return {
            "legal_actions": self.game.legal_actions(),
            "action_mask": self.action_mask_safe(),
            "winner": self.game.winner,
            "turn": self.game.turn,
            "invalid_action": self._last_invalid_action,
        }

    def action_mask_safe(self) -> np.ndarray:
        return self.action_masks()

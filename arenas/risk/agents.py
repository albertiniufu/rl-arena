"""Política baseline (bot) para o jogo de conquista."""
from __future__ import annotations

import numpy as np

from .board import ADJACENCY_INDEX


def _enemy_neighbors(game, territory: int, player: int) -> list[int]:
    return [int(t) for t in ADJACENCY_INDEX[territory]
            if game.owner[t] != player]


def _threat(game, territory: int, player: int) -> int:
    enemies = _enemy_neighbors(game, territory, player)
    return max((int(game.armies[t]) for t in enemies), default=0)


class RiskBot:
    """Bot heurístico: reforça a fronteira, ataca a vantagem e fortifica
    do interior para a borda. Pode explorar ações aleatórias."""

    def __init__(self, seed: int = 0, attack_threshold: int = 1,
                 explore: float = 0.05):
        self.rng = np.random.default_rng(seed)
        self.attack_threshold = attack_threshold
        self.explore = explore

    def act(self, game) -> int | None:
        legal = game.legal_actions()
        if not legal:
            return None
        if self.explore and self.rng.random() < self.explore:
            return int(legal[int(self.rng.integers(len(legal)))])
        if game.phase == "reinforce":
            return self._reinforce(game, legal)
        if game.phase == "attack":
            return self._attack(game, legal)
        return self._fortify(game, legal)

    def _reinforce(self, game, legal: list[int]) -> int:
        player = game.current_player
        best, best_score = legal[0], -1e9
        for action in legal:
            territory = game.decode(action)[1]
            score = (2.0 * _threat(game, territory, player)
                     - 0.5 * game.armies[territory])
            if score > best_score:
                best, best_score = action, score
        return int(best)

    def _attack(self, game, legal: list[int]) -> int:
        player = game.current_player
        best, best_score = 0, -1e9
        for action in legal:
            if action == 0:
                continue
            _, src, dst = game.decode(action)
            advantage = int(game.armies[src]) - int(game.armies[dst])
            score = advantage + (2.0 if game.armies[dst] == 1 else 0.0)
            if score > best_score:
                best, best_score = action, score
        if best_score < self.attack_threshold:
            return 0
        return int(best)

    def _fortify(self, game, legal: list[int]) -> int:
        player = game.current_player
        best, best_score = 0, -1e9
        for action in legal:
            if action == 0:
                continue
            _, src, dst = game.decode(action)
            score = (_threat(game, dst, player)
                     - _threat(game, src, player)
                     + 0.1 * game.armies[src])
            if score > best_score:
                best, best_score = action, score
        return int(best) if best_score > 0 else 0

"""Motor puro do jogo de conquista.

Regras simplificadas:

- cada território começa com 1 exército; cada jogador recebe
  `initial_armies` no total, distribuídos aleatoriamente;
- a cada turno o jogador reforça (`max(3, territórios // 3)` + bônus de
  continentes dominados), ataca e pode fortificar uma vez;
- combate por dados: atacante rola até 3 dados, defensor até 2; dados
  ordenados são comparados e a derrota é de quem tiver o menor valor
  (empate é do defensor);
- vence quem conquistar todos os territórios.

O motor não conhece Gym, pygame nem texto: só recebe ações e devolve estado.
"""
from __future__ import annotations

import numpy as np

from .board import (
    ADJACENCY_INDEX,
    CONTINENT_BONUS,
    CONTINENT_TERRITORIES,
    TERRITORIES,
)

PHASES = ("reinforce", "attack", "fortify")
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}


class RiskGame:
    """Estado e regras de uma partida de conquista de territórios."""

    def __init__(self, n_players: int = 2, initial_armies: int = 20,
                 seed: int = 0):
        if n_players < 2:
            raise ValueError("são necessários ao menos 2 jogadores")
        self.n_players = int(n_players)
        self.initial_armies = int(initial_armies)
        self.n_territories = len(TERRITORIES)
        if self.n_players > self.n_territories:
            raise ValueError("jogadores demais para o número de territórios")
        # 0 = encerrar fase; 1..n = colocar em um território;
        # n+1..n+n² = movimento (src, dst) em ataque ou fortificação.
        self.action_size = (self.n_territories * self.n_territories
                            + self.n_territories + 1)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> "RiskGame":
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        n = self.n_territories
        self.owner = np.full(n, -1, dtype=int)
        order = self.rng.permutation(n)
        for position, territory in enumerate(order):
            self.owner[territory] = position % self.n_players
        self.armies = np.ones(n, dtype=int)

        # Distribui os exércitos restantes de cada jogador.
        for player in range(self.n_players):
            owned = np.flatnonzero(self.owner == player)
            extra = max(0, self.initial_armies - len(owned))
            if extra:
                picks = self.rng.integers(0, len(owned), size=extra)
                np.add.at(self.armies, owned[picks], 1)

        self.current_player = 0
        self.phase = "reinforce"
        self.turn = 1
        self.reinforcements_left = self.reinforcement_count(0)
        self.fortified = False
        self.winner: int | None = None
        self.last_battle: dict | None = None
        return self

    # ------------------------------------------------------------------ #
    # Consultas
    # ------------------------------------------------------------------ #
    def territories_owned(self, player: int) -> np.ndarray:
        return np.flatnonzero(self.owner == player)

    def reinforcement_count(self, player: int) -> int:
        owned = self.territories_owned(player)
        total = max(3, len(owned) // 3)
        for continent, indices in CONTINENT_TERRITORIES.items():
            if len(indices) and np.all(self.owner[list(indices)] == player):
                total += CONTINENT_BONUS[continent]
        return int(total)

    def _encode_move(self, src: int, dst: int) -> int:
        return self.n_territories + 1 + src * self.n_territories + dst

    def decode(self, action: int) -> tuple[str, int | None, int | None]:
        """Converte a ação codificada em (tipo, origem, destino)."""
        action = int(action)
        if action == 0:
            return ("end", None, None)
        if 1 <= action <= self.n_territories:
            return ("place", action - 1, None)
        value = action - (self.n_territories + 1)
        if not 0 <= value < self.n_territories ** 2:
            raise ValueError(f"ação fora do espaço: {action}")
        return ("move", value // self.n_territories,
                value % self.n_territories)

    def legal_actions(self) -> list[int]:
        if self.winner is not None:
            return []
        n = self.n_territories
        player = self.current_player
        owned = self.territories_owned(player)

        if self.phase == "reinforce":
            if self.reinforcements_left <= 0:
                return []
            return [1 + int(t) for t in owned]

        if self.phase == "attack":
            actions = [0]  # encerrar a fase de ataque
            for src in owned:
                if self.armies[src] <= 1:
                    continue
                for dst in ADJACENCY_INDEX[int(src)]:
                    if self.owner[dst] != player:
                        actions.append(self._encode_move(int(src), int(dst)))
            return actions

        # fortify
        actions = [0]
        if not self.fortified:
            for src in owned:
                if self.armies[src] <= 1:
                    continue
                for dst in ADJACENCY_INDEX[int(src)]:
                    if self.owner[dst] == player:
                        actions.append(self._encode_move(int(src), int(dst)))
        return actions

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def observation(self, player: int = 0) -> np.ndarray:
        dim = self.n_territories * (self.n_players + 1) + 6
        obs = np.zeros(dim, dtype=np.float32)
        stride = self.n_players + 1
        for territory in range(self.n_territories):
            base = territory * stride
            owner = int(self.owner[territory])
            if 0 <= owner < self.n_players:
                obs[base + owner] = 1.0
            obs[base + self.n_players] = min(
                1.0, self.armies[territory] / 20.0)
        offset = self.n_territories * stride
        obs[offset + PHASE_INDEX[self.phase]] = 1.0
        obs[offset + 3] = min(1.0, self.reinforcements_left / 20.0)
        obs[offset + 4] = 1.0 if self.current_player == player else 0.0
        obs[offset + 5] = min(1.0, self.turn / 100.0)
        return obs

    # ------------------------------------------------------------------ #
    # Ações
    # ------------------------------------------------------------------ #
    def apply_action(self, action: int) -> dict | None:
        action = int(action)
        if self.winner is not None:
            raise RuntimeError("a partida já terminou")
        if action not in self.legal_actions():
            raise ValueError(f"ação inválida: {action} "
                             f"(fase {self.phase})")
        kind, src, dst = self.decode(action)
        if kind == "place":
            return self.place_army(int(src))
        if kind == "end":
            self.end_phase()
            return None
        if self.phase == "attack":
            return self.attack(int(src), int(dst))
        return self.fortify(int(src), int(dst))

    def place_army(self, territory: int) -> None:
        if self.phase != "reinforce":
            raise ValueError("só dá para reforçar na fase de reforço")
        if self.owner[territory] != self.current_player:
            raise ValueError("território não pertence ao jogador")
        if self.reinforcements_left <= 0:
            raise ValueError("não há reforços restantes")
        self.armies[territory] += 1
        self.reinforcements_left -= 1
        if self.reinforcements_left == 0:
            self.phase = "attack"
        return None

    def attack(self, src: int, dst: int) -> dict:
        if self.phase != "attack":
            raise ValueError("só dá para atacar na fase de ataque")
        if self.owner[src] != self.current_player:
            raise ValueError("a origem não pertence ao jogador")
        if self.owner[dst] == self.current_player:
            raise ValueError("não dá para atacar território próprio")
        if dst not in ADJACENCY_INDEX[src]:
            raise ValueError("territórios não são vizinhos")
        if self.armies[src] <= 1:
            raise ValueError("é preciso deixar ao menos 1 exército na origem")

        attacker_dice, defender_dice, attacker_losses, defender_losses = (
            self.roll_combat(int(self.armies[src]), int(self.armies[dst])))
        self.armies[src] -= attacker_losses
        self.armies[dst] -= defender_losses

        conquered = False
        moved = 0
        if self.armies[dst] <= 0:
            conquered = True
            # O atacante ocupa com até o número de dados rolados.
            moved = max(1, min(len(attacker_dice), int(self.armies[src]) - 1))
            self.armies[src] -= moved
            self.owner[dst] = self.current_player
            self.armies[dst] = moved

        if np.all(self.owner == self.current_player):
            self.winner = self.current_player

        self.last_battle = {
            "src": src,
            "dst": dst,
            "attacker": self.current_player,
            "attacker_dice": attacker_dice.tolist(),
            "defender_dice": defender_dice.tolist(),
            "attacker_losses": attacker_losses,
            "defender_losses": defender_losses,
            "conquered": conquered,
            "moved": moved,
            "winner": self.winner,
        }
        return self.last_battle

    def roll_combat(self, attacker_armies: int, defender_armies: int):
        """Rola os dados e devolve (dados_atacante, dados_defensor,
        perdas_atacante, perdas_defensor)."""
        n_attacker = min(3, max(1, attacker_armies - 1))
        n_defender = min(2, max(1, defender_armies))
        attacker = np.sort(
            self.rng.integers(1, 7, size=n_attacker))[::-1]
        defender = np.sort(
            self.rng.integers(1, 7, size=n_defender))[::-1]
        comparisons = min(n_attacker, n_defender)
        attacker_losses = int(np.sum(
            attacker[:comparisons] <= defender[:comparisons]))
        defender_losses = comparisons - attacker_losses
        return attacker, defender, attacker_losses, defender_losses

    def fortify(self, src: int, dst: int, count: int = 1) -> None:
        if self.phase != "fortify":
            raise ValueError("só dá para fortificar na fase de fortificação")
        if self.fortified:
            raise ValueError("já fortificou neste turno")
        if self.owner[src] != self.current_player:
            raise ValueError("a origem não pertence ao jogador")
        if self.owner[dst] != self.current_player:
            raise ValueError("o destino não pertence ao jogador")
        if dst not in ADJACENCY_INDEX[src]:
            raise ValueError("territórios não são vizinhos")
        if count < 1 or self.armies[src] - count < 1:
            raise ValueError("é preciso deixar ao menos 1 exército na origem")
        self.armies[src] -= count
        self.armies[dst] += count
        self.fortified = True
        return None

    def end_phase(self) -> None:
        if self.phase == "attack":
            self.phase = "fortify"
            self.fortified = False
            return
        if self.phase == "fortify":
            self._next_player()
            return
        raise ValueError("termine a fase de reforço colocando os reforços")

    def _next_player(self) -> None:
        for _ in range(self.n_players):
            self.current_player = (
                (self.current_player + 1) % self.n_players)
            if np.any(self.owner == self.current_player):
                break
        self.turn += 1
        self.phase = "reinforce"
        self.fortified = False
        self.reinforcements_left = self.reinforcement_count(
            self.current_player)
        self.last_battle = None

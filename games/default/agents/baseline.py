"""Políticas baseline para o adversário controlado pela arena.

Toda política recebe um `PolicyContext` e devolve um índice de ação.
O `ArenaEnv` monta o contexto a cada passo; nenhuma política toca o engine
nem o ambiente diretamente.
"""
from dataclasses import dataclass

import numpy as np

from arena.actions import JUMP, LEFT, NON_PUSH_ACTIONS, PUSH_LEFT, PUSH_RIGHT, RIGHT, STOP


@dataclass
class PolicyContext:
    """O que uma política pode observar em um passo.

    `obs` é a grade semântica de um frame único (5, H, W); `stacked_obs` é a
    pilha de frames `(5 * frame_stack, H, W)` que o DQN treinado vê (None
    quando o ambiente não usa frame stack). `self_character`/`player_character`
    dão o estado exato em pixels (personagem controlado e jogador).
    `rng` é o gerador do ambiente — use-o para toda aleatoriedade da política,
    assim o treino continua reproduzível com seeds.
    """
    obs: np.ndarray
    rng: np.random.Generator
    self_character: object = None
    player_character: object = None
    stacked_obs: np.ndarray = None


class RandomPolicy:
    """Sorteia uniformemente entre as `n_actions` primeiras ações."""
    def __init__(self, n_actions=len(NON_PUSH_ACTIONS)):
        self.n_actions = n_actions

    def act(self, ctx):
        return int(ctx.rng.integers(0, self.n_actions))


class PushRulePolicy:
    """Comportamento histórico do adversário `rule`: aproxima do jogador e,
    quando perto, às vezes tenta empurrar (uma das duas direções)."""
    def act(self, ctx):
        c = ctx.self_character
        player = ctx.player_character
        if c is None or player is None or not c.alive:
            return STOP
        if abs(player.x - c.x) < 35 and ctx.rng.random() < 0.35:
            return int(ctx.rng.choice([PUSH_LEFT, PUSH_RIGHT]))
        if player.x < c.x:
            return LEFT
        if player.x > c.x:
            return RIGHT
        return STOP


class CoinRulePolicy:
    """Vai na direção da moeda visível mais próxima e pula de vez em quando."""
    def act(self, ctx):
        ys, xs = np.where(ctx.obs[3] > 0)
        py, px = np.where(ctx.obs[1] > 0)
        if len(xs) and len(px):
            dx = xs[np.argmin(np.abs(ys - py[0]))] - px[0]
            if abs(dx) <= 1:
                return JUMP if ctx.rng.random() < 0.5 else STOP
            return LEFT if dx < 0 else RIGHT
        return int(ctx.rng.choice(NON_PUSH_ACTIONS))


# O adversário "random" histórico do ambiente não empurrava; o espaço novo
# tem empurrões nas ações 4 e 5, então restringimos o sorteio às 4 primeiras.
POLICIES = {
    "rule": PushRulePolicy,
    "random": lambda: RandomPolicy(n_actions=len(NON_PUSH_ACTIONS)),
    "coins": CoinRulePolicy,
}


def make_policy(spec):
    """Resolve um nome registrado em política; instâncias prontas passam direto."""
    if not isinstance(spec, str):
        return spec
    factory = POLICIES.get(spec)
    if factory is None:
        raise ValueError(
            f"Política de adversário desconhecida: {spec!r} (opções: {', '.join(POLICIES)})"
        )
    return factory()

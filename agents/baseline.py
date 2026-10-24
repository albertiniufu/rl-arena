"""Políticas baseline para o adversário controlado pela arena.

Toda política recebe um `PolicyContext` e devolve um índice de ação.
O `ArenaEnv` monta o contexto a cada passo; nenhuma política toca o engine
nem o ambiente diretamente.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyContext:
    """O que uma política pode observar em um passo.

    `obs` é a grade semântica (5, H, W); `self_character`/`player_character`
    dão o estado exato em pixels (personagem controlado e jogador).
    `rng` é o gerador do ambiente — use-o para toda aleatoriedade da política,
    assim o treino continua reproduzível com seeds.
    """
    obs: np.ndarray
    rng: np.random.Generator
    self_character: object = None
    player_character: object = None


class RandomPolicy:
    """Sorteia uniformemente entre as primeiras `n_actions` ações."""
    def __init__(self, n_actions=12):
        self.n_actions = n_actions

    def act(self, ctx):
        return int(ctx.rng.integers(0, self.n_actions))


class PushRulePolicy:
    """Comportamento histórico do adversário `rule`: aproxima do jogador e,
    quando perto, às vezes tenta empurrar."""
    def act(self, ctx):
        c = ctx.self_character
        player = ctx.player_character
        if c is None or player is None or not c.alive:
            return 0
        if abs(player.x - c.x) < 35 and ctx.rng.random() < 0.35:
            return int(ctx.rng.choice([6, 7, 8, 9, 10, 11]))
        if player.x < c.x:
            return 1
        if player.x > c.x:
            return 2
        return 0


class CoinRulePolicy:
    """Vai na direção da moeda visível mais próxima e pula de vez em quando."""
    def act(self, ctx):
        ys, xs = np.where(ctx.obs[3] > 0)
        py, px = np.where(ctx.obs[1] > 0)
        if len(xs) and len(px):
            dx = xs[np.argmin(np.abs(ys - py[0]))] - px[0]
            if abs(dx) <= 1:
                return 3 if ctx.rng.random() < 0.5 else 0
            return 1 if dx < 0 else 2
        return int(ctx.rng.choice([0, 1, 2, 3]))


# O adversário "random" histórico do ambiente sorteava 0..5 (sem empurrões);
# preservamos esse comportamento registrando RandomPolicy com 6 ações.
POLICIES = {
    "rule": PushRulePolicy,
    "random": lambda: RandomPolicy(n_actions=6),
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

"""Registro das arenas e dos entry points unificados.

Cada arena declara onde vivem os seus `train.py`, `evaluate.py` e `play.py`
e qual é o algoritmo padrão. Os entry points da raiz (`train.py`,
`evaluate.py`, `play.py`) leem `--arena` e delegam para o módulo da arena,
sem importar torch/pygame no despacho.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaSpec:
    """Metadados de uma arena e os seus entry points."""

    name: str
    description: str
    package: str
    train_entry: str | None
    evaluate_entry: str | None
    play_entry: str | None
    default_algorithm: str
    action_space: str  # "discrete" ou "box"
    # Algoritmos aceitos por `--alg`; None = todos compatíveis com a ação
    # (descobertos em `arenas/algs`).
    algorithms: tuple[str, ...] | None = None
    supports_render: bool = False


ARENAS: dict[str, ArenaSpec] = {
    "seek_coin": ArenaSpec(
        name="seek_coin",
        description="Plataforma 2D: coletar moedas e empurrar o adversário.",
        package="arenas.seek_coin",
        train_entry="arenas.seek_coin.train",
        evaluate_entry="arenas.seek_coin.evaluate",
        play_entry="arenas.seek_coin.play",
        default_algorithm="dqn",
        action_space="discrete",
        algorithms=None,
        supports_render=True,
    ),
    "risk": ArenaSpec(
        name="risk",
        description="Conquista de territórios inspirada em War/Risk.",
        package="arenas.risk",
        train_entry="arenas.risk.train",
        evaluate_entry="arenas.risk.evaluate",
        play_entry="arenas.risk.play",
        default_algorithm="dqn",
        action_space="discrete",
        algorithms=("dqn", "doubledqn_per_dueling", "ppo", "a2c"),
    ),
    "economy": ArenaSpec(
        name="economy",
        description="Banco-agente: concessão e preço por GH, RAROC líquido.",
        package="arenas.economy",
        train_entry="arenas.economy.train",
        evaluate_entry="arenas.economy.evaluate",
        play_entry="arenas.economy.play",
        default_algorithm="ppo_continuous",
        action_space="box",
        algorithms=("ppo_continuous", "ddpg", "td3", "sac"),
    ),
}


def available_arenas() -> tuple[str, ...]:
    return tuple(sorted(ARENAS))


def algorithms_for_arena(name: str) -> tuple[str, ...]:
    """Algoritmos aceitos por `--alg` na arena.

    Arenas com lista explícita devolvem-na; `None` significa "todos os
    pacotes compatíveis com o espaço de ação", descobertos na hora. A
    importação de `alg_registry` (e de torch) é adiada até esta chamada.
    """
    spec = get_arena(name)
    if spec.algorithms is not None:
        return tuple(spec.algorithms)
    from .algs.alg_registry import available_algorithms, load_algorithm
    continuous = spec.action_space == "box"
    return tuple(
        alg for alg in available_algorithms()
        if bool(getattr(load_algorithm(alg).Agent, "continuous", False))
        is continuous
    )


def get_arena(name: str) -> ArenaSpec:
    try:
        return ARENAS[name]
    except KeyError as exc:
        raise SystemExit(
            f"Arena desconhecida: {name!r} "
            f"(opções: {', '.join(available_arenas())})") from exc


def load_entry(name: str, kind: str):
    """Módulo de `train`/`evaluate`/`play` da arena `name`."""
    spec = get_arena(name)
    dotted = getattr(spec, f"{kind}_entry", None)
    if not dotted:
        raise SystemExit(f"A arena {name!r} não tem entry point de {kind}.")
    return importlib.import_module(dotted)

"""Catálogo de algoritmos por arena para a Fila de Experimentos.

Os nomes vêm de [`arenas/registry.py`](../registry.py): arenas com lista
explícita (risk, economy) devolvem o mesmo conjunto aceito por `--alg`; o
seek_coin descobre todos os pacotes compatíveis com ações discretas em
`arenas/algs/`.

A documentação é apontada para os arquivos já existentes no repositório,
servidos pelo endpoint `/docs/<caminho>` do servidor da fila:

- aulas HTML em `arenas/seek_coin/docs/<alg>-aula.html`;
- catálogo em `docs/algoritmos.md`;
- README da arena.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..registry import ARENAS, algorithms_for_arena

ROOT = Path(__file__).resolve().parents[2]

CATALOG_URL = "/docs/docs/algoritmos.md#catalogo"
WHO_TO_CHOOSE_URL = "/docs/docs/algoritmos.md#como-escolher"

README_ANCHORS = {
    "seek_coin": "algoritmos",
    "risk": "como-rodar",
    "economy": "como-rodar",
}

# Nomes de arquivo de aula que não coincidem com o `--alg`.
LESSON_STEMS = {"doubledqn_per_dueling": "doubledqn"}


def _lesson_path(algorithm: str) -> str:
    stem = LESSON_STEMS.get(algorithm, algorithm)
    return f"arenas/seek_coin/docs/{stem}-aula.html"


@lru_cache(maxsize=1)
def _capabilities() -> dict[str, dict]:
    """Lê `on_policy`/`continuous`/`multi_agent` das classes `Agent`.

    A importação de torch e dos pacotes acontece na primeira chamada; o
    resultado fica em cache enquanto o processo viver.
    """
    from ..algs.alg_registry import available_algorithms, load_algorithm

    result: dict[str, dict] = {}
    for name in available_algorithms():
        agent_class = load_algorithm(name).Agent
        result[name] = {
            "arch_version": str(getattr(agent_class, "arch_version", "")),
            "on_policy": bool(getattr(agent_class, "on_policy", False)),
            "continuous": bool(getattr(agent_class, "continuous", False)),
            "multi_agent": bool(getattr(agent_class, "multi_agent", False)),
        }
    return result


def _docs(algorithm: str, arena: str) -> list[dict]:
    docs: list[dict] = []
    lesson = _lesson_path(algorithm)
    if (ROOT / lesson).is_file():
        docs.append({"label": "Aula (HTML)", "url": f"/docs/{lesson}"})
    docs.append({"label": "Catálogo", "url": CATALOG_URL})
    package_path = ARENAS[arena].package.replace(".", "/")
    arena_readme = f"/docs/{package_path}/README.md"
    anchor = README_ANCHORS.get(arena)
    docs.append({
        "label": "README da arena",
        "url": f"{arena_readme}#{anchor}" if anchor else arena_readme,
    })
    return docs


def algorithm_catalog() -> dict:
    """Lista os algoritmos de treino de cada arena com links de documentação."""
    capabilities = _capabilities()
    arenas = []
    for name, spec in ARENAS.items():
        algorithms = algorithms_for_arena(name)
        rows = []
        for algorithm in algorithms:
            caps = capabilities.get(algorithm, {})
            rows.append({
                "name": algorithm,
                "package": f"alg_{algorithm}",
                "arch_version": caps.get("arch_version", ""),
                "on_policy": bool(caps.get("on_policy", False)),
                "continuous": bool(caps.get("continuous", False)),
                "multi_agent": bool(caps.get("multi_agent", False)),
                "default": algorithm == spec.default_algorithm,
                "docs": _docs(algorithm, name),
            })
        arenas.append({
            "name": name,
            "description": spec.description,
            "action_space": spec.action_space,
            "default_algorithm": spec.default_algorithm,
            "readme": f"/docs/{spec.package.replace('.', '/')}/README.md",
            "algorithms": rows,
        })
    return {"arenas": arenas, "choosing_guide": WHO_TO_CHOOSE_URL}

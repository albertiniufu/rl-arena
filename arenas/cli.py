"""Despacho dos entry points unificados.

`train.py`, `evaluate.py` e `play.py` na raiz chamam `dispatch(kind)`, que
consome `--arena` e delega para o `main()` da arena. O restante dos
argumentos segue intacto para o parser da arena.
"""
from __future__ import annotations

import argparse
import sys

from .common.yaml_config import arena_from_config
from .registry import available_arenas, get_arena, load_entry

GLOBAL_HELP = """\
Arenas disponíveis: {arenas}

Uso:
  python {kind}.py --arena <arena> [argumentos da arena]
  python {kind}.py --arena <arena> --help

Sem `--arena`, a arena padrão é `seek_coin`.
"""


def dispatch(kind: str, argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arena", default=None,
                        choices=available_arenas())
    parser.add_argument("--config", default=None)
    parsed, rest = parser.parse_known_args(argv)
    explicit_arena = any(
        arg == "--arena" or arg.startswith("--arena=") for arg in argv)
    config_arena = None
    if parsed.arena is None:
        try:
            config_arena = arena_from_config(parsed.config)
        except ValueError as exc:
            raise SystemExit(f"--config inválido: {exc}") from exc
        parsed.arena = config_arena or "seek_coin"

    if (not explicit_arena and config_arena is None and any(
            arg in ("-h", "--help") for arg in rest)):
        print(GLOBAL_HELP.format(
            arenas=", ".join(available_arenas()), kind=kind))
        return

    spec = get_arena(parsed.arena)
    module = load_entry(parsed.arena, kind)
    # Os mains das arenas leem `sys.argv` (compatível com o trainer original).
    # `--config` foi consumido pelo pre-parser; devolve para o parser da arena.
    forwarded = list(rest)
    if parsed.config:
        forwarded += ["--config", parsed.config]
    sys.argv = [f"{spec.name}.{kind}"] + forwarded
    module.main()

#!/usr/bin/env python3
"""Valida o YAML estruturado de uma proposta de experimento.

A lógica vive em `arenas/common/experiment_proposal.py`, compartilhada com a
aba **Proposta de experimentos** da Fila; este script é só o ponto de entrada
usado pela skill `experimenter-helper` e pela documentação.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arenas.common.experiment_proposal import validate  # noqa: E402

__all__ = ["validate"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args(argv)
    path = Path(args.path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"YAML inválido: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("o YAML raiz deve ser um mapa")
    errors = validate(payload)
    if errors:
        print("Proposta inválida:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {path}")
    print(f"id={payload['id']} status={payload['status']} "
          f"arena={payload['setup']['arena']}")


if __name__ == "__main__":
    main()

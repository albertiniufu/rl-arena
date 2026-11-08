"""Configuração de testes do monorepo.

Adiciona a raiz (para `games.*`) e o jogo default (para `arena`, `agents`,
`algs`) ao `sys.path`, independentemente do diretório de invocação do pytest.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "games" / "default"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

"""Configuração de testes do monorepo.

Coloca a raiz do repositório no `sys.path` para que os pacotes `arenas.*`
sejam importáveis independentemente do diretório de invocação do pytest.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

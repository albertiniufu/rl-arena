"""Descoberta e carregamento dinâmico de skins (renderers).

Convenção: cada arquivo `arena/renderer_{sufixo}.py` define uma classe
`ArenaRenderer(cfg)` com `render(engine, elapsed, info=None)` e `close()`.
Criar um arquivo novo é o suficiente — nenhum outro código precisa mudar:
as opções de `--skin` dos entry points são descobertas a partir dos arquivos.
"""
import importlib
from pathlib import Path

_ARENA_DIR = Path(__file__).parent
_PREFIX = "renderer_"


def list_skins():
    """Sufixos de skin disponíveis (o clássico `arena/renderer.py` é o padrão)."""
    skins = []
    for path in _ARENA_DIR.glob(f"{_PREFIX}*.py"):
        skins.append(path.stem[len(_PREFIX):])
    return sorted(skins)


def load_renderer_class(skin=None):
    """Classe `ArenaRenderer` da skin pedida; `skin=None` usa `arena/renderer.py`."""
    module_name = "arena.renderer" if not skin else f"arena.renderer_{skin}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        available = ", ".join(list_skins()) or "(nenhuma)"
        raise ValueError(
            f"Skin de renderer desconhecida: {skin!r}. "
            f"Disponíveis: {available} (padrão: arena/renderer.py)."
        ) from exc
    if not hasattr(module, "ArenaRenderer"):
        raise ValueError(f"{module_name} não define a classe ArenaRenderer.")
    return module.ArenaRenderer

#!/usr/bin/env python3
"""Atalho para `python -m arenas.common.report`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arenas.common.report import main  # noqa: E402

if __name__ == "__main__":
    main()

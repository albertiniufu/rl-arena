"""Entry point unificado de avaliação.

Exemplos:
    python evaluate.py --arena seek_coin --checkpoint checkpoints/best.pt
    python evaluate.py --arena risk --checkpoint checkpoints/risk/best.pt
    python evaluate.py --arena economy --checkpoint checkpoints/economy/best.pt
"""
from arenas.cli import dispatch

if __name__ == "__main__":
    dispatch("evaluate")

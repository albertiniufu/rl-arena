"""Entry point unificado de reprodução.

Exemplos:
    python play.py --arena seek_coin --checkpoint checkpoints/best.pt --render
    python play.py --arena risk --checkpoint checkpoints/risk/best.pt
    python play.py --arena economy --checkpoint checkpoints/economy/best.pt
"""
from arenas.cli import dispatch

if __name__ == "__main__":
    dispatch("play")

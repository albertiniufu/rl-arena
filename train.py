"""Entry point unificado de treino.

Exemplos:
    python train.py --arena seek_coin --alg dqn --episodes 300
    python train.py --arena risk --alg dqn --episodes 500
    python train.py --arena economy --alg ppo_continuous --episodes 200
"""
from arenas.cli import dispatch

if __name__ == "__main__":
    dispatch("train")

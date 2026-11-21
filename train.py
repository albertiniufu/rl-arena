"""Entry point unificado de treino.

Por padrão, um treino local registra o próprio PID na Fila de Experimentos e
espera a vez antes de iniciar. A fila é iniciada automaticamente se preciso.

Exemplos:
    python train.py --arena seek_coin --alg dqn --episodes 300
    python train.py --arena risk --alg dqn --episodes 500
    python train.py --arena economy --alg ppo_continuous --episodes 200
    python train.py --no-queue --episodes 10   # ignora a fila nesta execução
"""
from arenas.common.queue_gate import run_with_queue

if __name__ == "__main__":
    run_with_queue("train")

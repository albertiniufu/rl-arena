"""Pacotes de algoritmos de RL disponíveis para `--alg`.

Cada subdiretório `alg_<nome>` é autocontido (modelo, buffer e agente) e
`alg_registry.py` concentra a descoberta, os checkpoints e o adaptador de
política. O nome usado na CLI é o sufixo do diretório:
`--alg dqn` -> `algs/alg_dqn`.
"""

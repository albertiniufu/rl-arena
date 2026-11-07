"""IPPO: pacote de algoritmo multiagente descoberto por `--alg ippo`."""
from .agent import Agent, IPPOAgent, MultiRolloutBuffer, compute_gae

__all__ = ["Agent", "IPPOAgent", "MultiRolloutBuffer", "compute_gae"]

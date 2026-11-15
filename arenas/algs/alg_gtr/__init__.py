"""GTR: PPO com região de confiança gaussiana (descoberto por `--alg gtr`)."""
from .agent import Agent, GTRAgent, compute_gae, gaussian_penalty

__all__ = ["Agent", "GTRAgent", "compute_gae", "gaussian_penalty"]

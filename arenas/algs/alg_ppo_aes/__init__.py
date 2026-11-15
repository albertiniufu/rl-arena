"""ppo_aes: PPO com Adaptive Entropy Scheduling (descoberto por `--alg ppo_aes`)."""
from .aes import AESScheduler
from .agent import Agent, PPOAESAgent, compute_gae

__all__ = ["AESScheduler", "Agent", "PPOAESAgent", "compute_gae"]

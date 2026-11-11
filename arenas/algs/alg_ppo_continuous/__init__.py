"""PPO com ação contínua (Gaussiana) para arenas com espaço Box."""
from .agent import PPOContinuousAgent as Agent
from .model import GaussianActorCritic

__all__ = ["Agent", "GaussianActorCritic"]

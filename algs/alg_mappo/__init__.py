"""MAPPO: pacote de algoritmo multiagente descoberto por `--alg mappo`."""
from .agent import MAPPOAgent, compute_gae
from .model import CentralizedActorCriticNetwork
from .rollout_buffer import MultiAgentRolloutBuffer

Agent = MAPPOAgent

__all__ = [
    "Agent",
    "MAPPOAgent",
    "CentralizedActorCriticNetwork",
    "MultiAgentRolloutBuffer",
    "compute_gae",
]

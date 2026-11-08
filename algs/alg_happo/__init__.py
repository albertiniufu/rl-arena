"""HAPPO: pacote multiagente heterogêneo descoberto por `--alg happo`."""
from .agent import HAPPOAgent, compute_gae
from .model import ActorNetwork, CentralizedCritic, HAPPONetwork
from .rollout_buffer import HAPPORolloutBuffer

Agent = HAPPOAgent

__all__ = [
    "Agent",
    "HAPPOAgent",
    "ActorNetwork",
    "CentralizedCritic",
    "HAPPONetwork",
    "HAPPORolloutBuffer",
    "compute_gae",
]

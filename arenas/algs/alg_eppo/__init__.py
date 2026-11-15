"""EPPO: PPO com crítico evidencial (pacote descoberto por `--alg eppo`)."""
from .agent import (
    EPPOAgent,
    Agent,
    compute_gae_with_uncertainty,
    evidential_nll,
    evidential_prior_log_prob,
)
from .model import (
    EvidentialActorCriticNetwork,
    evidential_moments,
    evidential_parameters,
)

__all__ = [
    "Agent",
    "EPPOAgent",
    "EvidentialActorCriticNetwork",
    "compute_gae_with_uncertainty",
    "evidential_moments",
    "evidential_nll",
    "evidential_parameters",
    "evidential_prior_log_prob",
]

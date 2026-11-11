"""Carregamento de agentes a partir de checkpoints."""
from __future__ import annotations

from arenas.algs.alg_registry import (
    algorithm_from_checkpoint,
    load_checkpoint,
)

from .agent_factory import action_dim


def agent_from_checkpoint(path, env):
    """Reconstrói o agente de um checkpoint para o ambiente `env`."""
    module, meta = algorithm_from_checkpoint(path)
    agent_class = module.Agent
    obs_shape = tuple(meta["obs_shape"])
    if obs_shape != tuple(env.observation_space.shape):
        raise SystemExit(
            f"Observação do checkpoint {obs_shape} não casa com o ambiente "
            f"{tuple(env.observation_space.shape)}.")
    n_actions = action_dim(env.action_space)
    kwargs = {}
    if getattr(agent_class, "continuous", False):
        kwargs.update(action_low=env.action_space.low,
                      action_high=env.action_space.high)
    agent = agent_class(obs_shape, n_actions, **kwargs)
    load_checkpoint(path, agent, resume=False)
    return agent, meta

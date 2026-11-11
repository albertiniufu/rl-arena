"""Fábrica de agentes a partir do algoritmo e do espaço de ações."""
from __future__ import annotations

from gymnasium import spaces

from arenas.algs.alg_registry import available_algorithms, load_algorithm


def action_dim(action_space) -> int:
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, spaces.Box):
        return int(action_space.shape[0])
    raise TypeError(f"espaço de ação não suportado: {action_space!r}")


def default_algorithm(arena: str) -> str:
    return "ppo_continuous" if arena == "economy" else "dqn"


def build_agent(alg_name: str, obs_shape, action_space, *, lr=None,
                batch_size: int = 64, **kwargs):
    """Instancia o agente do algoritmo, validando o tipo de ação."""
    module = load_algorithm(alg_name)
    agent_class = module.Agent
    continuous = bool(getattr(agent_class, "continuous", False))
    is_box = isinstance(action_space, spaces.Box)

    if is_box and not continuous:
        raise SystemExit(
            f"O algoritmo {alg_name!r} não suporta ações contínuas; "
            "use `--alg ppo_continuous`.")
    if not is_box and continuous:
        raise SystemExit(
            f"O algoritmo {alg_name!r} é só para ações contínuas; "
            f"opções: {', '.join(available_algorithms())}.")

    agent_kwargs = {"batch_size": int(batch_size)}
    if lr is not None:
        agent_kwargs["lr"] = float(lr)
    if not getattr(agent_class, "on_policy", False):
        agent_kwargs["epsilon_min"] = kwargs.pop("epsilon_min", 0.1)
        agent_kwargs["warmup"] = kwargs.pop("warmup", 0)
    agent_kwargs.update(kwargs)

    agent = agent_class(tuple(obs_shape), action_dim(action_space),
                        **agent_kwargs)
    if continuous and hasattr(agent, "set_action_bounds"):
        agent.set_action_bounds(action_space.low, action_space.high)
    return agent

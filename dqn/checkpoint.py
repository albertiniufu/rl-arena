"""Checkpoints do treino, incluindo tudo o que uma retomada reproduzível precisa.

Além de pesos, otimizador, passos e epsilon, o checkpoint guarda o replay
buffer, o estado de todos os RNGs (Python, NumPy, Torch, engine e ambiente)
e um snapshot do ambiente. Com isso, retomar com os mesmos argumentos
(ex.: mesmo --seed) continua o treino bit a bit idêntico.
"""
from pathlib import Path
import random

import numpy as np
import torch

ENV_VERSION = "arena-v1"
ARCH_VERSION = "cnn-dqn-v1"


def capture_rng_state(env=None):
    """Estado de todos os geradores de números aleatórios relevantes."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "engine": env.engine.rng.bit_generator.state if env is not None else None,
        "env_np_random": env.np_random.bit_generator.state if env is not None else None,
    }


def restore_rng_state(state, env=None):
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        # map_location pode ter movido o tensor; set_rng_state exige CPU.
        torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if env is not None and state.get("engine") is not None:
        env.engine.rng.bit_generator.state = state["engine"]
    if env is not None and state.get("env_np_random") is not None:
        env.np_random.bit_generator.state = state["env_np_random"]


def save_checkpoint(path, agent, config=None, extra=None, env=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_version": ENV_VERSION, "arch_version": ARCH_VERSION, "n_actions": agent.n_actions,
        "online": agent.online.state_dict(), "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(), "steps": agent.steps, "epsilon": agent.epsilon,
        "gamma": agent.gamma, "config": vars(config) if config is not None else {},
        "buffer": agent.buffer.state_dict(),
        "rng_state": capture_rng_state(env),
        "env_state": env.get_state() if env is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, agent, resume=False, map_location=None, env=None):
    payload = torch.load(path, map_location=map_location or agent.device, weights_only=False)
    if payload.get("n_actions") != agent.n_actions:
        raise ValueError("Checkpoint action count does not match this environment.")
    if payload.get("arch_version") != ARCH_VERSION:
        raise ValueError("Unsupported checkpoint architecture.")
    agent.online.load_state_dict(payload["online"])
    agent.target.load_state_dict(payload["target"])
    if resume:
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps = int(payload.get("steps", 0))
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        # Buffer e RNGs só são restaurados na retomada; é o que torna a
        # continuação bit a bit idêntica. Checkpoints antigos sem esses campos
        # continuam carregando (sem reproduzibilidade).
        if payload.get("buffer") is not None:
            agent.buffer.load_state_dict(payload["buffer"])
        restore_rng_state(payload.get("rng_state") or {}, env)
        if env is not None and payload.get("env_state"):
            env.set_state(payload["env_state"])
    agent.online.eval() if not resume else agent.online.train()
    return payload

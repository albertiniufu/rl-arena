"""Checkpoints pesos-only: apenas o que uma retomada leve precisa.

O checkpoint guarda pesos das redes online e alvo, otimizador, passos de
treino, epsilon, gamma e o número de episódios consumidos (`episodes`).
Replay buffer, RNGs e estado do ambiente ficam de fora por decisão do autor:
retomar continua o treino, mas não é mais reproduzível bit a bit.
"""
from pathlib import Path

import torch

ENV_VERSION = "arena-v1"
ARCH_VERSION = "cnn-dqn-v2"


def save_checkpoint(path, agent, config=None, extra=None, episodes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_version": ENV_VERSION, "arch_version": ARCH_VERSION, "n_actions": agent.n_actions,
        "online": agent.online.state_dict(), "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(), "steps": agent.steps, "epsilon": agent.epsilon,
        "gamma": agent.gamma, "episodes": episodes,
        "config": vars(config) if config is not None else {},
        "extra": extra or {},
    }
    torch.save(payload, path)


def checkpoint_meta(path, map_location="cpu"):
    """Lê metadados do checkpoint sem carregar pesos no agente.

    Devolve {"config": dict da ArenaConfig gravada, "dueling": bool}.
    O payload guarda `config` via `vars(config)` no save_checkpoint; a
    presença de "v_head.weight" no state_dict online indica cabeça dueling
    (a opção `dueling` só existe no agente a partir da Task 11 — aqui a
    detecção é só leitura do payload).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # Checkpoints antigos podem não ter `config` (ou tê-la como None).
    config = payload.get("config") or {}
    online = payload.get("online") or {}
    return {"config": dict(config), "dueling": "v_head.weight" in online}


def apply_env_config(cfg, meta):
    """Aplica frame_stack/frame_skip do checkpoint ao cfg (defaults se ausentes).

    Quando a chave não existe no payload, mantém o valor atual do cfg — o que
    preserva 1/1 para checkpoints v2 gravados antes do frame stack.
    """
    config = (meta or {}).get("config") or {}
    cfg.frame_stack = int(config.get("frame_stack", cfg.frame_stack))
    cfg.frame_skip = int(config.get("frame_skip", cfg.frame_skip))
    return cfg


def load_checkpoint(path, agent, resume=False, map_location=None):
    payload = torch.load(path, map_location=map_location or agent.device, weights_only=False)
    if payload.get("n_actions") != agent.n_actions:
        raise ValueError("Checkpoint action count does not match this environment.")
    if payload.get("arch_version") != ARCH_VERSION:
        raise ValueError("Unsupported checkpoint architecture.")
    agent.online.load_state_dict(payload["online"])
    agent.target.load_state_dict(payload["target"])
    if resume:
        # A retomada restaura otimizador e contadores. Buffer e RNGs não são
        # persistidos, então o treino continua, mas não bit a bit idêntico.
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps = int(payload.get("steps", 0))
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        agent.gamma = float(payload.get("gamma", agent.gamma))
    agent.online.eval() if not resume else agent.online.train()
    return payload

"""Descoberta dos algoritmos e infraestrutura comum de checkpoint.

Cada algoritmo é um pacote `alg_<nome>` na raiz do repositório com um
`agent.py` que expõe a classe `Agent`. O nome usado na CLI (`--alg`) é o
sufixo do diretório: `alg_dqn` -> `--alg dqn`.

Este módulo concentra o que não é específico de nenhum algoritmo — descoberta
dos pacotes, checkpoint pesos-only e o adaptador de política — para que cada
`alg_*` contenha apenas o seu modelo, o seu replay e o seu loop de
aprendizado, sem duplicar serialização.
"""
import importlib
from pathlib import Path

import torch

ALG_PREFIX = "alg_"
ENV_VERSION = "arena-v1"


def available_algorithms():
    """Nomes dos pacotes `alg_*` que expõem um `agent.py` (sufixo do diretório)."""
    root = Path(__file__).resolve().parent
    return sorted(
        path.name[len(ALG_PREFIX):]
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(ALG_PREFIX)
        and (path / "agent.py").is_file()
    )


def load_algorithm(name):
    """Importa o pacote `alg_<name>`; erro claro se o nome não existir."""
    if name not in available_algorithms():
        raise ValueError(
            f"Algoritmo desconhecido: {name!r} "
            f"(opções: {', '.join(available_algorithms())})")
    return importlib.import_module(f"{ALG_PREFIX}{name}")


def checkpoint_meta(path, map_location="cpu"):
    """Lê metadados do checkpoint sem carregar pesos no agente."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return {
        "algorithm": payload.get("algorithm"),
        "arch_version": payload.get("arch_version"),
        "n_actions": payload.get("n_actions"),
        "obs_shape": tuple(payload.get("obs_shape") or ()),
        "config": dict(payload.get("config") or {}),
    }


def algorithm_from_checkpoint(path, map_location="cpu"):
    """(módulo do algoritmo, metadados) gravados no checkpoint.

    Checkpoints anteriores à separação dos algoritmos não têm o campo
    `algorithm`; nesse caso o erro orienta a retreinar.
    """
    meta = checkpoint_meta(path, map_location=map_location)
    name = meta.get("algorithm")
    if not name:
        raise ValueError(
            "Checkpoint sem o campo 'algorithm' (formato antigo ou de outra "
            "versão); retreine com a versão atual.")
    return load_algorithm(name), meta


def save_checkpoint(path, agent, config=None, extra=None, episodes=None):
    """Grava pesos-only: redes, otimizador, contadores e config do ambiente."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_version": ENV_VERSION,
        "algorithm": agent.algorithm,
        "arch_version": agent.arch_version,
        "n_actions": agent.n_actions,
        "obs_shape": list(agent.obs_shape),
        "online": agent.online.state_dict(),
        "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "steps": agent.steps,
        "epsilon": agent.epsilon,
        "gamma": agent.gamma,
        "episodes": episodes,
        "config": vars(config) if config is not None else {},
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, agent, resume=False, map_location=None):
    """Carrega o checkpoint em `agent`, validando tudo antes de tocar nos pesos.

    A validação antecipada evita o `load_state_dict` parcial: se a arquitetura,
    o algoritmo ou o formato da observação não casarem, o agente fica intacto.
    """
    payload = torch.load(
        path, map_location=map_location or agent.device, weights_only=False)
    saved_algorithm = payload.get("algorithm")
    if saved_algorithm != agent.algorithm:
        raise ValueError(
            f"Checkpoint do algoritmo {saved_algorithm!r} não carrega no "
            f"algoritmo {agent.algorithm!r}.")
    if payload.get("arch_version") != agent.arch_version:
        raise ValueError(
            f"Versão de arquitetura incompatível ({payload.get('arch_version')!r} "
            f"!= {agent.arch_version!r}); retreine com a versão atual.")
    if payload.get("n_actions") != agent.n_actions:
        raise ValueError(
            f"Número de ações do checkpoint ({payload.get('n_actions')!r}) não "
            f"casa com o ambiente ({agent.n_actions!r}).")
    saved_shape = tuple(payload.get("obs_shape") or ())
    if saved_shape and saved_shape != tuple(agent.obs_shape):
        raise ValueError(
            f"Formato da observação incompatível ({saved_shape} != "
            f"{tuple(agent.obs_shape)}); confira frame_stack/frame_skip.")
    agent.online.load_state_dict(payload["online"])
    agent.target.load_state_dict(payload["target"])
    if resume:
        # A retomada restaura otimizador e contadores. Buffer e RNGs não são
        # persistidos, então o treino continua, mas não bit a bit idêntico.
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps = int(payload.get("steps", 0))
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        agent.gamma = float(payload.get("gamma", agent.gamma))
    agent.online.train() if resume else agent.online.eval()
    return payload


def apply_env_config(cfg, meta):
    """Aplica frame_stack/frame_skip do checkpoint ao cfg (defaults se ausentes)."""
    config = (meta or {}).get("config") or {}
    cfg.frame_stack = int(config.get("frame_stack", cfg.frame_stack))
    cfg.frame_skip = int(config.get("frame_skip", cfg.frame_skip))
    return cfg


class CheckpointPolicy:
    """Adapta um agente ao contrato de política `act(ctx) -> int`.

    Age sempre de forma gulosa; permite colocar um agente treinado no lugar
    do adversário (ex.: `--opponent auto` no train.py).
    """
    def __init__(self, agent):
        self.agent = agent

    def act(self, ctx):
        # Com frame stack, a rede espera a pilha `(5 * frame_stack, H, W)`;
        # sem stack, cai no frame único `ctx.obs`.
        state = ctx.stacked_obs if ctx.stacked_obs is not None else ctx.obs
        return int(self.agent.act(state, greedy=True))

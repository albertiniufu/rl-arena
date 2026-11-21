"""Carregamento de configurações YAML para os entry points.

A precedência é sempre:

    CLI > YAML > defaults do argparse

Cada parser chama `apply_yaml_defaults(parser, argv, section)` antes do
`parse_args`. O YAML é mapeado para os mesmos nomes de destino da CLI:

    arena: seek_coin
    alg: ppo
    seed: 7
    level: 3
    episode_seconds: 90
    frame_stack: 4
    frame_skip: 3
    training:
      episodes: 1000000
      max_hours: 3
      checkpoint_dir: checkpoints/ppo-3
      agent_params:
        lr: 0.00068
        batch_size: 256
    evaluation:
      episodes: 50
      seed: 1000
      checkpoint: checkpoints/ppo-3/latest.pt

`--set chave=valor` sobrescreve o YAML; `--set training.agent_params.lr=...`
também funciona. Chaves desconhecidas geram erro com o nome do campo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_META_KEYS = {"arena", "setup", "name", "description", "notes"}
_TOP_KEYS = {
    "alg", "algorithm", "seed", "level", "episode_seconds",
    "frame_stack", "frame_skip", "opponent", "agents", "scenario", "teams",
    "checkpoint_dir", "resume", "episodes", "max_hours", "save_every",
    "save_every_minutes", "tensorboard", "tensorboard_dir", "run_name",
    "stats", "budget_snapshots", "eval_every", "eval_episodes", "eval_seed",
    "eval_opponent", "skin", "render", "learning_rate", "lr",
    "batch_size", "agent_params", "hyperparameters", "params",
    "agent_param", "envs", "overlap", "learn_every", "warmup",
    "epsilon_min", "epsilon_ramp_episodes", "push_coin_transfer",
    "push_coin_reward", "pbrs_coef", "checkpoint", "players", "armies",
    "max_turns", "invalid_action_penalty", "deadline", "log_dir",
    "tensorboard_every", "video_every_minutes", "video_dir", "video_fps",
    "horizon", "n_customers", "n_firms", "rollout_steps", "epochs",
    "policy", "steps", "every", "demo",
}
_TRAIN_KEYS = _TOP_KEYS | {
    "training", "evaluation", "episodes", "max_hours", "save_every",
    "save_every_minutes", "resume", "checkpoint_dir", "log_dir",
    "tensorboard", "tensorboard_dir", "run_name", "stats",
    "budget_snapshots", "envs", "overlap", "learn_every", "warmup",
}
_EVAL_KEYS = _TOP_KEYS | {"evaluation", "checkpoint", "episodes", "seed",
                          "render", "level", "episode_seconds", "opponent",
                          "skin", "policy", "steps", "every", "demo"}
_PLAY_KEYS = _EVAL_KEYS | {"player", "human_agent", "record"}


class ConfigError(ValueError):
    """Erro de configuração YAML com contexto do campo."""


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"não foi possível ler {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido em {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"o YAML raiz de {path} deve ser um mapa")
    return payload


def _deep_set(data: dict, dotted: str, value) -> None:
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        raise ConfigError("--set sem chave")
    current = data
    for part in parts[:-1]:
        node = current.get(part)
        if node is None:
            node = {}
            current[part] = node
        if not isinstance(node, dict):
            raise ConfigError(
                f"--set {dotted}: {part!r} não é um mapa")
        current = node
    current[parts[-1]] = value


def _format_agent_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _agent_params(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(item) for item in values]
    if not isinstance(values, dict):
        raise ConfigError("agent_params deve ser um mapa ou lista")
    return [f"{key}={_format_agent_value(value)}"
            for key, value in values.items()]


def _normalize_mapping(mapping: dict, section: str) -> dict:
    result = {}
    for raw_key, value in mapping.items():
        key = str(raw_key).replace("-", "_")
        if key in _META_KEYS or key in ("training", "evaluation", "config"):
            continue
        if key in ("algorithm", "alg"):
            result["alg"] = value
        elif key in ("agent_params", "hyperparameters", "params"):
            result["agent_param"] = _agent_params(value)
        elif key == "agent_param":
            result["agent_param"] = _agent_params(value)
        else:
            result[key] = value
    return result


def config_defaults(data: dict, section: str) -> dict:
    """Mapeia o YAML para os destinos do argparse da seção."""
    if section not in ("train", "evaluate", "play"):
        raise ValueError(f"seção desconhecida: {section!r}")
    allowed = ({"train": _TRAIN_KEYS, "evaluate": _EVAL_KEYS,
                "play": _PLAY_KEYS}[section]
               | _META_KEYS | {"training", "evaluation"})
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"chaves desconhecidas no YAML para {section}: {unknown}")

    result = _normalize_mapping(data, section)
    # Seções: treino usa `training`; avaliar/play usam `evaluation`.
    block_name = "training" if section == "train" else "evaluation"
    block = data.get(block_name) or {}
    if not isinstance(block, dict):
        raise ConfigError(f"{block_name} deve ser um mapa")
    block_unknown = sorted(set(block) - allowed)
    if block_unknown:
        raise ConfigError(
            f"chaves desconhecidas em {block_name}: {block_unknown}")
    result.update(_normalize_mapping(block, section))
    return result


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="arquivo YAML de configuração da execução")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="sobrescreve um campo do YAML (pode repetir)")


def apply_yaml_defaults(parser: argparse.ArgumentParser, argv,
                        section: str) -> None:
    """Aplica o YAML como default do parser; CLI continua com precedência."""
    args = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--set", action="append", default=[])
    known, _ = pre.parse_known_args(args)
    if not known.config and not known.set:
        return
    data = load_yaml(known.config) if known.config else {}
    for item in known.set:
        if "=" not in item:
            raise ConfigError(f"--set inválido: {item!r} (use KEY=VALUE)")
        key, value_text = item.split("=", 1)
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError:
            value = value_text
        _deep_set(data, key.strip(), value)
    parser.set_defaults(**config_defaults(data, section))


def arena_from_config(path: str | Path | None) -> str | None:
    """Lê apenas o campo `arena` do YAML (para o dispatch da raiz)."""
    if not path:
        return None
    data = load_yaml(path)
    arena = data.get("arena")
    return None if arena is None else str(arena)

#!/usr/bin/env python3
"""Submete um novo experimento à Fila de Experimentos.

Este é o caminho recomendado para “rodar um experimento novo”: em vez de
executar o treino diretamente no terminal, ele entra na fila persistente.
Se o servidor da fila não estiver no ar, ele é iniciado automaticamente.

Exemplos:

    python scripts/experiment.py --name ppo-30min \
        -- python train.py --config configs/seek_coin/ppo-level3.yaml \
        --set training.max_hours=0.5

    python scripts/experiment.py --name bo-piloto --max-runtime 3600 \
        -- python scripts/bo_campaign.py --algorithms ppo,eppo ...

    python scripts/experiment.py --yaml configs/queue-example.yaml

    python scripts/experiment.py --wait --name smoke -- python train.py ...

Também aceita `train.py`/`scripts/foo.py` sem prefixo `python`; o wrapper
usa o interpretador atual.
"""
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arenas.common.job_queue_client import (  # noqa: E402
    DEFAULT_URL,
    ensure_server,
    enqueue_job,
    enqueue_yaml,
    health,
    wait_for_job,
)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_tags(value: str | None) -> dict:
    tags = {}
    for item in _split_csv(value):
        if "=" not in item:
            raise SystemExit(f"tag inválida: {item!r} (use chave=valor)")
        key, val = item.split("=", 1)
        tags[key.strip()] = val.strip()
    return tags


def _build_command(raw: list[str]) -> list[str]:
    command = list(raw)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("informe um comando depois de `--` ou use --yaml")
    first = command[0]
    if "python" in Path(first).name:
        command[0] = sys.executable
    elif first.endswith(".py"):
        command.insert(0, sys.executable)
    return command


def _default_name(command: list[str]) -> str:
    for part in command:
        if part.endswith(".py"):
            stem = Path(part).stem
            return f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    return f"experiment-{time.strftime('%Y%m%d-%H%M%S')}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None)
    parser.add_argument("--yaml", default=None,
                        help="YAML de fila de jobs (em vez de um comando)")
    parser.add_argument("--experiment", default=None,
                        help="YAML de experimento (EXP-...yaml) para enfileirar")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--max-runtime", type=float, default=None,
                        help="tempo máximo em segundos")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--depends-on", default="",
                        help="IDs separados por vírgula")
    parser.add_argument("--artifacts", default="",
                        help="caminhos separados por vírgula")
    parser.add_argument("--tags", default="",
                        help="ex.: arena=seek_coin,alg=ppo")
    parser.add_argument("--url", default=None,
                        help=f"URL da fila (padrão: {DEFAULT_URL})")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", default="experiments/queue/queue.json")
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--no-start", action="store_true",
                        help="não inicia o servidor automaticamente")
    parser.add_argument("--allow-arbitrary", action="store_true",
                        help="repassa ao servidor ao iniciar (perigoso)")
    parser.add_argument("--wait", action="store_true",
                        help="espera o(s) job(s) terminar(em)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="timeout do --wait em segundos")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    url = args.url or f"http://{args.host}:{args.port}"
    parsed_url = urlparse(url)
    host = parsed_url.hostname or args.host
    port = parsed_url.port or args.port
    if not health(url):
        if args.no_start:
            raise SystemExit(
                f"fila não está no ar em {url}; rode sem --no-start ou "
                "inicie scripts/job_queue_server.py")
        ensure_server(
            url, state=args.state, logs_dir=args.logs_dir,
            host=host, port=port, max_parallel=args.max_parallel,
            allow_arbitrary=args.allow_arbitrary)
        print(f"fila iniciada em {url}", flush=True)

    if args.experiment:
        exp_path = Path(args.experiment)
        if not exp_path.is_absolute():
            exp_path = (ROOT / exp_path).resolve()
        payload = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise SystemExit("YAML de experimento deve ser um mapa")
        execution = payload.get("execution") or {}
        raw_command = execution.get("command") or payload.get("command")
        if raw_command is None:
            commands = payload.get("commands") or []
            if commands:
                raw_command = commands[0]
        if raw_command is None:
            raise SystemExit(
                "YAML de experimento sem execution.command ou commands[0]")
        if isinstance(raw_command, str):
            raw_command = shlex.split(raw_command)
        command = _build_command(list(raw_command))
        exp_id = str(payload.get("id") or "experiment")
        name = args.name or exp_id
        cwd = execution.get("cwd") or args.cwd
        max_runtime = (execution.get("max_runtime")
                       if execution.get("max_runtime") is not None
                       else args.max_runtime)
        priority = (execution.get("priority")
                    if execution.get("priority") is not None
                    else args.priority)
        tags = _parse_tags(args.tags)
        tags.setdefault("experiment", exp_id)
        arena = (payload.get("setup") or {}).get("arena")
        if arena:
            tags.setdefault("arena", str(arena))
        job = enqueue_job(
            url, name=name, command=command, cwd=cwd,
            priority=priority,
            depends_on=_split_csv(args.depends_on),
            max_runtime=max_runtime,
            artifacts=_split_csv(args.artifacts),
            tags=tags,
            experiment_yaml=str(exp_path))
        print(f"experimento adicionado: {exp_id}")
        print(f"job: {job['id']}")
        print(f"fila: {url}/")
        if args.wait:
            final = wait_for_job(url, job["id"], timeout=args.timeout)
            print(f"status final: {final['status']} "
                  f"(exit={final.get('exit_code')})")
            if final["status"] != "done":
                raise SystemExit(1)
        return

    if args.yaml:
        path = Path(args.yaml)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        result = enqueue_yaml(url, path)
        ids = result.get("ids", [])
        print(f"{len(ids)} job(s) carregado(s): {', '.join(ids)}")
        if args.wait:
            for job_id in ids:
                job = wait_for_job(url, job_id, timeout=args.timeout)
                print(f"{job_id}: {job['status']}")
        return

    command = _build_command(args.command)
    name = args.name or _default_name(command)
    job = enqueue_job(
        url, name=name, command=command, cwd=args.cwd,
        priority=args.priority,
        depends_on=_split_csv(args.depends_on),
        max_runtime=args.max_runtime,
        artifacts=_split_csv(args.artifacts),
        tags=_parse_tags(args.tags))
    print(f"job adicionado: {job['id']}")
    print(f"nome: {job['name']}")
    print(f"fila: {url}/")
    if args.wait:
        final = wait_for_job(url, job["id"], timeout=args.timeout)
        print(f"status final: {final['status']} "
              f"(exit={final.get('exit_code')})")
        if final["status"] != "done":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

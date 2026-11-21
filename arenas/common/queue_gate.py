"""Portão de permissão para treinos locais via Fila de Experimentos.

Quando `train.py` é executado diretamente, ele registra o próprio PID como um
job local na fila e **espera a vez** antes de chamar o treino de verdade.
Jobs iniciados pelo servidor da fila recebem `RL_ARENA_QUEUE_BYPASS=1` e não
passam por esse portão. Como o servidor não captura o stdout de processos
externos, o portão duplica a saída do job local para o log da Fila.

Use `--no-queue` (ou a env `RL_ARENA_QUEUE_BYPASS=1`) para rodar direto.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from arenas.common.job_queue_client import (
    DEFAULT_URL,
    ensure_server,
    finish_job,
    register_local_job,
    wait_for_turn,
)

ROOT = Path(__file__).resolve().parents[2]


class _Tee:
    """Duplica escrita para o terminal e para o log da Fila."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
            except (OSError, ValueError):
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self._streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")


def _local_log_path(job: dict) -> Path:
    """Resolve o log_path devolvido pela fila (absoluto ou relativo à raiz)."""
    path = Path(str(job.get("log_path") or ""))
    if not path.is_absolute():
        path = ROOT / path
    return path


@contextmanager
def _tee_to_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_out, handle)
    sys.stderr = _Tee(original_err, handle)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = original_out, original_err
        handle.close()


def _strip_no_queue(argv: list[str]) -> tuple[list[str], bool]:
    cleaned = []
    disabled = False
    for arg in argv:
        if arg == "--no-queue":
            disabled = True
        else:
            cleaned.append(arg)
    return cleaned, disabled


def _arg_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _job_name(kind: str, argv: list[str]) -> str:
    arena = _arg_value(argv, "--arena") or "seek_coin"
    alg = _arg_value(argv, "--alg") or "default"
    return f"local-{kind}-{arena}-{alg}-{time.strftime('%Y%m%d-%H%M%S')}"


def run_with_queue(kind: str) -> None:
    argv, disabled = _strip_no_queue(sys.argv[1:])
    bypass = (disabled or os.environ.get("RL_ARENA_QUEUE_BYPASS") == "1"
              or any(arg in ("-h", "--help") for arg in argv))
    if kind != "train" or bypass:
        from arenas.cli import dispatch
        dispatch(kind, argv)
        return

    url = os.environ.get("RL_ARENA_QUEUE_URL") or DEFAULT_URL
    state = os.environ.get("RL_ARENA_QUEUE_STATE") or str(
        ROOT / "experiments" / "queue" / "queue.json")
    parsed = urlparse(url)
    started = ensure_server(
        url, state=state, host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 8765)
    command = [sys.executable, str(ROOT / "train.py"), *argv]
    name = _job_name(kind, argv)
    arena = _arg_value(argv, "--arena") or "seek_coin"
    alg = _arg_value(argv, "--alg") or "default"
    job = register_local_job(
        url, name=name, command=command, cwd=str(Path.cwd()),
        pid=os.getpid(), tags={"arena": arena, "alg": alg})
    job_id = job["id"]
    log_path = _local_log_path(job)

    # Jobs locais são processos externos: o servidor não consegue capturar
    # stdout/stderr deles. O próprio processo duplica a saída para o log.
    with _tee_to_log(log_path):
        print(f"fila {'iniciada' if started else 'já estava no ar'} em {url}",
              flush=True)
        print(f"aguardando permissão da fila: {url}/")
        print(f"job local: {job_id} ({name})")
        print(f"log: {log_path}", flush=True)

        last_wait_print = 0.0

        def _wait(job_snapshot):
            nonlocal last_wait_print
            status = job_snapshot.get("status", "queued")
            now = time.monotonic()
            if status == "queued" and now - last_wait_print >= 10.0:
                print("  ainda na fila...", flush=True)
                last_wait_print = now

        job = wait_for_turn(url, job_id, on_wait=_wait)
        if job.get("status") != "running":
            raise SystemExit(
                f"job local não foi autorizado (status={job.get('status')}): "
                f"{job.get('error') or ''}")

        print("permissão concedida; iniciando treino local", flush=True)
        exit_code = 0
        error = ""
        try:
            from arenas.cli import dispatch
            dispatch(kind, argv)
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 0
            raise
        except KeyboardInterrupt:
            exit_code = 130
            error = "interrompido"
            raise
        except Exception as exc:  # noqa: BLE001
            exit_code = 1
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                finish_job(url, job_id, exit_code=exit_code, error=error)
            except Exception as exc:  # noqa: BLE001
                print(f"aviso: não foi possível finalizar o job na fila: {exc}",
                      file=sys.stderr)

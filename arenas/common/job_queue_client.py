"""Cliente HTTP e inicialização automática do servidor da fila."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "http://127.0.0.1:8765"


def _request(url: str, method: str = "GET", payload=None,
             timeout: float = 5.0):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers,
                                     method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(url: str = DEFAULT_URL, timeout: float = 1.0) -> bool:
    try:
        payload = _request(f"{url.rstrip('/')}/api/health", timeout=timeout)
        return bool(payload.get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def ensure_server(url: str = DEFAULT_URL, *, state: str | Path,
                  logs_dir: str | Path | None = None,
                  host: str = "127.0.0.1", port: int = 8765,
                  max_parallel: int = 1, wait_seconds: float = 15.0,
                  allow_arbitrary: bool = False) -> bool:
    """Garante que o servidor está no ar; devolve True se precisou iniciar."""
    if health(url):
        return False
    state = Path(state)
    state.parent.mkdir(parents=True, exist_ok=True)
    log_path = state.parent / "server.log"
    command = [
        sys.executable, str(ROOT / "scripts" / "job_queue_server.py"),
        "--state", str(state), "--host", host, "--port", str(port),
        "--max-parallel", str(max_parallel),
    ]
    if logs_dir:
        command += ["--logs-dir", str(logs_dir)]
    if allow_arbitrary:
        command.append("--allow-arbitrary")
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    (state.parent / "server.pid").write_text(str(process.pid),
                                             encoding="utf-8")
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if health(url):
            log.close()
            return True
        if process.poll() is not None:
            break
        time.sleep(0.2)
    log.close()
    if process.poll() is None:
        process.terminate()
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
    raise RuntimeError(
        f"não foi possível iniciar a fila em {url}.\n{tail}")


def enqueue_job(url: str = DEFAULT_URL, *, name: str, command, cwd: str = ".",
                priority: int = 0, depends_on=None, max_runtime=None,
                artifacts=None, tags=None, experiment_yaml=None) -> dict:
    return _request(
        f"{url.rstrip('/')}/api/jobs", "POST",
        {
            "name": name,
            "command": command,
            "cwd": cwd,
            "priority": int(priority),
            "depends_on": list(depends_on or []),
            "max_runtime": max_runtime,
            "artifacts": list(artifacts or []),
            "tags": dict(tags or {}),
            "experiment_yaml": experiment_yaml,
        })


def register_local_job(url: str = DEFAULT_URL, *, name: str, command,
                       cwd: str = ".", pid: int, priority: int = 0,
                       depends_on=None, max_runtime=None, artifacts=None,
                       tags=None, experiment_yaml=None) -> dict:
    """Registra um processo local que espera permissão da fila."""
    return _request(
        f"{url.rstrip('/')}/api/jobs/local", "POST",
        {
            "name": name,
            "command": command,
            "cwd": cwd,
            "pid": int(pid),
            "priority": int(priority),
            "depends_on": list(depends_on or []),
            "max_runtime": max_runtime,
            "artifacts": list(artifacts or []),
            "tags": dict(tags or {}),
            "experiment_yaml": experiment_yaml,
        })


def finish_job(url: str, job_id: str, exit_code: int = 0,
               error: str = "") -> dict:
    return _request(
        f"{url.rstrip('/')}/api/jobs/{job_id}/finish", "POST",
        {"exit_code": int(exit_code), "error": error})


def wait_for_turn(url: str, job_id: str, poll: float = 1.0,
                  on_wait=None) -> dict:
    """Espera o job sair de `queued`; retorna running ou final."""
    while True:
        job = get_job(url, job_id)
        if job is None:
            raise KeyError(f"job não encontrado: {job_id}")
        status = job.get("status")
        if status == "running" or status in ("done", "failed", "canceled"):
            return job
        if on_wait is not None:
            on_wait(job)
        time.sleep(max(0.05, poll))


def enqueue_experiment(url: str = DEFAULT_URL, path: str | Path = "",
                       **kwargs) -> dict:
    payload = {"path": str(path)}
    payload.update(kwargs)
    return _request(f"{url.rstrip('/')}/api/experiments", "POST", payload)


def enqueue_yaml(url: str = DEFAULT_URL, path: str | Path = "") -> dict:
    return _request(f"{url.rstrip('/')}/api/load", "POST",
                    {"path": str(path)})


def get_job(url: str, job_id: str) -> dict | None:
    payload = _request(f"{url.rstrip('/')}/api/state")
    for job in payload.get("jobs", []):
        if job.get("id") == job_id:
            return job
    return None


def wait_for_job(url: str, job_id: str, timeout: float | None = None,
                 poll: float = 1.0) -> dict:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        job = get_job(url, job_id)
        if job is None:
            raise KeyError(f"job não encontrado: {job_id}")
        if job.get("status") in ("done", "failed", "canceled"):
            return job
        if deadline is not None and time.monotonic() >= deadline:
            return job
        time.sleep(max(0.05, poll))

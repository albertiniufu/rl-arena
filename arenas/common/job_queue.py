"""Fila de Experimentos local com persistência e agendamento simples.

A fila é usada pela interface web (`scripts/job_queue_server.py`) para
administrar treinos, campanhas de BO e relatórios.

- Jobs são argv (sem shell), com `cwd`, prioridade, dependências e orçamento.
- O estado é persistido em JSON e sobrevive ao reinício do servidor.
- O scheduler respeita `max_parallel` e nunca inicia um job cujas dependências
  não terminaram com sucesso.
- Por padrão só comandos Python do próprio repositório são aceitos; isso evita
  que a API web vire execução arbitrária de shell.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import quote
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"queued", "running", "paused"}
FINAL = {"done", "failed", "canceled"}
JOB_TO_EXPERIMENT_STATUS = {
    "queued": "planned",
    "running": "running",
    "done": "done",
    "failed": "aborted",
    "canceled": "aborted",
}


@dataclass
class Job:
    id: str
    name: str
    command: list[str]
    cwd: str
    status: str = "queued"
    mode: str = "managed"  # managed = fila inicia; local = processo externo
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)
    max_runtime: float | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    log_path: str = ""
    artifacts: list[str] = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    attempts: int = 0
    error: str = ""
    experiment_yaml: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items()
                      if key in allowed})


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_command(command) -> list[str]:
    if isinstance(command, str):
        command = shlex.split(command)
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("command deve ser uma lista ou string não vazia")
    return [str(part) for part in command]


def _validate_command(command: list[str], cwd: Path,
                      allow_arbitrary: bool = False) -> None:
    if allow_arbitrary:
        return
    first = Path(command[0]).name
    if "python" not in first:
        raise ValueError(
            "por segurança, o primeiro argumento deve ser um executável "
            "Python (python/python3) ou use allow_arbitrary=True")
    if len(command) < 2:
        raise ValueError("comando Python sem script/argumento")
    second = command[1]
    if second in ("-c", "-m"):
        return
    target = (cwd / second).resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(
            f"script fora do repositório: {target}") from exc


class JobQueue:
    """Fila persistente com scheduler em thread."""

    def __init__(self, state_path: str | Path,
                 logs_dir: str | Path | None = None,
                 max_parallel: int = 1,
                 poll_interval: float = 0.5,
                 allow_arbitrary: bool = False):
        self.state_path = Path(state_path)
        self.queue_dir = self.state_path.parent
        self.logs_dir = Path(logs_dir) if logs_dir else self.queue_dir / "logs"
        self.max_parallel = max(1, int(max_parallel))
        self.poll_interval = max(0.02, float(poll_interval))
        self.allow_arbitrary = bool(allow_arbitrary)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._logs: dict[str, object] = {}
        self._start_times: dict[str, float] = {}
        self._paused = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Persistência

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._paused = bool(payload.get("paused", False))
        for item in payload.get("jobs", []):
            job = Job.from_dict(item)
            if job.status == "running" and job.pid:
                if not self._pid_alive(job.pid):
                    job.status = "failed"
                    job.error = "processo interrompido (PID não existe)"
                    job.finished_at = job.finished_at or _now()
            self._jobs[job.id] = job

    def _save(self) -> None:
        payload = {
            "paused": self._paused,
            "jobs": [job.to_dict() for job in self._sorted_jobs()],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.state_path)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def _resolve_under_root(value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(
                f"caminho fora do repositório: {path}") from exc
        return path

    def _experiment_path(self, job: Job) -> Path | None:
        if not job.experiment_yaml:
            return None
        path = self._resolve_under_root(job.experiment_yaml)
        if not path.exists():
            return None
        return path

    def experiment_data(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        path = self._experiment_path(job)
        if path is None:
            return None
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"YAML de experimento inválido: {path}")
        return payload

    def experiment_yaml_text(self, job_id: str) -> str | None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        path = self._experiment_path(job)
        return None if path is None else path.read_text(encoding="utf-8")

    @staticmethod
    def _artifact_kind(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".mp4", ".webm", ".mov", ".avi"}:
            return "video"
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
            return "image"
        if suffix in {".md", ".html", ".htm", ".txt", ".yaml", ".yml",
                      ".csv", ".json", ".jsonl", ".log"}:
            return "document"
        if path.is_dir():
            return "directory"
        return "file"

    def experiment_artifacts(self, job_id: str) -> list[dict]:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        data = self.experiment_data(job_id) or {}
        entries: list[tuple[str, str]] = []

        def add(label: str, value):
            if value is None:
                return
            if isinstance(value, list):
                for item in value:
                    add(label, item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    add(f"{label}.{key}" if label else str(key), item)
            elif isinstance(value, str):
                entries.append((label, value))

        for item in job.artifacts:
            add("job", item)
        artifacts = data.get("artifacts") or {}
        if isinstance(artifacts, dict):
            for key, value in artifacts.items():
                if key == "experiment_dir":
                    add("experiment_dir", value)
                else:
                    add(str(key), value)
        else:
            add("experiment", artifacts)

        result = []
        seen = set()
        for label, raw in entries:
            try:
                path = self._resolve_under_root(raw)
            except ValueError:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            exists = path.exists()
            try:
                rel = str(path.relative_to(ROOT))
            except ValueError:
                rel = str(path)
            result.append({
                "label": label,
                "path": rel,
                "name": path.name,
                "kind": self._artifact_kind(path),
                "exists": exists,
                "url": f"/api/files?path={quote(rel)}",
            })
        return result

    # ------------------------------------------------------------------
    # API de consulta

    def _sorted_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(),
                      key=lambda job: (-job.priority, job.created_at, job.id))

    @staticmethod
    def _effective_experiment_status(job: Job) -> str:
        return JOB_TO_EXPERIMENT_STATUS.get(job.status, job.status)

    def _experiment_summary(self, job: Job) -> dict | None:
        try:
            data = self.experiment_data(job.id)
        except (OSError, ValueError, KeyError):
            data = None
        if not data:
            return None
        declared = data.get("status")
        effective = self._effective_experiment_status(job)
        return {
            "id": data.get("id"),
            "title": data.get("title") or data.get("question"),
            "status": declared,
            "declared_status": declared,
            "effective_status": effective,
            "status_mismatch": declared != effective,
            "can_update_status": declared != effective,
            "status_source": "job",
            "path": job.experiment_yaml,
        }

    @staticmethod
    def _replace_top_level_status(text: str, status: str) -> str:
        """Troca só a linha `status:` do topo, preservando comentários/ordem."""
        pattern = re.compile(
            r"(?m)^(?P<key>status:)(?P<space>[ \t]*)"
            r"(?P<value>[^\s#]+)(?P<tail>[ \t]*(?:#.*)?)$")
        if pattern.search(text):
            return pattern.sub(
                lambda match: (f"{match.group('key')}{match.group('space')}"
                               f"{status}{match.group('tail')}"),
                text, count=1)
        return f"status: {status}\n" + text

    def sync_experiment_status(self, job_id: str) -> dict:
        """Grava no YAML o status efetivo do job, de forma atômica.

        A troca é feita na linha `status:` do topo para preservar comentários
        e a ordem das demais chaves. O arquivo só é alterado por ação
        explícita; a UI mostra o status efetivo mesmo sem sincronizar.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            path = self._experiment_path(job)
            if path is None:
                raise ValueError("job não tem YAML de experimento")
            text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict):
                raise ValueError(f"YAML de experimento inválido: {path}")
            declared = data.get("status")
            effective = self._effective_experiment_status(job)
            if declared == effective:
                return {
                    "changed": False,
                    "declared_status": declared,
                    "effective_status": effective,
                    "path": str(path.relative_to(ROOT)),
                }
            updated = self._replace_top_level_status(text, effective)
            check = yaml.safe_load(updated) or {}
            if not isinstance(check, dict) or check.get("status") != effective:
                raise ValueError("não foi possível atualizar o status do YAML")
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(updated, encoding="utf-8")
            try:
                tmp.chmod(path.stat().st_mode)
            except OSError:
                pass
            os.replace(tmp, path)
            return {
                "changed": True,
                "declared_status": declared,
                "effective_status": effective,
                "path": str(path.relative_to(ROOT)),
            }

    def _job_view(self, job: Job) -> dict:
        data = job.to_dict()
        data["experiment"] = self._experiment_summary(job)
        if job.status == "running" and job.started_at:
            try:
                started = time.mktime(time.strptime(
                    job.started_at, "%Y-%m-%d %H:%M:%S"))
                data["elapsed_seconds"] = max(0.0, time.time() - started)
            except ValueError:
                data["elapsed_seconds"] = 0.0
        return data

    def snapshot(self) -> dict:
        with self._lock:
            jobs = [self._job_view(job) for job in self._sorted_jobs()]
            return {
                "paused": self._paused,
                "max_parallel": self.max_parallel,
                "running": sum(1 for job in self._jobs.values()
                               if job.status == "running"),
                "queued": sum(1 for job in self._jobs.values()
                              if job.status == "queued"),
                "jobs": jobs,
            }

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def tail(self, job_id: str, lines: int = 200) -> str:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        path = Path(job.log_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max(1, int(lines)):])

    # ------------------------------------------------------------------
    # API de mutação

    def add(self, name: str, command, cwd: str = ".",
            priority: int = 0, depends_on: list[str] | None = None,
            max_runtime: float | None = None, artifacts: list[str] | None = None,
            tags: dict | None = None,
            experiment_yaml: str | None = None) -> Job:
        command = _parse_command(command)
        cwd_path = (ROOT / cwd).resolve() if not Path(cwd).is_absolute() else Path(cwd)
        _validate_command(command, cwd_path, self.allow_arbitrary)
        if not cwd_path.exists():
            raise ValueError(f"cwd não existe: {cwd_path}")
        exp_path = None
        if experiment_yaml:
            exp_path = self._resolve_under_root(experiment_yaml)
            if not exp_path.exists():
                raise ValueError(f"YAML de experimento não existe: {exp_path}")
        with self._lock:
            job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            log_path = self.logs_dir / f"{job_id}.log"
            job = Job(
                id=job_id, name=name, command=command, cwd=str(cwd_path),
                priority=int(priority),
                depends_on=list(depends_on or []),
                max_runtime=(None if max_runtime is None else float(max_runtime)),
                created_at=_now(), log_path=str(log_path),
                artifacts=list(artifacts or []), tags=dict(tags or {}),
                experiment_yaml=(None if exp_path is None
                                 else str(exp_path.relative_to(ROOT))),
            )
            for dependency in job.depends_on:
                if dependency not in self._jobs:
                    raise ValueError(f"dependência inexistente: {dependency}")
            self._jobs[job.id] = job
            self._save()
            return job

    def add_local(self, name: str, command, cwd: str, pid: int,
                  priority: int = 0, depends_on: list[str] | None = None,
                  max_runtime: float | None = None,
                  artifacts: list[str] | None = None,
                  tags: dict | None = None,
                  experiment_yaml: str | None = None) -> Job:
        """Registra um processo local que espera permissão para rodar."""
        command = _parse_command(command)
        cwd_path = Path(cwd).resolve()
        _validate_command(command, cwd_path, self.allow_arbitrary)
        if not cwd_path.exists():
            raise ValueError(f"cwd não existe: {cwd_path}")
        exp_path = None
        if experiment_yaml:
            exp_path = self._resolve_under_root(experiment_yaml)
            if not exp_path.exists():
                raise ValueError(f"YAML de experimento não existe: {exp_path}")
        with self._lock:
            job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            job = Job(
                id=job_id, name=name, command=command, cwd=str(cwd_path),
                mode="local", priority=int(priority),
                depends_on=list(depends_on or []),
                max_runtime=(None if max_runtime is None else float(max_runtime)),
                created_at=_now(), pid=int(pid),
                log_path=str(self.logs_dir / f"{job_id}.log"),
                artifacts=list(artifacts or []), tags=dict(tags or {}),
                experiment_yaml=(None if exp_path is None
                                 else str(exp_path.relative_to(ROOT))),
            )
            for dependency in job.depends_on:
                if dependency not in self._jobs:
                    raise ValueError(f"dependência inexistente: {dependency}")
            self._jobs[job.id] = job
            self._save()
            return job

    def finish(self, job_id: str, exit_code: int = 0, error: str = "") -> Job:
        """Marca um job local como concluído (chamado pelo processo externo)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in FINAL:
                return job
            job.exit_code = int(exit_code)
            job.status = "done" if int(exit_code) == 0 else "failed"
            job.finished_at = _now()
            if error:
                job.error = error
            self._close_log(job_id)
            self._processes.pop(job_id, None)
            self._save()
            return job

    def add_experiment(self, path: str | Path, *, name: str | None = None,
                       cwd: str | None = None, priority: int | None = None,
                       max_runtime: float | None = None,
                       tags: dict | None = None) -> Job:
        """Enfileira a execução descrita em um YAML de experimento."""
        exp_path = self._resolve_under_root(path)
        if not exp_path.exists():
            raise ValueError(f"YAML de experimento não existe: {exp_path}")
        data = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML de experimento deve ser um mapa")
        execution = data.get("execution") or {}
        command = execution.get("command") or data.get("command")
        if command is None:
            commands = data.get("commands") or []
            command = commands[0] if commands else None
        if command is None:
            raise ValueError(
                "experimento sem execution.command nem commands[0]")
        if isinstance(command, str):
            command = shlex.split(command)
        exp_id = str(data.get("id") or exp_path.stem)
        merged_tags = {"experiment": exp_id}
        arena = (data.get("setup") or {}).get("arena")
        if arena:
            merged_tags["arena"] = str(arena)
        merged_tags.update(tags or {})
        return self.add(
            name=name or exp_id, command=command,
            cwd=cwd or execution.get("cwd") or ".",
            priority=(priority if priority is not None
                      else int(execution.get("priority") or 0)),
            depends_on=execution.get("depends_on") or [],
            max_runtime=(max_runtime if max_runtime is not None
                         else execution.get("max_runtime")),
            tags=merged_tags, experiment_yaml=str(exp_path))

    def add_many(self, specs: list[dict]) -> list[Job]:
        """Adiciona jobs de uma lista YAML, ligando `depends_on` por nome."""
        name_to_id: dict[str, str] = {}
        created: list[Job] = []
        with self._lock:
            for spec in specs:
                if not isinstance(spec, dict):
                    raise ValueError("cada job deve ser um mapa")
                name = str(spec.get("name") or spec.get("id") or "job")
                deps = spec.get("depends_on") or []
                dep_ids = []
                for dep in deps:
                    if dep in name_to_id:
                        dep_ids.append(name_to_id[dep])
                    elif dep in self._jobs:
                        dep_ids.append(dep)
                    else:
                        raise ValueError(
                            f"dependência {dep!r} não encontrada para {name!r}")
                job = self.add(
                    name=name, command=spec.get("command"), cwd=spec.get("cwd", "."),
                    priority=spec.get("priority", 0), depends_on=dep_ids,
                    max_runtime=spec.get("max_runtime"),
                    artifacts=spec.get("artifacts"), tags=spec.get("tags"),
                    experiment_yaml=spec.get("experiment_yaml"),
                )
                name_to_id[name] = job.id
                created.append(job)
        return created

    def load_yaml(self, path: str | Path) -> list[Job]:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        specs = payload.get("jobs", payload) if isinstance(payload, dict) else payload
        if not isinstance(specs, list):
            raise ValueError("YAML deve conter uma lista em `jobs`")
        return self.add_many(specs)

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in FINAL:
                return job
            process = self._processes.get(job_id)
            if job.mode == "local" and job.status == "running" and job.pid:
                try:
                    os.kill(int(job.pid), signal.SIGTERM)
                except OSError:
                    pass
            elif job.status == "running" and process is not None:
                try:
                    os.killpg(job.pid, signal.SIGTERM)
                except (OSError, TypeError):
                    try:
                        process.terminate()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(job.pid, signal.SIGKILL)
                    except (OSError, TypeError):
                        process.kill()
            job.status = "canceled"
            job.finished_at = _now()
            job.exit_code = None
            self._close_log(job_id)
            self._processes.pop(job_id, None)
            self._save()
            return job

    def retry(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in FINAL:
                raise ValueError("só é possível retentar jobs finalizados")
            job.status = "queued"
            job.started_at = job.finished_at = None
            job.pid = job.exit_code = None
            job.error = ""
            job.attempts += 1
            self._save()
            return job

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._save()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._save()

    def move(self, job_id: str, delta: int) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.priority = int(job.priority) + int(delta)
            self._save()
            return job

    # ------------------------------------------------------------------
    # Scheduler

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="job-queue",
                                        daemon=True)
        self._thread.start()

    def stop(self, terminate: bool = False) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if terminate:
            for job in list(self._jobs.values()):
                if job.status == "running":
                    self.cancel(job.id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - scheduler não pode morrer
                print(f"[job-queue] erro no scheduler: {exc}", file=sys.stderr)
            self._stop.wait(self.poll_interval)

    def _tick(self) -> None:
        with self._lock:
            changed = self._reap()
            if not self._paused:
                running = sum(1 for job in self._jobs.values()
                              if job.status == "running")
                for job in self._sorted_jobs():
                    if running >= self.max_parallel:
                        break
                    if job.status != "queued":
                        continue
                    if not all(self._jobs.get(dep) and
                               self._jobs[dep].status == "done"
                               for dep in job.depends_on):
                        continue
                    if job.mode == "local":
                        job.status = "running"
                        job.started_at = _now()
                        self._start_times[job.id] = time.monotonic()
                    else:
                        self._launch(job)
                    running += 1
                    changed = True
            if changed:
                self._save()

    def _reap(self) -> bool:
        changed = False
        for job_id, job in list(self._jobs.items()):
            if job.status == "queued" and job.mode == "local":
                if job.pid and not self._pid_alive(job.pid):
                    job.status = "canceled"
                    job.error = "processo local encerrado antes da permissão"
                    job.finished_at = _now()
                    changed = True
                continue
            if job.status != "running":
                continue
            if job.mode == "local":
                if not job.pid or not self._pid_alive(job.pid):
                    job.status = "failed"
                    job.error = "processo local encerrado sem finish"
                    job.finished_at = _now()
                    changed = True
                elif (job.max_runtime
                      and time.monotonic() - self._start_monotonic(job_id)
                      > job.max_runtime):
                    self.cancel(job_id)
                    changed = True
                continue
            process = self._processes.get(job_id)
            if process is not None:
                code = process.poll()
                if code is None:
                    if (job.max_runtime
                            and job.started_at
                            and time.monotonic() - self._start_monotonic(job_id)
                            > job.max_runtime):
                        self.cancel(job_id)
                        changed = True
                    continue
                job.exit_code = int(code)
                job.status = "done" if code == 0 else "failed"
                job.finished_at = _now()
                self._close_log(job_id)
                self._processes.pop(job_id, None)
                if code != 0:
                    job.error = f"exit code {code}"
                changed = True
            else:
                # Processo adotado após reinício do servidor.
                if job.pid and not self._pid_alive(job.pid):
                    job.status = "failed"
                    job.error = "processo interrompido"
                    job.finished_at = _now()
                    changed = True
        return changed

    def _start_monotonic(self, job_id: str) -> float:
        return self._start_times.get(job_id, time.monotonic())

    def _launch(self, job: Job) -> None:
        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"\n=== {job.name} | {job.id} | {_now()} ===\n")
        handle.flush()
        env = os.environ.copy()
        env["RL_ARENA_QUEUE_BYPASS"] = "1"
        env["RL_ARENA_QUEUE_JOB_ID"] = job.id
        try:
            process = subprocess.Popen(
                job.command, cwd=job.cwd, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True, env=env)
        except OSError as exc:
            handle.write(f"[erro ao iniciar] {exc}\n")
            handle.close()
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = _now()
            return
        job.status = "running"
        job.started_at = _now()
        job.pid = process.pid
        self._processes[job.id] = process
        self._logs[job.id] = handle
        self._start_times[job.id] = time.monotonic()

    def _close_log(self, job_id: str) -> None:
        handle = self._logs.pop(job_id, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        self._start_times.pop(job_id, None)

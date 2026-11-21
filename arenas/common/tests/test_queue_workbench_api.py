"""Testes HTTP das abas de análise e proposta da Fila de Experimentos."""
import csv
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from arenas.common import experiment_analysis as analysis
from arenas.common import experiment_proposal as proposal
from arenas.common import job_queue as job_queue_module
from arenas.common.job_queue import JobQueue

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "job_queue_server_workbench_test",
    ROOT / "scripts" / "job_queue_server.py")
server_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server_module
SPEC.loader.exec_module(server_module)


def call(base, path, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get_raw(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return (response.status,
                    response.read().decode("utf-8", "replace"),
                    response.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        return (exc.code, exc.read().decode("utf-8", "replace"),
                exc.headers.get("Content-Type", ""))


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(job_queue_module, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(proposal, "ROOT", tmp_path)
    queue = JobQueue(tmp_path / "queue.json", poll_interval=0.02)
    queue.pause()
    queue.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
    server.queue = queue
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url, queue, tmp_path
    server.shutdown()
    server.server_close()
    queue.stop(terminate=True)


def make_experiment(root: Path, name: str = "EXP-20260101-api") -> Path:
    rows = []
    for seed in (7, 42, 123):
        rows.append({"algorithm": "dqn", "train_seed": seed,
                     "reward_mean": 10 + seed})
        rows.append({"algorithm": "ppo", "train_seed": seed,
                     "reward_mean": 5 + seed})
    data_dir = root / "experiments" / name
    data_dir.mkdir(parents=True)
    with (data_dir / "results.csv").open("w", encoding="utf-8",
                                         newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["algorithm", "train_seed", "reward_mean"])
        writer.writeheader()
        writer.writerows(rows)
    yaml_path = root / "docs" / "experimentos" / f"{name}.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(f"""id: {name}
status: planned
question: "API?"
setup:
  arena: seek_coin
execution:
  command: ["python", "-c", "pass"]
  cwd: "."
artifacts:
  experiment_dir: experiments/{name}
  results_csv: experiments/{name}/results.csv
""", encoding="utf-8")
    return yaml_path


def fields() -> dict:
    return {
        "title": "Proposta via API",
        "author": "teste",
        "question": "A proposta funciona pela API?",
        "hypothesis": "Sim.",
        "falsification": "Erros de validação.",
        "arena": "seek_coin",
        "candidates": "ppo, eppo",
        "baseline": "ppo",
        "metric_name": "reward",
        "metric_direction": "maximize",
        "metric_definition": "reward guloso",
        "seeds_train": "7, 42, 123",
        "eval_episodes": "50",
        "env_steps_per_run": "4M",
        "wall_hours_per_run": "1",
        "n_runs": "3",
        "tuning_method": "none",
        "checkpoint": "latest",
        "truncation": "sem bootstrap",
        "support": "sim",
        "inconclusive": "talvez",
        "abandon": "não",
        "command": "python -c pass",
        "cwd": ".",
        "max_runtime": "600",
    }


def test_algorithms_and_docs_endpoints(live):
    url, _, _ = live
    status, catalog = call(url, "/api/algorithms")
    assert status == 200
    assert [arena["name"] for arena in catalog["arenas"]] == [
        "seek_coin", "risk", "economy"]
    assert catalog["arenas"][0]["algorithms"]

    status, html, content_type = get_raw(url, "/docs/docs/algoritmos.md")
    assert status == 200
    assert "text/html" in content_type
    assert "Catálogo" in html and "<table>" in html

    status, lesson, content_type = get_raw(
        url, "/docs/arenas/seek_coin/docs/dqn-aula.html")
    assert status == 200
    assert "text/html" in content_type
    assert "<html" in lesson.lower()

    status, _, _ = get_raw(url, "/docs/%2e%2e/README.md")
    assert status == 403


def test_results_and_analysis_endpoints(live):
    url, queue, root = live
    exp = make_experiment(root)
    job = queue.add_experiment(exp)
    status, payload = call(url, f"/api/jobs/{job.id}/results")
    assert status == 200
    assert len(payload["sources"]) == 1
    artifact_paths = {item["path"] for item in payload["artifacts"]}
    assert "experiments/EXP-20260101-api/results.csv" in artifact_paths
    source = payload["sources"][0]
    assert source["algorithms"] == ["dqn", "ppo"]
    assert source["suggested_metric"] == "reward_mean"

    status, report = call(
        url, f"/api/jobs/{job.id}/analysis", "POST",
        {"input": source["path"], "metric": "reward_mean",
         "baseline": "dqn", "paired": True, "bootstrap": 100})
    assert status == 200
    assert report["analysis_quality"] == "primary"
    assert report["metric"] == "reward_mean"
    assert len(report["summaries"]) == 2
    assert "dqn" in report["html"]

    status, error = call(
        url, f"/api/jobs/{job.id}/analysis", "POST",
        {"input": "../../fora.csv", "bootstrap": 100})
    assert status == 400
    assert "fora do repositório" in error["error"]


def test_proposal_preview_validate_save_flow(live):
    url, _, root = live
    status, preview = call(url, "/api/proposals/preview", "POST",
                           {"fields": fields()})
    assert status == 200
    assert preview["errors"] == []
    assert preview["proposal"]["id"].startswith("EXP-")
    assert "## 1. Pergunta" in preview["markdown"]

    bad = fields()
    bad["seeds_train"] = "7"
    status, preview = call(url, "/api/proposals/preview", "POST",
                           {"fields": bad})
    assert status == 200
    assert any("3+ seeds" in error for error in preview["errors"])

    status, checked = call(url, "/api/proposals/validate", "POST",
                           {"yaml": preview["yaml"]})
    assert status == 200
    assert any("3+ seeds" in error for error in checked["errors"])

    status, saved = call(url, "/api/proposals/save", "POST",
                         {"fields": fields()})
    assert status == 201
    assert (root / saved["yaml_path"]).exists()
    assert (root / saved["markdown_path"]).exists()

    status, conflict = call(url, "/api/proposals/save", "POST",
                            {"fields": fields()})
    assert status == 409
    assert "já existe" in conflict["error"]


def test_proposal_enqueue_endpoint(live):
    url, queue, root = live
    status, payload = call(url, "/api/proposals/enqueue", "POST",
                           {"fields": fields()})
    assert status == 201
    job = payload["job"]
    assert job["experiment_yaml"].endswith(".yaml")
    assert queue.get(job["id"]) is not None
    assert (root / payload["proposal"]["yaml_path"]).exists()


@pytest.fixture
def read_only_live(tmp_path, monkeypatch):
    monkeypatch.setattr(job_queue_module, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(proposal, "ROOT", tmp_path)
    state = tmp_path / "queue.json"
    writer = JobQueue(state, poll_interval=0.5)
    writer.pause()
    writer.add("historico", ["python", "-c", "pass"])
    queue = JobQueue(state, poll_interval=0.5, read_only=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
    server.queue = queue
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url, queue, tmp_path
    server.shutdown()
    server.server_close()


def test_read_only_server_blocks_mutations(read_only_live):
    url, queue, _root = read_only_live
    status, state = call(url, "/api/state")
    assert status == 200
    assert state["read_only"] is True
    assert [job["name"] for job in state["jobs"]] == ["historico"]

    status, body = call(url, "/api/queue/pause", "POST", {})
    assert status == 403
    assert "somente leitura" in body["error"]

    status, body = call(url, "/api/jobs", "POST", {
        "name": "novo", "command": ["python", "-c", "pass"]})
    assert status == 403
    assert "somente leitura" in body["error"]

    status, body = call(
        url, "/api/jobs/" + state["jobs"][0]["id"] + "/cancel", "POST", {})
    assert status == 403
    assert "somente leitura" in body["error"]

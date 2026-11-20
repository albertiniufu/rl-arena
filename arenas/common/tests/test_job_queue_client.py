"""Testes do cliente HTTP da fila de experimentos."""
import importlib.util
import sys
import threading
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from arenas.common import job_queue_client as client
from arenas.common.job_queue import JobQueue

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "job_queue_server_test", ROOT / "scripts" / "job_queue_server.py")
server_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server_module
SPEC.loader.exec_module(server_module)


@pytest.fixture
def live_queue(tmp_path):
    queue = JobQueue(tmp_path / "queue.json", poll_interval=0.02)
    queue.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
    server.queue = queue
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url, queue
    server.shutdown()
    server.server_close()
    queue.stop(terminate=True)


def test_client_enqueues_and_waits(live_queue):
    url, queue = live_queue
    assert client.health(url)
    job = client.enqueue_job(
        url, name="client", command=[sys.executable, "-c", "print('x')"])
    final = client.wait_for_job(url, job["id"], timeout=5)
    assert final["status"] == "done"
    assert final["exit_code"] == 0
    assert "x" in queue.tail(job["id"])


def test_ensure_server_detects_running(live_queue, tmp_path):
    url, _ = live_queue
    assert client.ensure_server(
        url, state=tmp_path / "other.json") is False


def test_client_rejects_shell(live_queue):
    url, _ = live_queue
    with pytest.raises(urllib.error.HTTPError) as exc:
        client.enqueue_job(
            url, name="bad", command=["bash", "-c", "true"])
    assert exc.value.code == 400


def test_client_local_job_roundtrip(live_queue):
    import os
    url, queue = live_queue
    queue.pause()
    job = client.register_local_job(
        url, name="local", command=[sys.executable, "-c", "pass"],
        cwd=str(ROOT), pid=os.getpid())
    assert job["mode"] == "local"
    queue.resume()
    state = client.wait_for_turn(url, job["id"], poll=0.02)
    assert state["status"] == "running"
    finished = client.finish_job(url, job["id"], exit_code=0)
    assert finished["status"] == "done"


def test_enqueue_job_includes_experiment_yaml(monkeypatch):
    captured = {}

    def fake_request(url, method="GET", payload=None, timeout=5.0):
        captured.update(payload or {})
        return {"id": "job-x"}

    monkeypatch.setattr(client, "_request", fake_request)
    client.enqueue_job(
        "http://queue", name="exp", command=["python", "train.py"],
        experiment_yaml="docs/experimentos/EXP-teste.yaml")
    assert captured["experiment_yaml"] == "docs/experimentos/EXP-teste.yaml"

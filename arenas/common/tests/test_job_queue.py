"""Testes da fila local de processos."""
import sys
import time

import pytest

from arenas.common.job_queue import JobQueue


def wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def queue(tmp_path):
    q = JobQueue(tmp_path / "queue.json", max_parallel=1, poll_interval=0.02)
    q.start()
    yield q
    q.stop(terminate=True)


def test_rejects_shell_command(queue):
    with pytest.raises(ValueError):
        queue.add("shell", ["bash", "-c", "true"])
    with pytest.raises(ValueError):
        queue.add("python-sem-script", [sys.executable])


def test_runs_job_and_captures_log(queue):
    job = queue.add("ok", [sys.executable, "-c", "print('ok')"])
    assert wait_until(lambda: queue.get(job.id).status == "done")
    assert queue.get(job.id).exit_code == 0
    assert "ok" in queue.tail(job.id)


def test_dependency_waits_for_success(queue, tmp_path):
    marker = tmp_path / "marker.txt"
    first = queue.add("first", [
        sys.executable, "-c",
        "import pathlib,time; time.sleep(0.2); "
        f"pathlib.Path({str(marker)!r}).write_text('a')"])
    second = queue.add("second", [
        sys.executable, "-c",
        "import pathlib; "
        f"assert pathlib.Path({str(marker)!r}).exists()"],
        depends_on=[first.id])
    assert wait_until(lambda: queue.get(first.id).status == "done")
    assert wait_until(lambda: queue.get(second.id).status == "done")
    assert queue.get(second.id).exit_code == 0


def test_failure_and_retry(queue):
    job = queue.add("fail", [sys.executable, "-c", "raise SystemExit(3)"])
    assert wait_until(lambda: queue.get(job.id).status == "failed")
    queue.pause()
    retried = queue.retry(job.id)
    assert retried.status == "queued"
    assert retried.attempts == 1


def test_pause_and_cancel_queued(queue):
    queue.pause()
    job = queue.add("sleep", [sys.executable, "-c", "import time; time.sleep(1)"])
    time.sleep(0.1)
    assert queue.get(job.id).status == "queued"
    queue.cancel(job.id)
    assert queue.get(job.id).status == "canceled"
    queue.resume()


def test_cancel_running_job(queue):
    job = queue.add("sleep", [sys.executable, "-c", "import time; time.sleep(10)"])
    assert wait_until(lambda: queue.get(job.id).status == "running")
    queue.cancel(job.id)
    assert queue.get(job.id).status == "canceled"


def test_persistence(tmp_path):
    path = tmp_path / "queue.json"
    q = JobQueue(path, poll_interval=0.02)
    job = q.add("persist", [sys.executable, "-c", "print('x')"])
    q2 = JobQueue(path, poll_interval=0.02)
    assert q2.get(job.id) is not None
    assert q2.get(job.id).name == "persist"


def test_priority_reorders_snapshot(queue):
    queue.pause()
    low = queue.add("low", [sys.executable, "-c", "pass"], priority=0)
    high = queue.add("high", [sys.executable, "-c", "pass"], priority=5)
    names = [job["name"] for job in queue.snapshot()["jobs"]]
    assert names.index("high") < names.index("low")
    queue.move(high.id, -10)
    names = [job["name"] for job in queue.snapshot()["jobs"]]
    assert names.index("low") < names.index("high")


def test_load_yaml_links_dependencies_by_name(queue, tmp_path):
    queue.pause()
    path = tmp_path / "jobs.yaml"
    path.write_text("""
jobs:
  - name: first
    command: python -c "print(1)"
    cwd: .
  - name: second
    command: python -c "print(2)"
    cwd: .
    depends_on: [first]
""", encoding="utf-8")
    jobs = queue.load_yaml(path)
    assert [job.name for job in jobs] == ["first", "second"]
    assert jobs[1].depends_on == [jobs[0].id]
    assert queue.get(jobs[0].id).status == "queued"


def test_local_job_waits_for_permission_and_finishes(queue, tmp_path):
    queue.pause()
    job = queue.add_local(
        "local", [sys.executable, "-c", "pass"], cwd=str(tmp_path),
        pid=__import__("os").getpid())
    assert job.mode == "local"
    assert job.status == "queued"
    queue.resume()
    assert wait_until(lambda: queue.get(job.id).status == "running")
    queue.finish(job.id, exit_code=0)
    assert queue.get(job.id).status == "done"


def test_local_job_dead_pid_is_canceled(queue, tmp_path):
    queue.pause()
    job = queue.add_local(
        "dead", [sys.executable, "-c", "pass"], cwd=str(tmp_path),
        pid=999999)
    queue.resume()
    assert wait_until(lambda: queue.get(job.id).status == "canceled")


def test_add_experiment_yaml_and_artifacts(queue, tmp_path, monkeypatch):
    from arenas.common import job_queue as jq

    monkeypatch.setattr(jq, "ROOT", tmp_path)
    exp_dir = tmp_path / "docs" / "experimentos"
    exp_dir.mkdir(parents=True)
    artifact = tmp_path / "experiments" / "EXP-test" / "report.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("relatório", encoding="utf-8")
    exp = exp_dir / "EXP-20260101-teste.yaml"
    exp.write_text("""
id: EXP-20260101-teste
status: planned
title: Teste
question: "Funciona?"
setup:
  arena: seek_coin
execution:
  command: ["python", "-c", "pass"]
  cwd: "."
artifacts:
  reports:
    - experiments/EXP-test/report.md
""", encoding="utf-8")
    queue.pause()
    job = queue.add_experiment(exp)
    assert job.experiment_yaml == "docs/experimentos/EXP-20260101-teste.yaml"
    snapshot = queue.snapshot()
    item = next(j for j in snapshot["jobs"] if j["id"] == job.id)
    assert item["experiment"]["id"] == "EXP-20260101-teste"
    artifacts = queue.experiment_artifacts(job.id)
    assert len(artifacts) == 1
    assert artifacts[0]["path"] == "experiments/EXP-test/report.md"
    assert artifacts[0]["kind"] == "document"
    assert "EXP-20260101-teste" in queue.experiment_yaml_text(job.id)


def test_experiment_effective_status_and_sync(queue, tmp_path, monkeypatch):
    from arenas.common import job_queue as jq
    import yaml

    monkeypatch.setattr(jq, "ROOT", tmp_path)
    exp_dir = tmp_path / "docs" / "experimentos"
    exp_dir.mkdir(parents=True)
    exp = exp_dir / "EXP-20260101-status.yaml"
    exp.write_text("""# comentário preservado
id: EXP-20260101-status
status: planned  # status do experimento
title: Teste de status
question: "Funciona?"
setup:
  arena: seek_coin
execution:
  command: ["python", "-c", "pass"]
  cwd: "."
""", encoding="utf-8")
    queue.pause()
    job = queue.add_experiment(exp)
    item = next(j for j in queue.snapshot()["jobs"] if j["id"] == job.id)
    assert item["experiment"]["declared_status"] == "planned"
    assert item["experiment"]["effective_status"] == "planned"
    assert item["experiment"]["status_mismatch"] is False

    queue.finish(job.id, exit_code=0)
    item = next(j for j in queue.snapshot()["jobs"] if j["id"] == job.id)
    assert item["experiment"]["effective_status"] == "done"
    assert item["experiment"]["status_mismatch"] is True
    assert item["experiment"]["can_update_status"] is True

    result = queue.sync_experiment_status(job.id)
    assert result["changed"] is True
    text = exp.read_text(encoding="utf-8")
    assert "# comentário preservado" in text
    assert "status do experimento" in text
    assert yaml.safe_load(text)["status"] == "done"
    item = next(j for j in queue.snapshot()["jobs"] if j["id"] == job.id)
    assert item["experiment"]["status_mismatch"] is False
    assert queue.sync_experiment_status(job.id)["changed"] is False


def test_experiment_canceled_maps_to_aborted(queue, tmp_path, monkeypatch):
    from arenas.common import job_queue as jq

    monkeypatch.setattr(jq, "ROOT", tmp_path)
    exp_dir = tmp_path / "docs" / "experimentos"
    exp_dir.mkdir(parents=True)
    exp = exp_dir / "EXP-20260102-cancel.yaml"
    exp.write_text("""id: EXP-20260102-cancel
status: planned
setup:
  arena: risk
execution:
  command: ["python", "-c", "pass"]
""", encoding="utf-8")
    queue.pause()
    job = queue.add_experiment(exp)
    queue.cancel(job.id)
    result = queue.sync_experiment_status(job.id)
    assert result["effective_status"] == "aborted"
    assert "status: aborted" in exp.read_text(encoding="utf-8")


def test_read_only_queue_never_writes_state(tmp_path):
    state = tmp_path / "queue.json"
    writable = JobQueue(state, poll_interval=0.02)
    writable.pause()
    before = state.read_text(encoding="utf-8")

    viewer = JobQueue(state, poll_interval=0.02, read_only=True)
    assert viewer.read_only is True
    snapshot = viewer.snapshot()
    assert snapshot["read_only"] is True
    assert snapshot["paused"] is True

    with pytest.raises(RuntimeError):
        viewer.add("x", [sys.executable, "-c", "print(1)"])
    with pytest.raises(RuntimeError):
        viewer.pause()
    with pytest.raises(RuntimeError):
        viewer.resume()
    with pytest.raises(RuntimeError):
        viewer.cancel("job-inexistente")
    with pytest.raises(RuntimeError):
        viewer.retry("job-inexistente")
    with pytest.raises(RuntimeError):
        viewer.move("job-inexistente", 1)
    with pytest.raises(RuntimeError):
        viewer.finish("job-inexistente")

    viewer.start()
    assert viewer._thread is None
    assert state.read_text(encoding="utf-8") == before


def test_read_only_queue_shows_running_job(tmp_path):
    state = tmp_path / "queue.json"
    owner = JobQueue(state, poll_interval=0.02)
    job = owner.add("sleep", [
        sys.executable, "-c", "import time; time.sleep(30)"])
    owner.start()
    try:
        assert wait_until(
            lambda: owner.get(job.id).status == "running")
        assert wait_until(
            lambda: '"running"' in state.read_text(encoding="utf-8"))

        viewer = JobQueue(state, poll_interval=0.02, read_only=True)
        assert viewer.get(job.id).status == "running"
        assert viewer.snapshot()["running"] == 1
        assert viewer.snapshot()["jobs"][0]["pid"] == job.pid
    finally:
        owner.stop(terminate=True)


def test_read_only_mirror_follows_state_file_changes(tmp_path):
    state = tmp_path / "queue.json"
    owner = JobQueue(state, poll_interval=0.02)
    owner.pause()
    owner.add("primeiro", [sys.executable, "-c", "print(1)"])

    viewer = JobQueue(state, poll_interval=0.5, read_only=True)
    assert [job["name"] for job in viewer.snapshot()["jobs"]] == ["primeiro"]

    segundo = owner.add("segundo", [sys.executable, "-c", "print(2)"])
    # O espelho relê o arquivo e passa a mostrar o job novo, sem gravar nada.
    names = {job["name"] for job in viewer.snapshot()["jobs"]}
    assert names == {"primeiro", "segundo"}
    assert viewer.get(segundo.id) is not None
    assert viewer.snapshot()["read_only"] is True
    assert state.read_text(encoding="utf-8") == state.read_text(encoding="utf-8")

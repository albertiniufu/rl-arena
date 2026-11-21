"""Testes do portão da fila e da captura de log de jobs locais."""
import io
import sys
from pathlib import Path

from arenas.common import queue_gate


def test_tee_writes_to_all_streams():
    first, second = io.StringIO(), io.StringIO()
    tee = queue_gate._Tee(first, second)
    assert tee.write("linha") == 5
    tee.flush()
    assert first.getvalue() == "linha"
    assert second.getvalue() == "linha"


def test_local_log_path_resolves_relative_to_root():
    job = {"log_path": "experiments/queue/logs/job-1.log"}
    assert queue_gate._local_log_path(job) == (
        queue_gate.ROOT / "experiments/queue/logs/job-1.log")
    absolute = Path("/tmp/job-1.log")
    assert queue_gate._local_log_path({"log_path": str(absolute)}) == absolute


def test_tee_to_log_captures_stdout_and_stderr(tmp_path, monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    log_path = tmp_path / "logs" / "job.log"
    with queue_gate._tee_to_log(log_path):
        print("saída padrão")
        print("erro padrão", file=sys.stderr)
    assert "saída padrão" in out.getvalue()
    assert "erro padrão" in err.getvalue()
    text = log_path.read_text(encoding="utf-8")
    assert "saída padrão" in text
    assert "erro padrão" in text

"""Testes do sumarizador usado nas comparações controladas."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_summarizer():
    path = ROOT / "scripts" / "summarize_all.py"
    spec = importlib.util.spec_from_file_location("summarize_all_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoints_include_latest_for_wall_clock_runs():
    """Treinos por tempo de parede avaliam o checkpoint final."""
    module = load_summarizer()
    assert ("latest", "final") in module.CHECKPOINTS


def test_protocol_text_reports_wall_clock_budget():
    module = load_summarizer()
    text = module.protocol_text(3)
    assert "3 h de parede" in text
    assert "20 episódios" in text
    assert "100 episódios" not in text


def test_write_text_uses_wall_clock_protocol(tmp_path):
    module = load_summarizer()
    module.OUT = tmp_path
    module.write_text({}, {}, [], [], hours=3)
    text = (tmp_path / "summary-all.txt").read_text()
    assert "3 h de parede" in text
    assert "100 episódios" not in text


def test_write_markdown_uses_wall_clock_protocol(tmp_path):
    module = load_summarizer()
    module.OUT = tmp_path
    module.write_markdown({}, {}, [], [], hours=3)
    md = (tmp_path / "comparacao-all.md").read_text()
    assert "3 h de parede" in md
    assert "100 episódios" not in md
    assert "3h" in md

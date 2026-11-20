"""Testes da análise estatística da Fila de Experimentos."""
import csv
from pathlib import Path

import pytest

from arenas.common import experiment_analysis as analysis


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    return tmp_path


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def paired_rows() -> list[dict]:
    rows = []
    for seed in (7, 42, 123, 314, 999):
        rows.append({"algorithm": "dqn", "train_seed": seed,
                     "reward_mean": 10 + seed, "coins_mean": 5 + seed})
        rows.append({"algorithm": "ppo", "train_seed": seed,
                     "reward_mean": 5 + seed, "coins_mean": 3 + seed})
    return rows


def test_inspect_detects_columns_and_suggestions(root):
    path = write_csv(root / "results.csv", paired_rows(),
                     ["algorithm", "train_seed", "reward_mean", "coins_mean"])
    info = analysis.inspect_results(path)
    assert info["rows"] == 10
    assert info["algorithms"] == ["dqn", "ppo"]
    assert info["seed_fields"] == ["train_seed"]
    assert info["metrics"] == ["reward_mean", "coins_mean"]
    assert info["suggested_metric"] == "reward_mean"
    assert info["suggested_seed_field"] == "train_seed"
    assert info["path"] == "results.csv"


def test_analysis_primary_by_train_seed(root):
    path = write_csv(root / "results.csv", paired_rows(),
                     ["algorithm", "train_seed", "reward_mean", "coins_mean"])
    result = analysis.analyze_results(
        path, metric="reward_mean", baseline="dqn", paired=True, n_boot=200)
    assert result["analysis_unit"] == "train_seed"
    assert result["analysis_quality"] == "primary"
    assert result["paired"] is True
    assert {item["algorithm"] for item in result["summaries"]} == {"dqn", "ppo"}
    assert len(result["comparisons"]) == 1
    assert (result["comparisons"][0]["a"],
            result["comparisons"][0]["b"]) == ("dqn", "ppo")
    assert "html" in result and "dqn" in result["html"]
    assert result["input"] == "results.csv"


def test_analysis_warns_when_unit_is_episode(root):
    rows = []
    for algorithm, base in (("dqn", 10), ("ppo", 5)):
        for episode in range(1, 7):
            rows.append({"algorithm": algorithm, "episode": episode,
                         "reward_mean": base + episode})
    path = write_csv(root / "results.csv", rows,
                     ["algorithm", "episode", "reward_mean"])
    result = analysis.analyze_results(path, n_boot=200)
    assert result["analysis_unit"] == "episode"
    assert result["analysis_quality"] == "secondary"
    assert "secundária" in result["analysis_note"]
    assert any("pseudo-replicação" in warning
               for warning in result["warnings"])


def test_analysis_switch_metric_and_baseline(root):
    path = write_csv(root / "results.csv", paired_rows(),
                     ["algorithm", "train_seed", "reward_mean", "coins_mean"])
    result = analysis.analyze_results(
        path, metric="coins_mean", baseline="ppo", n_boot=200)
    assert result["metric"] == "coins_mean"
    assert result["comparisons"][0]["a"] == "ppo"
    assert "coins_mean" in result["html"]


def test_analysis_aggregate_collapses_repeated_units(root):
    rows = []
    for seed in (7, 42, 123):
        for episode in (1, 2, 3):
            rows.append({"algorithm": "dqn", "train_seed": seed,
                         "episode": episode,
                         "reward_mean": 10 + seed + episode})
    path = write_csv(root / "results.csv", rows,
                     ["algorithm", "train_seed", "episode", "reward_mean"])
    result = analysis.analyze_results(
        path, seed_field="train_seed", aggregate=True, n_boot=200)
    summary = result["summaries"][0]
    assert summary["n"] == 3
    assert any("agregada" in warning for warning in result["warnings"])


def test_analysis_rejects_unsafe_or_missing_inputs(root, tmp_path):
    path = write_csv(root / "results.csv", paired_rows(),
                     ["algorithm", "train_seed", "reward_mean", "coins_mean"])
    outside = tmp_path.parent / "fora.csv"
    write_csv(outside, paired_rows(),
              ["algorithm", "train_seed", "reward_mean", "coins_mean"])
    with pytest.raises(analysis.AnalysisError):
        analysis.inspect_results(outside)
    with pytest.raises(analysis.AnalysisError):
        analysis.inspect_results(root / "nao-existe.csv")
    with pytest.raises(analysis.AnalysisError):
        analysis.analyze_results(
            path, baseline="inexistente", n_boot=100)
    rows = [{"algorithm": "dqn", "reward_mean": 1.0},
            {"algorithm": "ppo", "reward_mean": 2.0}]
    no_seed = write_csv(root / "no-seed.csv", rows,
                        ["algorithm", "reward_mean"])
    with pytest.raises(analysis.AnalysisError):
        analysis.analyze_results(no_seed, paired=True, n_boot=100)


def test_discover_sources_uses_experiment_dir_and_extras(root):
    csv_path = write_csv(root / "experiments" / "EXP-x" / "results.csv",
                         paired_rows(),
                         ["algorithm", "train_seed", "reward_mean",
                          "coins_mean"])
    extra = write_csv(root / "logs" / "results-extra.csv", paired_rows(),
                      ["algorithm", "train_seed", "reward_mean",
                       "coins_mean"])
    data = {"artifacts": {
        "experiment_dir": "experiments/EXP-x",
        "results_csv": "experiments/EXP-x/results.csv",
    }}
    sources = analysis.discover_sources(data, ["logs/results-extra.csv"])
    paths = {source["path"] for source in sources}
    assert "experiments/EXP-x/results.csv" in paths
    assert "logs/results-extra.csv" in paths
    assert str(csv_path.relative_to(root)) in paths
    assert all(source["algorithms"] for source in sources)
    assert extra.exists()

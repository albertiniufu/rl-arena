"""Testes do relatório estatístico de experimentos."""
import csv
import json
import math

import numpy as np
import pytest

from arenas.common import report as stats


def test_bootstrap_ci_contains_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = stats.bootstrap_ci(values, n_boot=2000, rng=7)
    assert low <= np.mean(values) <= high
    assert low < high


def test_summarize_descriptive_fields():
    mean, median, std, q1, q3, iqr, low, high = stats.summarize(
        [1, 2, 3, 4, 5], n_boot=1000, rng=7)
    assert mean == pytest.approx(3.0)
    assert median == pytest.approx(3.0)
    assert std == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert q1 == pytest.approx(2.0)
    assert q3 == pytest.approx(4.0)
    assert iqr == pytest.approx(2.0)
    assert low <= mean <= high


def test_wilcoxon_detects_consistent_difference():
    a = [10, 12, 14, 16, 18]
    b = [0, 0, 0, 0, 0]
    statistic, p_value = stats.wilcoxon_signed_rank(a, b)
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(0.0625)
    assert p_value < 0.1


def test_wilcoxon_without_difference_is_not_significant():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    _, p_value = stats.wilcoxon_signed_rank(a, b)
    assert p_value == pytest.approx(1.0)


def test_mann_whitney_detects_separation():
    a = [10, 11, 12, 13, 14]
    b = [0, 1, 2, 3, 4]
    statistic, p_value = stats.mann_whitney_u(a, b)
    assert statistic == pytest.approx(0.0)
    assert p_value < 0.05


def test_mann_whitney_without_difference_is_not_significant():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    _, p_value = stats.mann_whitney_u(a, b)
    assert p_value > 0.8  # aproximação normal com correção de continuidade


def test_effect_sizes_have_expected_sign():
    a = [10, 11, 12]
    b = [1, 2, 3]
    assert stats.cliffs_delta(a, b) == pytest.approx(1.0)
    assert stats.cohens_d(a, b, paired=False) > 0
    assert stats.cohens_d([3, 3, 3], [1, 1, 1], paired=True) == math.inf


def test_holm_bonferroni_known_example():
    adjusted = stats.holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_benjamini_hochberg_known_example():
    adjusted = stats.benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted.tolist() == pytest.approx([0.03, 0.04, 0.04])


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_csv_and_build_report(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 42, "reward_mean": 11.0},
        {"algorithm": "dqn", "seed": 123, "reward_mean": 12.0},
        {"algorithm": "dqn", "seed": 1, "reward_mean": 13.0},
        {"algorithm": "dqn", "seed": 2, "reward_mean": 14.0},
        {"algorithm": "dqn", "seed": 3, "reward_mean": 15.0},
        {"algorithm": "ppo", "seed": 7, "reward_mean": 4.0},
        {"algorithm": "ppo", "seed": 42, "reward_mean": 5.0},
        {"algorithm": "ppo", "seed": 123, "reward_mean": 6.0},
        {"algorithm": "ppo", "seed": 1, "reward_mean": 7.0},
        {"algorithm": "ppo", "seed": 2, "reward_mean": 8.0},
        {"algorithm": "ppo", "seed": 3, "reward_mean": 9.0},
    ])
    observations, warnings = stats.load_observations(
        path, metric="reward_mean")
    assert not warnings
    report = stats.build_report(
        observations, metric="reward_mean", paired=True,
        correction="holm", n_boot=1000)
    assert [s.algorithm for s in report.summaries] == ["dqn", "ppo"]
    assert len(report.comparisons) == 1
    comparison = report.comparisons[0]
    assert comparison.paired is True
    assert comparison.diff == pytest.approx(6.0)
    assert comparison.significant is True
    text = stats.render_markdown(report)
    assert "Relatório estatístico" in text
    assert "Wilcoxon" in text
    assert "dqn" in text and "ppo" in text


def test_load_jsonl_nested_fields(tmp_path):
    path = tmp_path / "manual_eval.jsonl"
    rows = [
        {"tag": "happo", "episode": 1, "metrics": {"overall": {"reward_mean": 2.0}}},
        {"tag": "happo", "episode": 2, "metrics": {"overall": {"reward_mean": 3.0}}},
        {"tag": "mappo", "episode": 1, "metrics": {"overall": {"reward_mean": 1.0}}},
        {"tag": "mappo", "episode": 2, "metrics": {"overall": {"reward_mean": 1.5}}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows),
                    encoding="utf-8")
    observations, _ = stats.load_observations(
        path, metric="reward_mean",
        value_field="metrics.overall.reward_mean",
        algorithm_field="tag", seed_field="episode")
    assert len(observations) == 4
    report = stats.build_report(observations, metric="reward_mean",
                                paired=True, n_boot=500)
    assert report.comparisons[0].a == "happo"
    assert report.comparisons[0].b == "mappo"


def test_filter_and_missing_value_warning(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "level": 1, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 42, "level": 2, "reward_mean": 20.0},
        {"algorithm": "ppo", "seed": 7, "level": 1, "reward_mean": 5.0},
        {"algorithm": "ppo", "seed": 42, "level": 2, "reward_mean": 6.0},
    ])
    observations, warnings = stats.load_observations(
        path, metric="reward_mean", filters={"level": "1"})
    assert len(observations) == 2
    assert not warnings
    assert {o.level for o in observations} == {"1"}


def test_cli_writes_markdown(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 42, "reward_mean": 11.0},
        {"algorithm": "dqn", "seed": 123, "reward_mean": 12.0},
        {"algorithm": "ppo", "seed": 7, "reward_mean": 4.0},
        {"algorithm": "ppo", "seed": 42, "reward_mean": 5.0},
        {"algorithm": "ppo", "seed": 123, "reward_mean": 6.0},
    ])
    out = tmp_path / "report.md"
    stats.main([
        "--input", str(path), "--metric", "reward_mean",
        "--paired", "--correction", "holm", "--bootstrap", "500",
        "--out", str(out),
    ])
    text = out.read_text(encoding="utf-8")
    assert "## Descritivas" in text
    assert "## Comparações" in text
    assert "## Conclusão" in text


def test_cli_writes_json(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 42, "reward_mean": 11.0},
        {"algorithm": "ppo", "seed": 7, "reward_mean": 4.0},
        {"algorithm": "ppo", "seed": 42, "reward_mean": 5.0},
    ])
    out = tmp_path / "report.json"
    stats.main([
        "--input", str(path), "--metric", "reward_mean",
        "--bootstrap", "200", "--out", str(out),
    ])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metric"] == "reward_mean"
    assert len(payload["summaries"]) == 2


def test_render_html_embeds_report_data(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 42, "reward_mean": 11.0},
        {"algorithm": "ppo", "seed": 7, "reward_mean": 4.0},
        {"algorithm": "ppo", "seed": 42, "reward_mean": 5.0},
    ])
    observations, _ = stats.load_observations(path, metric="reward_mean")
    report = stats.build_report(observations, metric="reward_mean",
                                paired=True, n_boot=200)
    html = stats.render_html(report)
    assert "<!doctype html>" in html
    assert "chart-means" in html and "chart-effect" in html
    assert "const REPORT =" in html
    assert "dqn" in html and "ppo" in html
    assert "__DATA__" not in html
    assert "Plotly" in html
    assert "Plotly.newPlot" in html


def test_cli_writes_html(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 42, "reward_mean": 11.0},
        {"algorithm": "dqn", "seed": 123, "reward_mean": 12.0},
        {"algorithm": "ppo", "seed": 7, "reward_mean": 4.0},
        {"algorithm": "ppo", "seed": 42, "reward_mean": 5.0},
        {"algorithm": "ppo", "seed": 123, "reward_mean": 6.0},
    ])
    out = tmp_path / "report.html"
    stats.main([
        "--input", str(path), "--metric", "reward_mean",
        "--paired", "--bootstrap", "200", "--out", str(out),
    ])
    html = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "table-comparisons" in html
    assert "Wilcoxon" in html


def test_aggregate_by_unit_averages_repeated_episodes():
    observations = [
        stats.Observation("dqn", 10.0, seed="7", source="a:1"),
        stats.Observation("dqn", 12.0, seed="7", source="a:2"),
        stats.Observation("dqn", 14.0, seed="42", source="a:3"),
        stats.Observation("ppo", 4.0, seed="7", source="a:4"),
        stats.Observation("ppo", 6.0, seed="7", source="a:5"),
        stats.Observation("ppo", 8.0, seed="42", source="a:6"),
    ]
    warnings = []
    aggregated = stats.aggregate_by_unit(observations, warnings=warnings)
    assert len(aggregated) == 4
    assert aggregated[0].value == pytest.approx(11.0)
    assert aggregated[0].seed == "7"
    assert len(warnings) == 2
    report = stats.build_report(aggregated, metric="reward_mean",
                                paired=True, n_boot=200)
    assert report.comparisons[0].n == 2


def test_unpaired_report_warns_about_repeated_seed(tmp_path):
    path = tmp_path / "results.csv"
    _write_csv(path, [
        {"algorithm": "dqn", "seed": 7, "episode": 1, "reward_mean": 10.0},
        {"algorithm": "dqn", "seed": 7, "episode": 2, "reward_mean": 12.0},
        {"algorithm": "ppo", "seed": 7, "episode": 1, "reward_mean": 4.0},
        {"algorithm": "ppo", "seed": 7, "episode": 2, "reward_mean": 6.0},
    ])
    observations, _ = stats.load_observations(
        path, metric="reward_mean", seed_field="seed")
    report = stats.build_report(observations, metric="reward_mean",
                                n_boot=100)
    assert any("seed repetida" in warning for warning in report.warnings)

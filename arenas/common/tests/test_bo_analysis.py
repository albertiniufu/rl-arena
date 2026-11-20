"""Testes do parser e das estatísticas exploratórias de BO/ASHA."""
import json

import pytest

from arenas.common import bo_analysis
from arenas.common import experiment_analysis as analysis

OK_TRIALS = 12


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    return tmp_path


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8")


def _bo_trial(index, lr, batch, score, status="ok", kind="trial"):
    record = {
        "algorithm": "ppo", "index": index,
        "params": {"lr": lr, "batch_size": batch, "gamma": 0.9 + index * 0.005},
        "reward_mean": score - 0.5, "coins_mean": 10.0,
        "deaths_mean": 1.0, "score": score,
        "seconds": 600.0 + index * 10, "status": status,
        "trial_dir": "", "error": "",
    }
    if kind == "final":
        record["trial_dir"] = "experiments/BO-test/ppo/final"
    return record


def make_campaign(root, with_manifest=True, with_best=True,
                  failed=True, asha=False):
    campaign = root / "experiments" / "BO-test"
    ppo = campaign / "ppo"
    ppo.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(OK_TRIALS):
        records.append(_bo_trial(
            index, lr=1e-4 * (index + 1), batch=(64 if index % 2 else 128),
            score=10.0 + index, status="ok"))
    if failed:
        records.append(_bo_trial(OK_TRIALS, lr=2e-3, batch=64,
                                 score=float("-inf"), status="erro"))
    final = _bo_trial(OK_TRIALS + 1, lr=1.3e-3, batch=128,
                      score=24.0, kind="final")
    final["score"] = 24.0
    records.append(final)
    if asha:
        for record in records:
            record.pop("seconds", None)
            record["minutes"] = 10.0
            record["rung"] = 0 if record["index"] % 2 else 1
        _write_jsonl(ppo / "trials.jsonl", records)
        (ppo / "results.csv").write_text(
            "index,rung,score,reward,coins,deaths,status,params\n"
            '0,0,10,9.5,10,1,ok,"{}"\n', encoding="utf-8")
    else:
        _write_jsonl(ppo / "trials.jsonl", records)
    if with_best:
        _write(ppo / "best.json", {
            "algorithm": "ppo", "best_trial": 9,
            "best_params": {"lr": 1e-3, "batch_size": 64, "gamma": 0.945},
            "selection_metric": "score", "selection_eval_seed": 1000,
            "best_low_fidelity": {"score": 19.0},
            "n_trials": len(records),
            "final_eval_seed": 2000,
            "final": {"status": "ok", "reward_mean": 23.5, "coins_mean": 10,
                      "deaths_mean": 0, "params": {
                          "lr": 1e-3, "batch_size": 64, "gamma": 0.945},
                      "minutes": 30, "eval_seed": 2000,
                      "held_out": True},
        })
    if with_manifest:
        _write(campaign / "campaign.json", {
            "algorithms": ["ppo"], "base_config": "configs/base.yaml",
            "space_dir": "configs/spaces", "hours_per_alg": 1.0,
            "started_at": "2026-11-20 10:00:00",
        })
        (root / "configs" / "spaces").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "spaces" / "ppo.yaml").write_text(
            "params:\n"
            "  - name: lr\n    type: float\n    low: 0.0001\n    high: 0.001\n"
            "  - name: batch_size\n    type: choice\n    choices: [64, 128]\n"
            "  - name: gamma\n    type: float\n    low: 0.9\n    high: 0.99\n",
            encoding="utf-8")
        (root / "configs" / "base.yaml").write_text(
            "arena: seek_coin\ntraining:\n  agent_params:\n    lr: 0.001\n"
            "    batch_size: 64\n", encoding="utf-8")
    return campaign


def test_load_bo_normalizes_trials_and_marks_best(root):
    campaign = make_campaign(root)
    payload = bo_analysis.load_bo(campaign)
    assert payload["root"] == "experiments/BO-test"
    assert [campaign["algorithm"] for campaign in payload["campaigns"]] == ["ppo"]
    entry = payload["campaigns"][0]
    assert entry["source"] == "bo"
    assert entry["n_trials"] == OK_TRIALS + 2
    assert entry["n_ok"] == OK_TRIALS + 1
    assert entry["n_failed"] == 1
    best = [trial for trial in entry["trials"] if trial["is_best"]]
    assert len(best) == 1 and best[0]["index"] == 9
    for trial in entry["trials"]:
        assert set(("index", "status", "score", "reward", "coins", "deaths",
                    "duration_seconds", "params", "artifact", "log")) <= set(trial)
    failed = [trial for trial in entry["trials"] if not trial["ok"]][0]
    assert failed["score"] is None and failed["status"] == "erro"
    final = [trial for trial in entry["trials"] if trial["kind"] == "final"][0]
    assert final["score"] == 24.0


def test_best_so_far_is_monotonic_and_ignores_failures(root):
    payload = bo_analysis.load_bo(make_campaign(root))
    series = payload["campaigns"][0]["best_so_far"]
    bests = [point["best"] for point in series if point["best"] is not None]
    assert bests == sorted(bests)
    assert series[0]["best"] == 10.0
    failed = [point for point in series if point["status"] == "erro"]
    assert failed and failed[0]["score"] is None


def test_numeric_parameter_has_bins_spearman_and_points(root):
    payload = bo_analysis.load_bo(make_campaign(root))
    params = {param["name"]: param
              for param in payload["campaigns"][0]["params"]}
    lr = params["lr"]
    assert lr["kind"] == "numeric"
    assert lr["n"] == OK_TRIALS
    assert lr["spearman"] == pytest.approx(1.0)
    assert len(lr["bins"]) <= 4
    assert sum(bin_["n"] for bin_ in lr["bins"]) == OK_TRIALS
    means = [bin_["mean"] for bin_ in lr["bins"]]
    assert means == sorted(means)
    assert all(bin_["ci95"] is not None for bin_ in lr["bins"])


def test_categorical_parameter_groups_and_variance(root):
    payload = bo_analysis.load_bo(make_campaign(root))
    batch = {param["name"]: param
             for param in payload["campaigns"][0]["params"]}["batch_size"]
    assert batch["kind"] == "categorical"
    assert len(batch["groups"]) == 2
    assert sum(group["n"] for group in batch["groups"]) == OK_TRIALS
    assert batch["eta_squared"] is not None


def test_interaction_and_importance_ranking(root):
    payload = bo_analysis.load_bo(make_campaign(root))
    entry = payload["campaigns"][0]
    assert entry["params"][0]["rank"] == 1
    interaction = entry["interaction"]
    assert interaction is not None
    assert interaction["x_param"] != interaction["y_param"]
    assert len(interaction["means"]) == 3
    assert sum(sum(row) for row in interaction["counts"]) == OK_TRIALS


def test_improvement_compares_best_with_median_first_and_baseline(root):
    payload = bo_analysis.load_bo(make_campaign(root))
    improvement = payload["campaigns"][0]["improvement"]
    assert improvement["best_score"] == pytest.approx(19.0)
    assert improvement["median_score"] is not None
    assert improvement["first_score"] == 10.0
    assert improvement["delta_vs_first"] == pytest.approx(9.0)
    assert improvement["baseline_params"]["lr"] == 0.001
    assert "gamma" in improvement["changed_params"]
    assert improvement["final"]["held_out"] is True


def test_asha_source_uses_rungs_for_budget(root):
    payload = bo_analysis.load_bo(make_campaign(root, asha=True))
    entry = payload["campaigns"][0]
    assert entry["source"] == "asha"
    assert {band["label"] for band in entry["budget"]} == {"rung 0", "rung 1"}
    assert all("n" in band and "mean" in band for band in entry["budget"])


def test_missing_best_space_and_manifest_are_warnings(root):
    campaign = make_campaign(root, with_best=False)
    (root / "configs" / "spaces" / "ppo.yaml").unlink()
    payload = bo_analysis.load_bo(campaign)
    warnings = payload["campaigns"][0]["warnings"]
    assert any("best.json ausente" in warning for warning in warnings)
    assert any("espaço de busca" in warning for warning in warnings)


def test_missing_campaign_and_outside_root_fail(root, tmp_path):
    empty = root / "experiments" / "vazio"
    empty.mkdir(parents=True)
    with pytest.raises(bo_analysis.BoAnalysisError):
        bo_analysis.load_bo(empty)
    outside = tmp_path.parent / "fora"
    outside.mkdir(exist_ok=True)
    with pytest.raises(bo_analysis.BoAnalysisError):
        bo_analysis.load_bo(outside)


def test_load_for_job_uses_out_flag(root):
    campaign = make_campaign(root)
    job = {"id": "job-1", "cwd": str(root),
           "command": ["python", "scripts/bo_campaign.py",
                       "--out", "experiments/BO-test"]}
    payload = bo_analysis.load_bo_for_job(job)
    assert payload["root"] == "experiments/BO-test"
    assert payload["campaigns"][0]["algorithm"] == "ppo"
    assert campaign.exists()

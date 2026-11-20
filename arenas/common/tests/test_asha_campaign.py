"""Testes do scheduler ASHA."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "asha_campaign", ROOT / "scripts" / "asha_campaign.py")
asha = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = asha
SPEC.loader.exec_module(asha)
from scripts.bo_campaign import Param, Trial  # noqa: E402


def test_should_promote_keeps_top_fraction():
    assert asha.should_promote([], 1.0, 2) is True
    assert asha.should_promote([1.0, 2.0], 2.0, 2) is True
    assert asha.should_promote([1.0, 2.0], 1.0, 2) is False
    assert asha.should_promote([1.0, 2.0, 3.0, 4.0], 3.0, 2) is True
    assert asha.should_promote([1.0, 2.0, 3.0, 4.0], 2.0, 2) is False


def test_run_algorithm_promotes_and_writes_results(tmp_path, monkeypatch):
    space = (Param("x", "float", 0.0, 1.0),)

    def fake_run_trial(algorithm, params, trial_dir, minutes,
                       eval_episodes, eval_seed, base_config):
        trial_dir.mkdir(parents=True, exist_ok=True)
        return Trial(
            algorithm=algorithm, index=-1, params=params,
            reward_mean=float(params["x"]), coins_mean=0.0,
            deaths_mean=0.0, score=0.0, seconds=0.01, status="ok",
            trial_dir=str(trial_dir))

    monkeypatch.setattr(asha, "run_trial", fake_run_trial)
    args = SimpleNamespace(
        out=str(tmp_path), rungs="1,2", eta=2, initial_trials=4,
        min_completions=2, max_parallel=2, budget_hours=1.0,
        eval_episodes=1, eval_seed=1000, seed=7,
        rung_minutes_guard=False)
    result = asha.run_algorithm("ppo", space, {"arena": "seek_coin"}, args)
    assert result["n_trials"] >= 4
    assert (tmp_path / "ppo" / "results.csv").exists()
    assert (tmp_path / "ppo" / "best.json").exists()

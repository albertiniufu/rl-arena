"""Testes do motor de BO e do carregamento de espaços YAML."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "bo_campaign", ROOT / "scripts" / "bo_campaign.py")
bo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bo
SPEC.loader.exec_module(bo)


def test_load_space_from_yaml(tmp_path):
    path = tmp_path / "space.yaml"
    path.write_text("""
algorithm: ppo
params:
  - name: lr
    type: float
    low: 0.0001
    high: 0.001
    log: true
  - name: batch_size
    type: choice
    choices: [64, 128, 256]
""", encoding="utf-8")
    space = bo.load_space(path)
    assert [p.name for p in space] == ["lr", "batch_size"]
    lr = space[0].decode(0.5)
    assert 0.0001 < lr < 0.001
    assert space[1].decode(0.0) == 64
    assert space[1].decode(1.0) == 256


def test_suggest_returns_valid_params_after_initial_trials():
    space = (
        bo.Param("x", "float", 0.0, 1.0),
        bo.Param("choice", "choice", choices=("a", "b")),
    )
    rng = np.random.default_rng(7)
    X, y = [], []
    for _ in range(6):
        params = bo.random_params(space, rng)
        X.append(bo.encode_params(space, params))
        # Objetivo sintético: maximiza x e prefere "b".
        y.append(params["x"] + (1.0 if params["choice"] == "b" else 0.0))
    params = bo.suggest(space, X, y, rng, n_candidates=500)
    assert set(params) == {"x", "choice"}
    assert 0.0 <= params["x"] <= 1.0
    assert params["choice"] in ("a", "b")


def test_gp_handles_constant_objective():
    space = (bo.Param("x", "float", 0.0, 1.0),)
    rng = np.random.default_rng(7)
    X = [bo.encode_params(space, bo.random_params(space, rng))
         for _ in range(6)]
    params = bo.suggest(space, X, [0.0] * 6, rng, n_candidates=100)
    assert 0.0 <= params["x"] <= 1.0


def test_selection_value_respects_primary_metric():
    trial = bo.Trial(
        algorithm="ppo", index=0, params={}, reward_mean=10.0,
        coins_mean=4.0, deaths_mean=1.0, score=10.2, seconds=1.0,
        status="ok", trial_dir=".",
    )
    assert bo.selection_value(trial, "score") == 10.2
    assert bo.selection_value(trial, "reward") == 10.0
    assert bo.selection_value(trial, "coins") == 4.0


def test_cli_defaults_use_held_out_final_eval_seed():
    args = bo.parse_args([])
    assert args.eval_seed == 1000
    assert args.final_eval_seed is None
    assert args.selection_metric == "score"

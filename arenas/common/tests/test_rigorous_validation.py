"""Testes do runner de validação multi-seed."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "rigorous_validation", ROOT / "scripts" / "rigorous_validation.py")
rv = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rv
SPEC.loader.exec_module(rv)


def test_load_config_params_from_repo_configs():
    ppo = rv._load_config_params("configs/seek_coin/ppo-level3.yaml")
    eppo = rv._load_config_params("configs/seek_coin/eppo-level3.yaml")
    assert ppo["lr"] == 0.0006854892425731042
    assert "ucb_radius" in eppo


def test_parse_args_accepts_config_mode():
    args = rv.parse_args([
        "--hours", "2", "--seeds", "7,42",
        "--ppo-config", "configs/seek_coin/ppo-level3.yaml",
        "--eppo-config", "configs/seek_coin/eppo-level3.yaml",
        "--out", "experiments/teste",
    ])
    assert args.hours == 2.0
    assert args.bo_root is None
    assert args.ppo_config.endswith("ppo-level3.yaml")
    assert args.eppo_config.endswith("eppo-level3.yaml")


def test_main_runs_one_training_at_a_time(tmp_path, monkeypatch):
    """Os treinos são sequenciais: PPO e EPPO nunca rodam juntos."""
    import pytest

    calls = []

    def fake_run_one(algorithm, seed, params, out, hours, eval_episodes,
                     eval_seed, level=3, view_size=None, super_coin=None):
        calls.append((algorithm, seed, level, view_size, super_coin))

    monkeypatch.setattr(rv, "run_one", fake_run_one)

    with pytest.raises(SystemExit):
        rv.main([
            "--hours", "0.01", "--seeds", "7,42",
            "--ppo-config", "configs/seek_coin/ppo-level3.yaml",
            "--eppo-config", "configs/seek_coin/eppo-level3.yaml",
            "--out", str(tmp_path / "experimento"),
        ])

    assert [(alg, seed) for alg, seed, *_ in calls] == [
        ("ppo", 7), ("eppo", 7), ("ppo", 42), ("eppo", 42)]
    # O nível 3 vem do preset e não inventa view_size/super moeda.
    assert {(call[2], call[3]) for call in calls} == {(3, None)}


def test_main_passes_level4_scenario_from_presets(tmp_path, monkeypatch):
    """Nível 4: nível, view_size e super moeda saem dos presets do nível 4."""
    import pytest

    calls = []

    def fake_run_one(algorithm, seed, params, out, hours, eval_episodes,
                     eval_seed, level=3, view_size=None, super_coin=None):
        calls.append((level, view_size, super_coin))

    monkeypatch.setattr(rv, "run_one", fake_run_one)

    with pytest.raises(SystemExit):
        rv.main([
            "--hours", "0.01", "--seeds", "7",
            "--ppo-config", "configs/seek_coin/ppo-level4.yaml",
            "--eppo-config", "configs/seek_coin/eppo-level4.yaml",
            "--out", str(tmp_path / "experimento4"),
        ])

    assert calls == [(4, 9, False), (4, 9, False)]


def test_scenario_cli_overrides_presets():
    args = rv.parse_args([
        "--level", "5", "--view-size", "7", "--super-coin",
        "--ppo-config", "configs/seek_coin/ppo-level4.yaml",
        "--eppo-config", "configs/seek_coin/eppo-level4.yaml",
    ])
    scenarios = {
        "ppo": rv._load_config_scenario(args.ppo_config),
        "eppo": rv._load_config_scenario(args.eppo_config),
    }
    assert rv._resolve_scenario(args, scenarios) == (5, 7, True)
    assert rv._scenario_args(4, 9, False) == [
        "--level", "4", "--view-size", "9", "--no-super-coin"]


def test_runner_does_not_spawn_parallel_processes():
    """Sem Popen: o runner só usa `subprocess.run` (um por vez)."""
    source = (ROOT / "scripts" / "rigorous_validation.py").read_text(
        encoding="utf-8")
    assert "Popen" not in source
    assert "ThreadPoolExecutor" not in source


def _write_by_seed(path, rows):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["algorithm", "seed", "n_eval",
                                "reward", "coins", "deaths"])
        writer.writeheader()
        writer.writerows(rows)


SEEDS7 = [7, 42, 123, 314, 999, 2024, 2025]


def test_family_tests_applies_holm_across_metrics(tmp_path):
    import json

    import pytest

    out = tmp_path / "exp"
    rows = []
    for index, seed in enumerate(SEEDS7):
        rows.append({"algorithm": "ppo", "seed": seed, "n_eval": 50,
                     "reward": 100.0, "coins": 50.0, "deaths": 3.0})
        rows.append({"algorithm": "eppo", "seed": seed, "n_eval": 50,
                     "reward": 110.0 + index, "coins": 55.0 + index,
                     "deaths": 3.0})
    _write_by_seed(out / "results-by-train-seed.csv", rows)

    payload = rv.family_tests(out, n_boot=200)

    assert payload["family"] == ["reward", "coins"]
    assert payload["correction"] == "holm"
    for test in payload["tests"]:
        assert test["n_pairs"] == 7
        # 7 pares, todas as diferenças na mesma direção: p exato = 2/2^7
        assert test["p_raw"] == pytest.approx(2 / 2 ** 7)
        assert test["p_holm"] == pytest.approx(2 * 2 / 2 ** 7)
        assert test["reject"] is True
    assert (out / "family-tests.md").exists()
    saved = json.loads((out / "family-tests.json").read_text(encoding="utf-8"))
    assert [test["metric"] for test in saved["tests"]] == ["reward", "coins"]


def test_family_tests_holm_is_monotone_and_skips_missing_seeds(tmp_path):
    import pytest

    out = tmp_path / "exp"
    rows = []
    for index, seed in enumerate(SEEDS7):
        rows.append({"algorithm": "ppo", "seed": seed, "n_eval": 50,
                     "reward": 100.0, "coins": 50.0, "deaths": 3.0})
        # reward: EPPO sempre melhor; coins: empate em uma seed
        rows.append({"algorithm": "eppo", "seed": seed, "n_eval": 50,
                     "reward": 110.0 + index,
                     "coins": 55.0 if index != 3 else 50.0,
                     "deaths": 3.0})
    # seed sem par de EPPO não entra na família
    rows.append({"algorithm": "ppo", "seed": 3141, "n_eval": 50,
                 "reward": 100.0, "coins": 50.0, "deaths": 3.0})
    _write_by_seed(out / "results-by-train-seed.csv", rows)

    payload = rv.family_tests(out, n_boot=200)
    reward, coins = payload["tests"]
    assert reward["n_pairs"] == 7
    assert coins["p_raw"] > reward["p_raw"]
    # Holm força monotonicidade: o segundo p ajustado nunca é menor
    assert coins["p_holm"] >= reward["p_holm"]
    assert "deaths" not in payload["family"]

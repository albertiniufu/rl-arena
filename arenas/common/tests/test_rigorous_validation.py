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

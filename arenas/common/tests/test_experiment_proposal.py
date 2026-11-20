"""Testes de geração, validação e gravação de propostas de experimento."""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from arenas.common import experiment_proposal as proposal

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fields() -> dict:
    return {
        "title": "PPO vs EPPO no nível 3",
        "author": "equipe",
        "question": "O EPPO supera o PPO no nível 3?",
        "hypothesis": "O crítico evidencial reduz a variância.",
        "falsification": "O IC inclui zero ou o PPO é melhor.",
        "claim_allowed": "Afirmação restrita ao setup S6.",
        "arena": "seek_coin",
        "level": "3",
        "scenario": "single-agent contra rule",
        "setup_id": "S6",
        "config_files": "configs/seek_coin/ppo-level3.yaml",
        "candidates": "ppo, eppo",
        "baseline": "ppo",
        "metric_name": "reward",
        "metric_direction": "maximize",
        "metric_definition": "reward guloso médio por seed de treino",
        "seeds_train": "7, 42, 123",
        "eval_episodes": "50",
        "env_steps_per_run": "4M",
        "wall_hours_per_run": "2",
        "n_runs": "6",
        "tuning_method": "none",
        "checkpoint": "latest",
        "truncation": "sem bootstrap",
        "independent_unit": "train_seed",
        "support": "EPPO > PPO com IC acima de zero.",
        "inconclusive": "IC inclui zero.",
        "abandon": "PPO significativamente melhor.",
        "command": "python train.py --config configs/seek_coin/ppo-level3.yaml",
        "cwd": ".",
        "max_runtime": "3600",
    }


def test_build_and_validate_valid_proposal(fields):
    payload = proposal.build_proposal(fields)
    assert payload["id"].startswith("EXP-")
    assert payload["setup"]["level"] == 3
    assert payload["budget"]["seeds_train"] == [7, 42, 123]
    assert payload["budget"]["n_runs"] == 6
    assert payload["execution"]["command"][0] == "python"
    assert payload["algorithms"]["baseline"] == ["ppo"]
    assert proposal.validate(payload) == []


def test_preview_has_yaml_and_markdown(fields):
    preview = proposal.preview_proposal(fields)
    assert preview["errors"] == []
    assert "## 1. Pergunta" in preview["markdown"]
    assert "## 17. Checklist final" in preview["markdown"]
    assert preview["proposal"]["question"] in preview["markdown"]
    loaded = yaml.safe_load(preview["yaml"])
    assert loaded["id"] == preview["proposal"]["id"]
    assert ("python train.py" in preview["markdown"])


def test_invalid_proposals_report_errors(fields):
    bad = dict(fields)
    bad["seeds_train"] = "7, 42"
    bad["metric_direction"] = "sideways"
    bad["arena"] = "outra"
    errors = proposal.preview_proposal(bad)["errors"]
    assert any("3+ seeds" in error for error in errors)
    assert any("maximize/minimize" in error for error in errors)
    assert any("setup.arena" in error for error in errors)

    shell = dict(fields)
    shell["command"] = "bash -c true"
    assert any("Python" in error for error in proposal.validate(
        proposal.build_proposal(shell)))

    outside = dict(fields)
    outside["command"] = "python /tmp/script.py"
    assert any("fora do repositório" in error for error in proposal.validate(
        proposal.build_proposal(outside)))

    missing = dict(fields)
    missing["question"] = ""
    assert any("question" in error for error in proposal.validate(
        proposal.build_proposal(missing)))

    unbalanced = dict(fields, command="python 'sem-fechar")
    errors = proposal.preview_proposal(unbalanced)["errors"]
    assert any("execution.command" in error for error in errors)


def test_validate_yaml_text():
    assert proposal.validate_yaml_text("id: [")["errors"]
    assert proposal.validate_yaml_text("- lista")["errors"]
    assert proposal.validate_yaml_text("id: EXP-20260101-x")["errors"]


def test_save_refuses_overwrite_and_accepts_explicit_overwrite(
        fields, tmp_path, monkeypatch):
    monkeypatch.setattr(proposal, "ROOT", tmp_path)
    saved = proposal.save_proposal(fields)
    yaml_path = tmp_path / saved["yaml_path"]
    md_path = tmp_path / saved["markdown_path"]
    assert yaml_path.exists() and md_path.exists()
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["id"] == \
        saved["id"]
    assert saved["id"] in md_path.read_text(encoding="utf-8")

    with pytest.raises(proposal.ProposalConflict):
        proposal.save_proposal(fields)
    again = proposal.save_proposal(fields, overwrite=True)
    assert again["id"] == saved["id"]


def test_save_rejects_directory_outside_repo(fields, tmp_path, monkeypatch):
    monkeypatch.setattr(proposal, "ROOT", tmp_path)
    with pytest.raises(proposal.ProposalError):
        proposal.save_proposal(fields, directory="../fora")


def test_save_rejects_invalid_proposal(fields, tmp_path, monkeypatch):
    monkeypatch.setattr(proposal, "ROOT", tmp_path)
    bad = dict(fields, seeds_train="7")
    with pytest.raises(proposal.ProposalError):
        proposal.save_proposal(bad)


def test_skill_validator_delegates_to_shared_module(fields):
    script = (REPO_ROOT / ".agents" / "skills" / "experimenter-helper" /
              "scripts" / "validate_yaml.py")
    spec = importlib.util.spec_from_file_location("validate_yaml_skill", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payload = proposal.build_proposal(fields)
    assert module.validate(payload) == []
    assert module.validate(dict(payload, status="talvez"))

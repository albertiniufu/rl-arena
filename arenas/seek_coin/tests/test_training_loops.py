"""Testes de fumaça dos quatro loops de treino do seek_coin.

Cada teste roda `train.main()` de verdade, com uma `ArenaConfig` mínima
(episódios de 0,2 s) e o diretório de trabalho apontado para `tmp_path`.
Assim os `logs/` e `checkpoints/` relativos não sujam o repositório.

O objetivo é cobrir os caminhos que a suíte histórica não exercitava:
single-agent, vetorizado off-policy, vetorizado on-policy e multiagente.
"""
import sys

import pytest

from arenas.seek_coin import train as train_mod
from arenas.seek_coin.arena.config import ArenaConfig as RealArenaConfig

FAST_CONFIG = {
    "episode_seconds": 0.2,
    "respawn_seconds": 0.1,
    "coin_count": 2,
}


@pytest.fixture
def fast_config(monkeypatch):
    """Torna os episódios curtos sem mudar o código de produção."""
    monkeypatch.setattr(
        train_mod, "ArenaConfig",
        lambda: RealArenaConfig(**FAST_CONFIG),
    )


def run_train(monkeypatch, tmp_path, *argv):
    """Executa `train.main()` como se fosse a CLI, dentro de `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    train_mod.main()


def read_training_csv(tmp_path):
    path = tmp_path / "logs" / "training.csv"
    assert path.exists(), "logs/training.csv não foi criado"
    return path.read_text(encoding="utf-8").splitlines()


def test_single_agent_loop_writes_log_and_checkpoint(
        monkeypatch, tmp_path, fast_config):
    run_train(
        monkeypatch, tmp_path,
        "--alg", "dqn", "--episodes", "2", "--opponent", "rule",
        "--save-every", "1", "--checkpoint-dir", "ckpt",
    )

    rows = read_training_csv(tmp_path)
    assert len(rows) >= 3  # cabeçalho + 2 episódios
    assert (tmp_path / "ckpt" / "latest.pt").exists()
    assert (tmp_path / "ckpt" / "best.pt").exists()


def test_vectorized_offpolicy_loop(monkeypatch, tmp_path, fast_config):
    run_train(
        monkeypatch, tmp_path,
        "--alg", "dqn", "--episodes", "2", "--envs", "2",
        "--opponent", "rule", "--save-every", "1",
        "--checkpoint-dir", "ckpt",
    )

    rows = read_training_csv(tmp_path)
    assert len(rows) >= 2  # pelo menos um episódio terminou
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_vectorized_onpolicy_loop(monkeypatch, tmp_path, fast_config):
    run_train(
        monkeypatch, tmp_path,
        "--alg", "ppo", "--episodes", "2", "--envs", "2",
        "--opponent", "rule", "--save-every", "1",
        "--checkpoint-dir", "ckpt",
    )

    rows = read_training_csv(tmp_path)
    assert len(rows) >= 2
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_overlap_loop(monkeypatch, tmp_path, fast_config):
    run_train(
        monkeypatch, tmp_path,
        "--alg", "ppo", "--episodes", "1", "--envs", "2", "--overlap",
        "--opponent", "rule", "--save-every", "1",
        "--checkpoint-dir", "ckpt",
    )

    rows = read_training_csv(tmp_path)
    assert len(rows) >= 2
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_multi_agent_loop(monkeypatch, tmp_path, fast_config):
    run_train(
        monkeypatch, tmp_path,
        "--alg", "ippo", "--agents", "2", "--scenario", "all-vs-all",
        "--episodes", "1", "--save-every", "1",
        "--checkpoint-dir", "ckpt",
    )

    rows = read_training_csv(tmp_path)
    assert len(rows) >= 2  # cabeçalho + 1 episódio
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_save_every_zero_disables_periodic_save(
        monkeypatch, tmp_path, fast_config):
    """`--save-every 0` desliga o save periódico, mas o final ainda grava."""
    run_train(
        monkeypatch, tmp_path,
        "--alg", "dqn", "--episodes", "1", "--opponent", "rule",
        "--save-every", "0", "--checkpoint-dir", "ckpt",
    )

    # O checkpoint final é sempre gravado para permitir retomada.
    assert (tmp_path / "ckpt" / "latest.pt").exists()
    # E o loop não pode quebrar no `episódio % 0` (regressão do ZeroDivision).
    assert len(read_training_csv(tmp_path)) >= 2


def test_resume_continues_episode_numbering(
        monkeypatch, tmp_path, fast_config):
    """A retomada usa `episodes` do checkpoint e continua a numeração."""
    run_train(
        monkeypatch, tmp_path,
        "--alg", "dqn", "--episodes", "1", "--opponent", "rule",
        "--save-every", "1", "--checkpoint-dir", "ckpt",
    )
    run_train(
        monkeypatch, tmp_path,
        "--alg", "dqn", "--episodes", "1", "--opponent", "rule",
        "--save-every", "0", "--checkpoint-dir", "ckpt",
        "--resume", "ckpt/latest.pt",
    )

    rows = read_training_csv(tmp_path)
    assert rows[0].startswith("episode,")
    assert rows[-1].startswith("2,")


def test_budget_snapshots_are_opt_in():
    """O protocolo A/B de snapshots de orçamento não roda por padrão."""
    defaults = train_mod.parse_args(["--episodes", "1"])
    assert defaults.budget_snapshots is False
    enabled = train_mod.parse_args(["--episodes", "1", "--budget-snapshots"])
    assert enabled.budget_snapshots is True

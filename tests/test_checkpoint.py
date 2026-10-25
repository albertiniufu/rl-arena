import pytest
import torch

from dqn.agent import DQNAgent
from dqn.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_lightweight(tmp_path):
    agent = DQNAgent((5, 32, 20), 12)
    path = tmp_path / "c.pt"
    save_checkpoint(path, agent, extra={"a": 1}, episodes=5)
    payload = torch.load(path, weights_only=False)
    assert "buffer" not in payload and "env_state" not in payload and "rng_state" not in payload
    assert payload["episodes"] == 5
    agent2 = DQNAgent((5, 32, 20), 12)
    load_checkpoint(path, agent2, resume=True)  # sem env
    assert agent2.steps == agent.steps


def test_evaluate_opponent_flag():
    from evaluate import parse_args

    assert parse_args([]).opponent == "rule"
    for name in ("rule", "random", "coins"):
        assert parse_args(["--opponent", name]).opponent == name


def test_train_frame_stack_and_skip_flags():
    from train import parse_args

    defaults = parse_args([])
    assert defaults.frame_stack == 1
    assert defaults.frame_skip == 1
    args = parse_args(["--frame-stack", "4", "--frame-skip", "3"])
    assert args.frame_stack == 4
    assert args.frame_skip == 3


def test_budget_snapshots():
    from train import budget_snapshots

    assert budget_snapshots(300) == {50, 100, 150, 300}
    assert budget_snapshots(100) == {50, 100}
    assert budget_snapshots(49) == set()


def test_try_load_skips_incompatible(tmp_path):
    from train import _try_load_checkpoint

    agent = DQNAgent((5, 32, 20), 12)
    junk = tmp_path / "old.pt"
    junk.write_bytes(b"lixo")  # não é um checkpoint válido
    with pytest.warns(UserWarning):
        assert _try_load_checkpoint(junk, agent) is False
    # e um checkpoint válido (Task 1) carrega: True
    good = tmp_path / "good.pt"
    save_checkpoint(good, agent)
    assert _try_load_checkpoint(good, agent) is True


def test_auto_opponent_ignores_budget_subdir(tmp_path):
    from arena.config import ArenaConfig
    from arena.environment import ArenaEnv
    from train import AutoOpponent

    (tmp_path / "budget").mkdir()
    (tmp_path / "budget" / "ep050.pt").write_bytes(b"lixo")
    (tmp_path / "best.pt").write_bytes(b"lixo")
    env = ArenaEnv(ArenaConfig(), opponent_policy="random")
    auto = AutoOpponent(env, checkpoint_dir=str(tmp_path))
    # Glob não-recursivo: snapshots de orçamento em budget/ não viram adversário.
    assert auto._checkpoints() == [tmp_path / "best.pt"]
    env.close()


def test_auto_opponent_init_from_disk_skips_incompatible(tmp_path):
    from arena.config import ArenaConfig
    from arena.environment import ArenaEnv
    from train import AutoOpponent

    (tmp_path / "best.pt").write_bytes(b"lixo")  # ex.: best.pt de arquitetura antiga
    env = ArenaEnv(ArenaConfig(), opponent_policy="random")
    auto = AutoOpponent(env, checkpoint_dir=str(tmp_path))
    with pytest.warns(UserWarning):
        auto.init_from_disk()  # avisa e mantém os pesos iniciais, sem levantar
    env.close()

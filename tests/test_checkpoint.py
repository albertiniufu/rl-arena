import pytest
import torch

from alg_doubledqn_per_dueling.agent import DoubleDQNPerDuelingAgent
from alg_dqn.agent import DQNAgent
from alg_registry import (
    algorithm_from_checkpoint,
    apply_env_config,
    checkpoint_meta,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)
from arena.config import ArenaConfig


def test_checkpoint_meta_reads_config_and_algorithm(tmp_path):
    agent = DQNAgent((5, 32, 20), 6)
    path = tmp_path / "c.pt"
    save_checkpoint(path, agent, ArenaConfig(frame_stack=4, frame_skip=3))
    meta = checkpoint_meta(path)
    assert meta["algorithm"] == "dqn"
    assert meta["arch_version"] == "alg-dqn-v1"
    assert meta["n_actions"] == 6
    assert meta["obs_shape"] == (5, 32, 20)
    assert meta["config"]["frame_stack"] == 4
    assert meta["config"]["frame_skip"] == 3


def test_checkpoint_meta_advanced_algorithm(tmp_path):
    agent = DoubleDQNPerDuelingAgent((5, 32, 20), 6)
    path = tmp_path / "advanced.pt"
    save_checkpoint(path, agent)
    meta = checkpoint_meta(path)
    assert meta["algorithm"] == "doubledqn_per_dueling"
    assert meta["arch_version"] == "alg-ddqn-per-dueling-v1"


def test_algorithm_from_checkpoint_resolves_package(tmp_path):
    path = tmp_path / "c.pt"
    save_checkpoint(path, DQNAgent((5, 32, 20), 6))
    alg, meta = algorithm_from_checkpoint(path)
    assert alg.Agent is DQNAgent
    assert meta["n_actions"] == 6
    assert alg is load_algorithm("dqn")


def test_old_checkpoint_without_algorithm_is_rejected(tmp_path):
    """Checkpoints da arquitetura antiga (12 ações, sem 'algorithm') não carregam."""
    path = tmp_path / "antigo.pt"
    torch.save({"arch_version": "cnn-dqn-v2", "n_actions": 12}, path)
    assert checkpoint_meta(path)["algorithm"] is None
    with pytest.raises(ValueError, match="algorithm"):
        algorithm_from_checkpoint(path)
    with pytest.raises(ValueError, match="None"):
        load_checkpoint(path, DQNAgent((5, 32, 20), 6))


def test_load_checkpoint_rejects_algorithm_mismatch_without_touching_weights(tmp_path):
    """Checkpoint e agente precisam ser do mesmo pacote alg_*; falha antes do load."""
    plain = DQNAgent((5, 32, 20), 6)
    advanced = DoubleDQNPerDuelingAgent((5, 32, 20), 6)
    plain_path = tmp_path / "plain.pt"
    advanced_path = tmp_path / "advanced.pt"
    save_checkpoint(plain_path, plain)
    save_checkpoint(advanced_path, advanced)
    with torch.no_grad():
        for p in plain.online.parameters():
            p.fill_(1.0)
        for p in advanced.online.parameters():
            p.fill_(1.0)
    with pytest.raises(ValueError, match="algoritmo"):
        load_checkpoint(advanced_path, plain)
    with pytest.raises(ValueError, match="algoritmo"):
        load_checkpoint(plain_path, advanced)
    # A validação acontece antes de qualquer cópia de pesos.
    assert all(torch.all(p == 1.0) for p in plain.online.parameters())
    assert all(torch.all(p == 1.0) for p in advanced.online.parameters())


def test_load_checkpoint_rejects_action_count_mismatch(tmp_path):
    path = tmp_path / "seis.pt"
    save_checkpoint(path, DQNAgent((5, 32, 20), 6))
    other = DQNAgent((5, 32, 20), 7)
    with torch.no_grad():
        for p in other.online.parameters():
            p.fill_(1.0)
    with pytest.raises(ValueError, match="ações"):
        load_checkpoint(path, other)
    assert all(torch.all(p == 1.0) for p in other.online.parameters())


def test_load_checkpoint_rejects_obs_shape_mismatch(tmp_path):
    path = tmp_path / "cinco.pt"
    save_checkpoint(path, DQNAgent((5, 32, 20), 6))
    other = DQNAgent((10, 32, 20), 6)  # ex.: frame_stack diferente
    with torch.no_grad():
        for p in other.online.parameters():
            p.fill_(1.0)
    with pytest.raises(ValueError, match="observação"):
        load_checkpoint(path, other)
    assert all(torch.all(p == 1.0) for p in other.online.parameters())


def test_load_checkpoint_rejects_corrupt_target_without_touching_online(tmp_path):
    """Um target corrompido não pode deixar a online parcialmente carregada."""
    path = tmp_path / "corrompido.pt"
    save_checkpoint(path, DQNAgent((5, 32, 20), 6))
    payload = torch.load(path, weights_only=False)
    payload["target"].pop(next(iter(payload["target"])))
    torch.save(payload, path)
    fresh = DQNAgent((5, 32, 20), 6)
    with torch.no_grad():
        for p in fresh.online.parameters():
            p.fill_(1.0)
    with pytest.raises(ValueError, match="target"):
        load_checkpoint(path, fresh)
    assert all(torch.all(p == 1.0) for p in fresh.online.parameters())


def test_apply_env_config():
    cfg = ArenaConfig()
    assert cfg.frame_stack == 1 and cfg.frame_skip == 1
    apply_env_config(cfg, {"config": {"frame_stack": 4, "frame_skip": 3}})
    assert cfg.frame_stack == 4 and cfg.frame_skip == 3
    # Checkpoint sem as chaves: mantém os defaults 1/1 de um cfg novo.
    cfg = ArenaConfig()
    apply_env_config(cfg, {"config": {}})
    assert cfg.frame_stack == 1 and cfg.frame_skip == 1


def test_try_load_checkpoint_skips_frame_stack_mismatch(tmp_path):
    from train import _try_load_checkpoint

    agent = DQNAgent((5, 32, 20), 6)
    path = tmp_path / "stacked.pt"
    save_checkpoint(path, agent, ArenaConfig(frame_stack=4))
    fresh = DQNAgent((5, 32, 20), 6)
    # Com stack diferente, a primeira conv tem outro número de canais: o
    # load estouraria no meio e deixaria pesos parcialmente sobrescritos.
    # O skip acontece antes, com aviso, sem tocar no agente.
    with pytest.warns(UserWarning):
        assert _try_load_checkpoint(path, fresh, frame_stack=1) is False
    assert _try_load_checkpoint(path, fresh, frame_stack=4) is True


def test_play_parse_args():
    from play import parse_args

    defaults = parse_args([])
    assert defaults.player is None
    assert defaults.opponent == "rule"
    assert defaults.record is None
    args = parse_args(["--player", "coins", "--record", "x.mp4"])
    assert args.player == "coins"
    assert args.record == "x.mp4"


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


@pytest.mark.parametrize("agent_cls", [DQNAgent, DoubleDQNPerDuelingAgent])
def test_checkpoint_lightweight(tmp_path, agent_cls):
    """Pesos-only: sem buffer, ambiente ou RNGs; contadores voltam no resume."""
    agent = agent_cls((5, 32, 20), 6)
    agent.steps = 7
    agent.epsilon = 0.25
    path = tmp_path / "c.pt"
    save_checkpoint(path, agent, extra={"a": 1}, episodes=5)
    payload = torch.load(path, weights_only=False)
    assert "buffer" not in payload and "env_state" not in payload and "rng_state" not in payload
    assert payload["episodes"] == 5
    agent2 = agent_cls((5, 32, 20), 6)
    load_checkpoint(path, agent2, resume=True)  # sem env
    assert agent2.steps == 7
    assert agent2.epsilon == 0.25


def test_try_load_skips_incompatible(tmp_path):
    from train import _try_load_checkpoint

    agent = DQNAgent((5, 32, 20), 6)
    junk = tmp_path / "old.pt"
    junk.write_bytes(b"lixo")  # não é um checkpoint válido
    with pytest.warns(UserWarning):
        assert _try_load_checkpoint(junk, agent) is False
    # e um checkpoint válido carrega: True
    good = tmp_path / "good.pt"
    save_checkpoint(good, agent)
    assert _try_load_checkpoint(good, agent) is True


def test_try_load_skips_checkpoint_of_other_algorithm(tmp_path):
    from train import _try_load_checkpoint

    advanced = DoubleDQNPerDuelingAgent((5, 32, 20), 6)
    path = tmp_path / "advanced.pt"
    save_checkpoint(path, advanced)
    plain = DQNAgent((5, 32, 20), 6)
    with torch.no_grad():
        for p in plain.online.parameters():
            p.fill_(0.5)
    with pytest.warns(UserWarning):
        assert _try_load_checkpoint(path, plain) is False
    assert all(torch.all(p == 0.5) for p in plain.online.parameters())


def test_auto_opponent_ignores_budget_subdir(tmp_path):
    from arena.environment import ArenaEnv
    from train import AutoOpponent

    (tmp_path / "budget").mkdir()
    (tmp_path / "budget" / "ep050.pt").write_bytes(b"lixo")
    (tmp_path / "best.pt").write_bytes(b"lixo")
    env = ArenaEnv(ArenaConfig(), opponent_policy="random")
    auto = AutoOpponent(env, load_algorithm("dqn"), checkpoint_dir=str(tmp_path))
    # Glob não-recursivo: snapshots de orçamento em budget/ não viram adversário.
    assert auto._checkpoints() == [tmp_path / "best.pt"]
    env.close()


def test_auto_opponent_init_from_disk_skips_incompatible(tmp_path):
    from arena.environment import ArenaEnv
    from train import AutoOpponent

    (tmp_path / "best.pt").write_bytes(b"lixo")  # ex.: best.pt de arquitetura antiga
    env = ArenaEnv(ArenaConfig(), opponent_policy="random")
    auto = AutoOpponent(env, load_algorithm("dqn"), checkpoint_dir=str(tmp_path))
    with pytest.warns(UserWarning):
        auto.init_from_disk()  # avisa e mantém os pesos iniciais, sem levantar
    env.close()

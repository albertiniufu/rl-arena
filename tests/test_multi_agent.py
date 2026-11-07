import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from arena.actions import JUMP, PUSH_RIGHT, STOP
from arena.config import ArenaConfig
from arena.engine import ArenaEngine, Coin
from arena.environment import ArenaEnv
from arena.multi_environment import ArenaMultiEnv
from arena.reward_shaping import coin_potential
from arena.scenario import Scenario
from algs.alg_ippo.agent import IPPOAgent
from algs.alg_registry import (
    apply_env_config,
    checkpoint_meta,
    load_checkpoint,
    save_checkpoint,
)

OBS_SHAPE = (6, 20, 32)


def make_env(n_agents=4, teams=None, **cfg_kwargs):
    if teams is None:
        scenario = Scenario.all_vs_all(n_agents)
    else:
        scenario = Scenario.teams(n_agents, teams)
    env = ArenaMultiEnv(ArenaConfig(**cfg_kwargs), scenario=scenario)
    return env, scenario


# ----------------------------------------------------------------- cenários
def test_scenario_all_vs_all_and_teams():
    solo = Scenario.all_vs_all(4)
    assert solo.team_ids == (0, 1, 2, 3)
    assert solo.n_teams == 4
    assert solo.enemies(0) == [1, 2, 3]
    assert solo.teammates(0) == []

    duplas = Scenario.teams(4, 2)
    assert duplas.team_ids == (0, 0, 1, 1)
    assert duplas.n_teams == 2
    assert duplas.teammates(0) == [1]
    assert duplas.enemies(0) == [2, 3]

    with pytest.raises(ValueError):
        Scenario.all_vs_all(1)
    with pytest.raises(ValueError):
        Scenario.teams(4, 3)   # não divisível
    with pytest.raises(ValueError):
        Scenario.teams(4, 1)   # precisa de pelo menos 2 times
    with pytest.raises(ValueError):
        Scenario(4, (0, 0, 1))  # tamanho errado de team_ids


def test_scenario_spawn_positions_keep_teams_apart():
    cfg = ArenaConfig()
    positions = Scenario.teams(4, 2).spawn_positions(cfg)
    assert len(positions) == 4
    xs = [p[0] for p in positions]
    assert all(p[1] == cfg.height - 80 for p in positions)
    assert len(set(xs)) == 4
    # Time 0 (agentes 0 e 1) fica à esquerda do time 1 (agentes 2 e 3).
    assert max(xs[0], xs[1]) < min(xs[2], xs[3])


# -------------------------------------------------------------- observação
def test_multi_observation_is_egocentric():
    env, _ = make_env(4, teams=2, frame_stack=1)
    env.reset(seed=0)
    for i, c in enumerate(env.engine.characters):
        c.x, c.y = 40.0 + i * 100.0, 320.0

    obs0 = env._observation(0)
    assert obs0[1].sum() == 1        # self
    assert obs0[5].sum() == 1        # aliado (agente 1)
    assert obs0[2].sum() == 2        # inimigos (2 e 3)

    obs2 = env._observation(2)
    assert obs2[1].sum() == 1        # self
    assert obs2[5].sum() == 1        # aliado (agente 3)
    assert obs2[2].sum() == 2        # inimigos (0 e 1)

    # Morto vai para o canal 4 e deixa de aparecer como aliado.
    env.engine.characters[1].alive = False
    obs0 = env._observation(0)
    assert obs0[5].sum() == 0
    assert obs0[4].sum() == 1
    env.close()


def test_multi_observation_in_all_vs_all_has_no_ally_channel():
    env, _ = make_env(3)
    env.reset(seed=0)
    obs0 = env._observation(0)
    assert obs0[5].sum() == 0
    assert obs0[2].sum() == 2
    env.close()


# ------------------------------------------------------------------- passo
def test_multi_step_shapes_rewards_and_time_truncation():
    env, _ = make_env(4, teams=2, episode_seconds=0.1, frame_skip=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (4, *OBS_SHAPE)
    assert env.observation_space.contains(obs[0])
    assert len(info["agents"]) == 4 and len(info["teams"]) == 2
    assert info["team_ids"] == (0, 0, 1, 1)

    truncated = np.zeros(4, dtype=bool)
    for _ in range(5):
        obs, rewards, terminated, truncated, info = env.step([STOP] * 4)
        assert obs.shape == (4, *OBS_SHAPE)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,) and not terminated.any()
        assert len(info["agents"]) == 4 and len(info["teams"]) == 2
        if truncated.all():
            break
    assert truncated.all()
    env.close()


def test_team_reward_is_mean_of_members():
    env, _ = make_env(4, teams=2)
    raw = np.array([1.0, 3.0, 5.0, 7.0])
    assert np.allclose(env._aggregate_rewards(raw), [2.0, 2.0, 6.0, 6.0])

    solo, _ = make_env(4)
    assert np.allclose(solo._aggregate_rewards(raw), raw)
    env.close()
    solo.close()


def test_push_ignores_teammates_and_penalizes_miss():
    env, _ = make_env(4, teams=2, frame_skip=1)
    env.reset(seed=2)
    actor = env.engine.characters[0]
    ally = env.engine.characters[1]
    enemy = env.engine.characters[2]
    actor.x, actor.y = 100.0, 320.0
    ally.x, ally.y = 100.0 + env.cfg.player_size + 1.0, 320.0
    enemy.x, enemy.y = 600.0, 320.0
    ally.vx = 0.0

    _, _, _, _, info = env.step([PUSH_RIGHT, STOP, STOP, STOP])
    # Aliado em contato não é alvo: o empurrão é contado como erro.
    assert ally.vx == pytest.approx(0.0)
    assert info["agents"][0]["raw_reward"] == pytest.approx(
        env.cfg.push_miss_penalty)

    # Controle: sem o filtro de time, o engine empurraria o aliado.
    env.engine.push(0, 1, 3)
    assert ally.vx > 0
    env.close()


def test_push_transfers_coins_from_target_to_pusher():
    engine = ArenaEngine(ArenaConfig(push_coin_transfer=5))
    engine.reset(seed=0, n_characters=2,
                 spawn_positions=[(100, 320), (128, 320)])
    actor, target = engine.characters
    target.coins = 12

    assert engine.push(0, 1, 1.0) is True
    assert (actor.coins, target.coins) == (5, 7)

    # Quem tem menos que a transferência passa o que tem, sem ficar negativo.
    target.coins = 3
    assert engine.push(0, 1, 1.0) is True
    assert (actor.coins, target.coins) == (8, 0)


def test_push_transfer_can_be_disabled():
    engine = ArenaEngine(ArenaConfig(push_coin_transfer=0))
    engine.reset(seed=0, n_characters=2,
                 spawn_positions=[(100, 320), (128, 320)])
    actor, target = engine.characters
    target.coins = 12

    assert engine.push(0, 1, 1.0) is True
    assert (actor.coins, target.coins) == (0, 12)


def test_multi_env_rewards_push_transfer_signed():
    env, _ = make_env(2, frame_skip=1)
    env.reset(seed=3)
    actor, target = env.engine.characters
    actor.x, actor.y = 100.0, 320.0
    target.x, target.y = 128.0, 320.0
    target.coins = 8
    env.engine.coins.clear()  # isola a recompensa da coleta de moedas

    _, rewards, _, _, info = env.step([PUSH_RIGHT, JUMP])
    assert (actor.coins, target.coins) == (5, 3)
    assert rewards[0] == pytest.approx(5.0)
    assert rewards[1] == pytest.approx(-5.0)
    assert info["agents"][0]["coins_transferred_this_step"] == 5
    assert info["agents"][1]["coins_transferred_this_step"] == -5
    env.close()


def test_coin_potential_uses_team_margin():
    engine = ArenaEngine(ArenaConfig())
    engine.reset(seed=0, n_characters=4,
                 spawn_positions=[(10, 320), (20, 320), (30, 320), (40, 320)])
    for character, coins in zip(engine.characters, [10, 2, 3, 1]):
        character.coins = coins

    team_ids = (0, 0, 1, 1)
    assert coin_potential(engine, 0, team_ids) == pytest.approx(8.0)
    assert coin_potential(engine, 2, team_ids) == pytest.approx(-8.0)
    # Sem times: o viewer contra todos os demais.
    assert coin_potential(engine, 0) == pytest.approx(10 - (2 + 3 + 1))


def test_pbrs_shapes_reward_with_coin_margin():
    env, _ = make_env(2, frame_skip=1, pbrs_coef=1.0, pbrs_gamma=0.99)
    env.reset(seed=3)
    actor, target = env.engine.characters
    actor.x, actor.y = 100.0, 320.0
    target.x, target.y = 128.0, 320.0
    target.coins = 8
    env.engine.coins.clear()

    _, rewards, _, _, info = env.step([PUSH_RIGHT, JUMP])
    # Densa: +5/-5 pela transferência. PBRS: gamma*Phi' - Phi.
    # Phi(ator): -8 -> 2; Phi(alvo): 8 -> -2.
    assert rewards[0] == pytest.approx(5.0 + (0.99 * 2 - (-8)))
    assert rewards[1] == pytest.approx(-5.0 + (0.99 * -2 - 8))
    assert info["agents"][0]["pbrs_this_step"] == pytest.approx(9.98)
    assert info["agents"][1]["pbrs_this_step"] == pytest.approx(-9.98)
    assert info["teams"][0]["pbrs"] == pytest.approx(9.98)
    assert info["teams"][1]["pbrs"] == pytest.approx(-9.98)
    env.close()


def test_single_env_pbrs_only_counts_balance_changes():
    env = ArenaEnv(ArenaConfig(opponent_count=1, frame_skip=1, pbrs_coef=1.0))
    env.reset(seed=1)
    player, opponent = env.engine.characters
    player.coins, opponent.coins = 2, 6
    env.engine.coins = [Coin(player.x, player.y - env.cfg.player_size / 2, 60)]

    _, reward, _, _, info = env.step(JUMP)
    # Coleta +1 (densa); Phi: -4 -> -3, então PBRS = 0.99*(-3) - (-4).
    assert reward == pytest.approx(1.0 + 1.03)
    assert info["pbrs_this_step"] == pytest.approx(1.03)
    env.close()


def test_engine_reset_accepts_explicit_spawn_positions():
    engine = ArenaEngine(ArenaConfig())
    engine.reset(seed=0, n_characters=3,
                 spawn_positions=[(10, 300), (200, 300), (400, 300)])
    assert len(engine.characters) == 3
    assert [(c.x, c.y) for c in engine.characters] == [
        (10.0, 300.0), (200.0, 300.0), (400.0, 300.0)]
    with pytest.raises(ValueError):
        engine.reset(seed=0, n_characters=3, spawn_positions=[(1, 2)])


# ---------------------------------------------------------------- algoritmo
def test_ippo_buffers_fill_together_then_update():
    agent = IPPOAgent(OBS_SHAPE, 6, n_agents=2, rollout_steps=4,
                      batch_size=4, epochs=1, device="cpu")
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    for _ in range(4):
        for i in range(2):
            agent.add_transition(i, obs, 0, 0.0, obs, False)
    assert len(agent.buffer) == 8
    loss = agent.learn()
    assert loss is not None
    assert agent.steps == 1
    assert len(agent.buffer) == 0

    # Com um buffer incompleto, learn() não atualiza.
    for _ in range(3):
        for i in range(2):
            agent.add_transition(i, obs, 1, 0.0, obs, False)
    assert agent.learn() is None
    assert agent.steps == 1


def test_ippo_act_batch_end_episode_and_finish():
    agent = IPPOAgent(OBS_SHAPE, 6, n_agents=3, rollout_steps=4,
                      batch_size=4, epochs=1, device="cpu")
    obs = np.zeros((3, *OBS_SHAPE), dtype=np.float32)
    assert agent.act_batch(obs).shape == (3,)
    assert agent.act_batch(obs, greedy=True).shape == (3,)

    for i in range(3):
        agent.add_transition(i, obs[i], 0, 0.0, obs[i], False)
    agent.end_episode()
    for i in range(3):
        *_, ends = agent.buffer.buffers[i].as_arrays()
        assert ends[-1]

    assert agent.finish_training() is not None
    assert len(agent.buffer) == 0


def test_ippo_checkpoint_round_trip_and_env_config(tmp_path):
    agent = IPPOAgent(OBS_SHAPE, 6, n_agents=4, team_ids=(0, 0, 1, 1),
                      device="cpu")
    cfg = ArenaConfig(n_agents=4, team_ids=(0, 0, 1, 1),
                      frame_stack=2, frame_skip=3)
    path = tmp_path / "ippo.pt"
    save_checkpoint(path, agent, cfg, {"multi_agent": True}, episodes=3)

    meta = checkpoint_meta(path)
    assert meta["algorithm"] == "ippo"
    restored = apply_env_config(ArenaConfig(), meta)
    assert restored.n_agents == 4
    assert restored.team_ids == (0, 0, 1, 1)
    assert restored.frame_stack == 2 and restored.frame_skip == 3

    other = IPPOAgent(OBS_SHAPE, 6, n_agents=4, team_ids=(0, 0, 1, 1),
                      device="cpu")
    load_checkpoint(path, other)
    with pytest.raises(ValueError):
        load_checkpoint(
            path, IPPOAgent((5, 20, 32), 6, n_agents=4, device="cpu"))


# --------------------------------------------------------------------- CLI
def test_train_cli_multi_agent_validation():
    from train import parse_args, validate_train_args

    args = parse_args(["--alg", "ippo"])
    assert args.agents == 4 and args.scenario == "all-vs-all"
    validate_train_args(args)

    validate_train_args(parse_args(
        ["--alg", "ippo", "--agents", "4", "--scenario", "teams",
         "--teams", "2"]))
    validate_train_args(parse_args(
        ["--alg", "ippo", "--agents", "6", "--scenario", "teams",
         "--teams", "3"]))

    for bad in (
        ["--alg", "dqn", "--agents", "4"],
        ["--alg", "ippo", "--agents", "3", "--scenario", "teams",
         "--teams", "2"],
        ["--alg", "ippo", "--agents", "4", "--scenario", "teams",
         "--teams", "1"],
        ["--alg", "ippo", "--envs", "2"],
        ["--alg", "ippo", "--opponent", "rule"],
        ["--alg", "ippo", "--overlap"],
        ["--alg", "dqn", "--scenario", "teams"],
    ):
        with pytest.raises(SystemExit):
            validate_train_args(parse_args(bad))

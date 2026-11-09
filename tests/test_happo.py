"""Testes do cenário coletores × empurradores e do HAPPO.

Cobre a mecânica nova (papéis, empurrão de meia arena, recarga de 2 s) e o
algoritmo heterogêneo (máscara de ações, atores independentes, update
sequencial, checkpoint e CLI).
"""
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from arena.actions import PUSH_INTENSITY, PUSH_RIGHT, STOP
from arena.config import (
    COLLECTOR_ROLE,
    PUSHER_ROLE,
    ArenaConfig,
)
from arena.engine import ArenaEngine
from arena.multi_environment import ArenaMultiEnv
from arena.scenario import Scenario
from algs.alg_happo.agent import HAPPOAgent
from algs.alg_happo.model import HAPPONetwork
from algs.alg_happo.rollout_buffer import HAPPORolloutBuffer
from algs.alg_registry import (
    apply_env_config,
    available_algorithms,
    checkpoint_meta,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)

OBS_SHAPE = (6, 20, 32)


def make_agent(n_agents=2, roles=None, **kwargs):
    params = dict(n_agents=n_agents, rollout_steps=2, batch_size=4,
                  epochs=1, device="cpu")
    params.update(kwargs)
    if roles is not None:
        params["agent_roles"] = roles
    return HAPPOAgent(OBS_SHAPE, 6, **params)


def fill_step(agent, step=0, reward=0.0):
    obs = np.full((agent.n_agents, *OBS_SHAPE), step, dtype=np.float32)
    for i in range(agent.n_agents):
        agent.add_transition(i, obs[i], 0, reward, obs[i], False)


def ratio_scenario(**cfg_kwargs):
    """Cenário com a física do empurrão forte já calibrada."""
    scenario = Scenario.collectors_vs_pushers(4, 2)
    cfg = ArenaConfig(**cfg_kwargs)
    cfg.push_impulse = cfg.push_impulse_for_range(
        cfg.width / 2.0, PUSH_INTENSITY)
    cfg.push_speed_cap = cfg.push_impulse * PUSH_INTENSITY * 1.1
    cfg.push_cooldown = 2.0
    cfg.agent_roles = scenario.roles
    return scenario, cfg


# ----------------------------------------------------------------- cenário
def test_collectors_vs_pushers_mixes_roles_inside_each_team():
    scenario = Scenario.collectors_vs_pushers(4, 2)
    assert scenario.team_ids == (0, 0, 1, 1)
    # Cada time tem 1 coletor + 1 empurrador (papéis alternam por time).
    assert scenario.roles == (
        COLLECTOR_ROLE, PUSHER_ROLE, COLLECTOR_ROLE, PUSHER_ROLE)
    assert scenario.has_roles
    assert scenario.collectors() == [0, 2]
    assert scenario.pushers() == [1, 3]
    assert scenario.can_push(0) is False
    assert scenario.can_collect(0) is True
    assert scenario.can_push(1) is True
    assert scenario.can_collect(1) is False
    assert scenario.can_push(2) is False
    assert scenario.can_collect(3) is False

    # Simétrico continua sem papéis.
    assert Scenario.teams(4, 2).role_of(0) is None
    assert Scenario.teams(4, 2).can_push(0) is True

    # O tamanho do time precisa ser par; o número de times não.
    with pytest.raises(ValueError):
        Scenario.collectors_vs_pushers(4, 3)   # não divisível
    with pytest.raises(ValueError):
        Scenario.collectors_vs_pushers(5, 2)   # não divisível
    with pytest.raises(ValueError):
        Scenario.collectors_vs_pushers(6, 2)   # time de 3 (ímpar)
    six = Scenario.collectors_vs_pushers(6, 3)
    assert six.team_ids == (0, 0, 1, 1, 2, 2)
    assert six.roles == (
        COLLECTOR_ROLE, PUSHER_ROLE, COLLECTOR_ROLE, PUSHER_ROLE,
        COLLECTOR_ROLE, PUSHER_ROLE)
    with pytest.raises(ValueError):
        Scenario(2, (0, 1), ("collector",))    # tamanho errado


def test_each_team_has_one_collector_and_one_pusher():
    scenario = Scenario.collectors_vs_pushers(4, 2)
    for team in (0, 1):
        members = [i for i, t in enumerate(scenario.team_ids) if t == team]
        assert sorted(scenario.role_of(i) for i in members) == [
            COLLECTOR_ROLE, PUSHER_ROLE]
    positions = scenario.spawn_positions(ArenaConfig())
    xs = [p[0] for p in positions]
    assert len(set(xs)) == 4
    assert max(xs[0], xs[1]) < min(xs[2], xs[3])


# ------------------------------------------------------------------ engine
def test_collector_cannot_push_and_pusher_cannot_collect():
    _, cfg = ratio_scenario()
    engine = ArenaEngine(cfg)
    engine.reset(seed=0, n_characters=4,
                 spawn_positions=[(100, 320), (128, 320),
                                  (400, 320), (428, 320)])
    # Papéis mistos: time 0 = (coletor, empurrador); time 1 = (coletor, empurrador).
    collector0, pusher1, collector2, pusher3 = engine.characters

    # Coletor 0 tenta empurrar o empurrador 1 (inimigo): ação inerte.
    engine.step([PUSH_RIGHT, STOP, STOP, STOP], 1 / 30)
    assert pusher1.vx == pytest.approx(0.0)
    assert collector0.push_cooldown == 0.0

    # Empurrador 1 empurra o coletor 0 (inimigo) normalmente.
    pusher1.x, pusher1.y = 100.0, 320.0
    collector0.x, collector0.y = 128.0, 320.0
    pusher1.vx = collector0.vx = 0.0
    engine.step([STOP, PUSH_RIGHT, STOP, STOP], 1 / 30)
    assert collector0.vx > 0
    assert pusher1.push_cooldown == pytest.approx(cfg.push_cooldown)

    # Moeda em cima de cada um: só coletores coletam do mapa.
    from arena.engine import Coin
    collector0.x, collector0.y = 200.0, 320.0
    pusher1.x, pusher1.y = 300.0, 320.0
    collector0.vx = pusher1.vx = 0.0
    collector0.coins = pusher1.coins = 0
    engine.coins = [Coin(collector0.x, collector0.y - 12, 60.0),
                    Coin(pusher1.x, pusher1.y - 12, 60.0)]
    engine.step([STOP] * 4, 1 / 30)
    assert collector0.coins == 1
    assert pusher1.coins == 0


def test_strong_push_moves_half_arena_and_waits_two_seconds():
    _, cfg = ratio_scenario()
    engine = ArenaEngine(cfg)
    engine.reset(seed=0, n_characters=2,
                 spawn_positions=[(100, 320), (128, 320)])
    pusher, target = engine.characters
    start = target.x

    engine.step([PUSH_RIGHT, STOP], 1 / 30)
    assert pusher.push_cooldown == pytest.approx(2.0)
    for _ in range(59):
        engine.step([STOP, STOP], 1 / 30)
    moved = target.x - start
    assert moved == pytest.approx(cfg.width / 2.0, rel=0.05)

    # A recarga acabou (2 s = 60 substeps de 1/30): empurra de novo.
    # Reposiciona para não medir o efeito da parede.
    target.x = 128.0
    target.vx = 0.0
    engine.step([PUSH_RIGHT, STOP], 1 / 30)
    assert target.vx > 0
    assert pusher.push_cooldown == pytest.approx(2.0)


def test_push_is_blocked_while_cooldown_active():
    _, cfg = ratio_scenario()
    engine = ArenaEngine(cfg)
    engine.reset(seed=0, n_characters=2,
                 spawn_positions=[(100, 320), (128, 320)])
    pusher, target = engine.characters
    engine.step([PUSH_RIGHT, STOP], 1 / 30)
    vx_after_first = target.vx

    # Um substep depois ainda há ~2 s de recarga; o vx não recebe impulso.
    engine.step([PUSH_RIGHT, STOP], 1 / 30)
    assert target.vx < vx_after_first  # só o atrito, nenhum impulso novo
    assert pusher.push_cooldown > 1.9


# -------------------------------------------------------------- multi env
def test_multi_env_coerces_collector_push_and_reports_role():
    scenario, cfg = ratio_scenario(episode_seconds=1.0, frame_skip=1)
    env = ArenaMultiEnv(cfg, scenario=scenario)
    env.reset(seed=0)
    # Coletor 0 com o empurrador 1 (inimigo) em contato: mesmo pedindo
    # PUSH_RIGHT, o ambiente executa STOP e não move o alvo.
    actor, target = env.engine.characters[0], env.engine.characters[1]
    actor.x, actor.y = 100.0, 320.0
    target.x, target.y = 128.0, 320.0
    target.vx = 0.0
    _, _, _, _, info = env.step([PUSH_RIGHT, STOP, STOP, STOP])
    assert target.vx == pytest.approx(0.0)
    assert [a["role"] for a in info["agents"]] == [
        COLLECTOR_ROLE, PUSHER_ROLE, COLLECTOR_ROLE, PUSHER_ROLE]
    env.close()


# ------------------------------------------------------------------ modelo
def test_happo_has_independent_actors_and_centralized_critic():
    net = HAPPONetwork(OBS_SHAPE, 6, n_agents=3)
    assert len(net.actors) == 3
    assert net.actors[0] is not net.actors[1]
    values = net.joint_values(torch.zeros(1, 3, *OBS_SHAPE))
    assert values.shape == (1, 3)
    # O V do agente 0 depende da observação dos outros (crítico central).
    changed = torch.zeros(1, 3, *OBS_SHAPE)
    changed[:, 1] += 1.0
    assert not torch.allclose(values[:, 0], net.joint_values(changed)[:, 0])
    assert net.actor_logits(torch.zeros(2, *OBS_SHAPE), 2).shape == (2, 6)
    assert hasattr(net, "features")  # telemetria do train.py


def test_happo_buffer_assembles_joint_steps():
    buffer = HAPPORolloutBuffer(2)
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    assert buffer.as_arrays() is None
    buffer.add(0, obs, 1, 0.0, obs, False)
    assert buffer.n_steps == 0 and len(buffer) == 1
    with pytest.raises(ValueError):
        buffer.add(0, obs, 1, 0.0, obs, False)
    buffer.add(1, obs, 2, 1.0, obs, True)
    assert buffer.n_steps == 1
    buffer.mark_episode_end(1)
    arrays = buffer.as_arrays()
    assert arrays["states"].shape == (1, 2, *OBS_SHAPE)
    assert arrays["actions"].tolist() == [[1, 2]]
    assert arrays["episode_ends"][0].tolist() == [False, True]
    buffer.clear()
    assert len(buffer) == 0 and buffer.as_arrays() is None


# ------------------------------------------------------------------ agente
def test_happo_masks_push_actions_for_collectors():
    scenario = Scenario.collectors_vs_pushers(4, 2)
    agent = make_agent(n_agents=4, roles=scenario.roles)
    # Papéis mistos: 0 e 2 coletores; 1 e 3 empurradores.
    assert agent.action_masks[0].tolist() == [
        True, True, True, True, False, False]
    assert agent.action_masks[1].tolist() == [True] * 6
    assert agent.action_masks[2].tolist() == [
        True, True, True, True, False, False]
    assert agent.action_masks[3].tolist() == [True] * 6
    obs = np.zeros((4, *OBS_SHAPE), dtype=np.float32)
    # Com os logits mascarados a -1e9, a amostragem nunca escolhe 4/5.
    for _ in range(20):
        actions = agent.act_batch(obs)
        assert actions[0] not in (4, 5)
        assert actions[2] not in (4, 5)
    assert agent.act_batch(obs, greedy=True).shape == (4,)


def test_happo_learn_waits_then_updates_independent_actors():
    # 2 agentes, 2 times: cenário manual porque o cenário misto exige
    # time de tamanho par (1 coletor + 1 empurrador dentro do time).
    scenario = Scenario(2, (0, 1), (COLLECTOR_ROLE, PUSHER_ROLE))
    agent = make_agent(n_agents=2, roles=scenario.roles,
                       rollout_steps=3, batch_size=4)
    # Não há parâmetro compartilhado entre os atores.
    p0 = next(agent.online.actors[0].parameters())
    p1 = next(agent.online.actors[1].parameters())
    assert p0.data_ptr() != p1.data_ptr()
    before0 = [p.detach().clone()
               for p in agent.online.actors[0].parameters()]
    before1 = [p.detach().clone()
               for p in agent.online.actors[1].parameters()]

    fill_step(agent, 0)
    fill_step(agent, 1)
    assert agent.learn() is None
    assert agent.steps == 0
    fill_step(agent, 2)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert agent.steps == 1 and len(agent.buffer) == 0
    assert agent.last_grad_norm >= 0.0
    # Cada ator recebeu o seu próprio passo do update sequencial.
    assert any(not torch.allclose(a, b) for a, b in zip(
        before0, agent.online.actors[0].parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(
        before1, agent.online.actors[1].parameters()))


def test_happo_act_episode_end_and_finish():
    scenario = Scenario.collectors_vs_pushers(4, 2)
    agent = make_agent(n_agents=4, roles=scenario.roles, rollout_steps=4)
    obs = np.zeros((4, *OBS_SHAPE), dtype=np.float32)
    assert agent.act_batch(obs).shape == (4,)
    assert len(agent.q_stats(obs[0])) == 2
    assert len(agent.q_stats(obs)) == 2
    with pytest.raises(ValueError):
        agent.q_stats(np.zeros((2, *OBS_SHAPE), dtype=np.float32))

    fill_step(agent)
    agent.end_episode()
    assert agent.buffer.as_arrays()["episode_ends"][-1].all()
    assert agent.finish_training() is not None
    assert len(agent.buffer) == 0


def test_happo_update_with_value_clip():
    agent = make_agent(n_agents=2, value_clip=0.2)
    fill_step(agent)
    fill_step(agent, 1)
    assert agent.learn() is not None


def test_happo_trains_in_real_role_env():
    scenario, cfg = ratio_scenario(episode_seconds=0.3, frame_skip=1)
    env = ArenaMultiEnv(cfg, scenario=scenario)
    agent = HAPPOAgent(
        env.observation_space.shape, env.action_space.n, n_agents=4,
        team_ids=scenario.team_ids, agent_roles=scenario.roles,
        rollout_steps=4, batch_size=8, epochs=2, device="cpu")
    obs, _ = env.reset(seed=0)
    updates = 0
    while True:
        actions = agent.act_batch(obs)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(4):
            agent.add_transition(
                i, obs[i], int(actions[i]), float(rewards[i]),
                next_obs[i], bool(terminated[i]))
        if agent.learn() is not None:
            updates += 1
        obs = next_obs
        if bool(np.all(truncated)):
            break
    agent.end_episode(info)
    assert updates >= 1
    env.close()


# -------------------------------------------------------------- checkpoint
def test_happo_checkpoint_round_trip_and_env_config(tmp_path):
    scenario = Scenario.collectors_vs_pushers(4, 2)
    agent = HAPPOAgent(OBS_SHAPE, 6, n_agents=4,
                       team_ids=scenario.team_ids,
                       agent_roles=scenario.roles, device="cpu")
    cfg = ArenaConfig(n_agents=4, team_ids=scenario.team_ids,
                      agent_roles=scenario.roles, frame_stack=2,
                      frame_skip=3, push_impulse=436.36,
                      push_speed_cap=1200.0, push_cooldown=2.0,
                      push_coin_transfer=15)
    path = tmp_path / "happo.pt"
    save_checkpoint(path, agent, cfg, {"multi_agent": True}, episodes=3)

    meta = checkpoint_meta(path)
    assert meta["algorithm"] == "happo"
    restored = apply_env_config(ArenaConfig(), meta)
    assert restored.n_agents == 4
    assert restored.team_ids == scenario.team_ids
    assert restored.agent_roles == scenario.roles
    assert restored.push_impulse == pytest.approx(436.36)
    assert restored.push_cooldown == pytest.approx(2.0)
    assert restored.push_coin_transfer == 15
    assert restored.push_coin_reward == pytest.approx(1.0)

    other = HAPPOAgent(OBS_SHAPE, 6, n_agents=4,
                       team_ids=scenario.team_ids,
                       agent_roles=scenario.roles, device="cpu")
    load_checkpoint(path, other)
    with pytest.raises(ValueError):
        load_checkpoint(path, HAPPOAgent(
            (5, 20, 32), 6, n_agents=4, team_ids=scenario.team_ids,
            agent_roles=scenario.roles, device="cpu"))


# --------------------------------------------------------------------- CLI
def test_happo_algorithm_discovery():
    names = available_algorithms()
    assert "happo" in names
    assert load_algorithm("happo").Agent is HAPPOAgent


def test_train_cli_accepts_happo_role_scenario():
    from train import parse_args, validate_train_args

    args = parse_args(["--alg", "happo", "--scenario",
                       "collectors-vs-pushers", "--teams", "2"])
    assert args.agents == 4
    validate_train_args(args)
    # O número de times não precisa ser par; o tamanho do time sim.
    validate_train_args(parse_args(
        ["--alg", "happo", "--scenario", "collectors-vs-pushers",
         "--agents", "6", "--teams", "3"]))
    with pytest.raises(SystemExit):
        validate_train_args(parse_args(
            ["--alg", "happo", "--scenario", "collectors-vs-pushers",
             "--agents", "6", "--teams", "2"]))
    # Cenário com papéis continua exigindo algoritmo multiagente.
    with pytest.raises(SystemExit):
        validate_train_args(parse_args(
            ["--alg", "ppo", "--scenario", "collectors-vs-pushers"]))

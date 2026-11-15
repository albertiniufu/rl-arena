"""Cenário assimétrico coletores × empurradores e papéis por agente."""
import numpy as np
import pytest

from arenas.algs.alg_happo.agent import HAPPOAgent
from arenas.seek_coin.arena.actions import (
    PUSH_INTENSITY,
    PUSH_RIGHT,
    STOP,
)
from arenas.seek_coin.arena.config import (
    COLLECTOR_ROLE,
    PUSHER_ROLE,
    ArenaConfig,
)
from arenas.seek_coin.arena.engine import ArenaEngine, Coin
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario

OBS_SHAPE = (6, 20, 32)


def ratio_scenario(**cfg_kwargs):
    """Cenário e física do empurrão forte já calibrados."""
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

    # Simétrico continua sem papéis.
    assert Scenario.teams(4, 2).role_of(0) is None
    assert Scenario.teams(4, 2).can_push(0) is True
    assert Scenario.teams(4, 2).can_collect(0) is True

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
    with pytest.raises(ValueError):
        Scenario(2, (0, 1), ("coletor", "coletor"))  # papel inválido


def test_from_config_round_trips_roles():
    scenario = Scenario.collectors_vs_pushers(4, 2)
    cfg = ArenaConfig(
        n_agents=scenario.n_agents,
        team_ids=tuple(scenario.team_ids),
        agent_roles=tuple(scenario.roles),
    )
    rebuilt = Scenario.from_config(cfg)
    assert rebuilt == scenario
    # Checkpoint/config sem papéis continua simétrico.
    assert Scenario.from_config(ArenaConfig(n_agents=2)).roles == ()


# ------------------------------------------------------------------ engine
def test_collector_cannot_push_and_pusher_cannot_collect():
    _, cfg = ratio_scenario()
    engine = ArenaEngine(cfg)
    engine.reset(seed=0, n_characters=4,
                 spawn_positions=[(100, 320), (128, 320),
                                  (400, 320), (428, 320)])
    collector0, pusher1, _, _ = engine.characters

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


# ------------------------------------------------------------------ agente
def test_happo_masks_push_actions_for_collectors():
    scenario = Scenario.collectors_vs_pushers(4, 2)
    agent = HAPPOAgent(
        OBS_SHAPE, 6, n_agents=4, agent_roles=scenario.roles,
        rollout_steps=2, batch_size=4, epochs=1, device="cpu")
    # Papéis mistos: 0 e 2 coletores; 1 e 3 empurradores.
    assert agent.action_masks[0].tolist() == [
        True, True, True, True, False, False]
    assert agent.action_masks[1].tolist() == [True] * 6
    assert agent.action_masks[2].tolist() == [
        True, True, True, True, False, False]
    assert agent.action_masks[3].tolist() == [True] * 6

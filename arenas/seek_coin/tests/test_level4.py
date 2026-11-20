"""Testes do nível 4: visão local (recorte egocêntrico da observação)."""
import numpy as np
import pytest

from arenas.algs.alg_registry import apply_env_config
from arenas.seek_coin.arena.actions import STOP
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.engine import ArenaEngine, Coin, SuperCoin
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.arena.levels import (
    BUTTONS,
    STATIC_PLATFORMS,
    get_hazard_kinds,
    get_level_channel_names,
    get_super_coin_position,
    uses_local_view,
)
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.observations import (
    LOCAL_VIEW_PAD,
    observation_grid_shape,
    render_grid,
)
from arenas.seek_coin.arena.scenario import Scenario
from arenas.seek_coin.play import parse_args as parse_play_args
from arenas.seek_coin.train import (
    parse_args as parse_train_args,
    validate_train_args,
)

VIEW = 9
CENTER = VIEW // 2
# Nível 4 = nível 3: 5 canais base + botões/super moeda.
BASE_CHANNELS = 5
# Nível 4 desliga botões/super moeda por padrão: só os canais base.
N_CHANNELS = BASE_CHANNELS
# Com a mecânica ligada, entram os 3 canais do nível 3.
SUPER_CHANNELS = BASE_CHANNELS + len(get_level_channel_names(4))
SUPER_CHANNEL = SUPER_CHANNELS - 1  # super_coin


def _place(engine, x=300.0, y=347.5, platform_index=0):
    character = engine.characters[0]
    character.x, character.y = x, y
    character.vx = character.vy = 0.0
    character.alive = True
    character.on_ground = True
    character.platform_index = platform_index
    return character


def _frame(env):
    """Último frame da pilha, sem os frames repetidos do reset."""
    return env._observation()


def test_level_four_reuses_the_level_three_world():
    assert STATIC_PLATFORMS[4] == STATIC_PLATFORMS[3]
    assert BUTTONS[4] == BUTTONS[3]
    assert get_super_coin_position(4) == get_super_coin_position(3)
    assert get_hazard_kinds(4) == ()
    assert get_level_channel_names(4) == (
        "activate_button", "cancel_button", "super_coin")


def test_local_view_levels():
    # Nível 4 (mapa fixo) e nível 5 (mundo procedural) usam recorte local.
    assert [level for level in range(6) if uses_local_view(level)] == [4, 5]


def test_observation_grid_shape():
    assert observation_grid_shape(ArenaConfig(level=3)) == (20, 32)
    assert observation_grid_shape(ArenaConfig(level=4)) == (VIEW, VIEW)
    assert observation_grid_shape(
        ArenaConfig(level=4, view_size=7)) == (7, 7)


@pytest.mark.parametrize("size", [1, 2, 8, 10, 0, -3])
def test_invalid_view_size_is_rejected(size):
    with pytest.raises(ValueError):
        ArenaEnv(ArenaConfig(level=4, view_size=size),
                 opponent_policy="random")


def test_env_observation_shapes_single_and_multi():
    single = ArenaEnv(ArenaConfig(level=4, frame_stack=2),
                      opponent_policy="random")
    assert single.observation_space.shape == (2 * N_CHANNELS, VIEW, VIEW)
    obs, _ = single.reset(seed=0)
    assert obs.shape == single.observation_space.shape
    single.close()

    multi = ArenaMultiEnv(ArenaConfig(level=4, frame_stack=2),
                          scenario=Scenario.all_vs_all(2))
    assert multi.observation_space.shape == (2 * 6, VIEW, VIEW)
    obs, _ = multi.reset(seed=0)
    assert obs.shape == (2, *multi.observation_space.shape)
    multi.close()


def test_super_coin_channels_only_when_enabled():
    off = ArenaEnv(ArenaConfig(level=4), opponent_policy="random")
    assert off.engine.buttons == []
    assert off.engine.super_coin_position is None
    on = ArenaEnv(ArenaConfig(level=4, super_coin_enabled=True),
                  opponent_policy="random")
    assert len(on.engine.buttons) == 2
    assert on.observation_space.shape == (SUPER_CHANNELS, VIEW, VIEW)
    off.close()
    on.close()

    level3 = ArenaEnv(ArenaConfig(level=3), opponent_policy="random")
    assert len(level3.engine.buttons) == 2
    assert level3.observation_space.shape == (SUPER_CHANNELS, 20, 32)
    level3_off = ArenaEnv(ArenaConfig(level=3, super_coin_enabled=False),
                          opponent_policy="random")
    assert level3_off.engine.buttons == []
    assert level3_off.observation_space.shape == (N_CHANNELS, 20, 32)
    level3.close()
    level3_off.close()


def test_agent_is_always_at_the_center_of_its_crop():
    env = ArenaEnv(ArenaConfig(level=4), opponent_policy="random")
    env.reset(seed=0)
    for x in (12.5, 90.0, 300.0, 627.5):
        _place(env.engine, x=x)
        frame = _frame(env)
        cells = np.argwhere(frame[1])
        assert len(cells) == 1
        assert tuple(cells[0]) == (CENTER, CENTER)


def test_outside_the_map_is_padded_with_the_fixed_value():
    env = ArenaEnv(ArenaConfig(level=4), opponent_policy="random")
    env.reset(seed=0)
    _place(env.engine, x=12.5)  # encostado na borda esquerda do mapa
    frame = _frame(env)
    # A janela vai de -4 a +4 colunas; as 4 primeiras estão fora do mapa.
    assert frame[:, :, :CENTER].sum() == 0
    assert np.all(frame[:, :, :CENTER] == LOCAL_VIEW_PAD)
    # A parte dentro do mapa continua desenhada (chão do nível).
    assert frame[0].sum() > 0


def test_entities_outside_the_crop_are_invisible():
    env = ArenaEnv(ArenaConfig(level=4), opponent_policy="random")
    env.reset(seed=0)
    _place(env.engine, x=300.0)
    env.engine.coins = [Coin(300.0, 340.0, 60.0)]
    inside = _frame(env)
    assert inside[3].sum() == 1
    env.engine.coins = [Coin(600.0, 340.0, 60.0)]
    outside = _frame(env)
    assert outside[3].sum() == 0


def test_level_four_does_not_activate_super_coin_by_default():
    env = ArenaEnv(ArenaConfig(level=4, frame_skip=1),
                   opponent_policy="random")
    env.reset(seed=1)
    _place(env.engine)
    env.step(STOP)
    assert env.engine.super_coin is None
    assert env.engine.super_coin_position is None


def test_level_four_activates_when_super_coin_is_enabled():
    env = ArenaEnv(ArenaConfig(level=4, frame_skip=1,
                               super_coin_enabled=True),
                   opponent_policy="random")
    env.reset(seed=1)
    _place(env.engine, x=50.0)
    env.step(STOP)
    assert env.engine.super_coin is not None
    env.close()


def test_super_coin_channel_is_cropped_too():
    env = ArenaEnv(ArenaConfig(level=4, super_coin_enabled=True),
                   opponent_policy="random")
    env.reset(seed=0)
    env.engine.super_coin = SuperCoin(610.0, 342.0, 12.0)
    _place(env.engine, x=610.0)
    assert _frame(env)[SUPER_CHANNEL].sum() > 0  # super moeda visível
    _place(env.engine, x=90.0)
    assert _frame(env)[SUPER_CHANNEL].sum() == 0  # fora do recorte
    assert env.engine.super_coin is not None
    env.close()


def test_multi_agent_has_egocentric_crops():
    env = ArenaMultiEnv(ArenaConfig(level=4),
                        scenario=Scenario.all_vs_all(2))
    env.reset(seed=0)
    engine = env.engine
    engine.characters[0].x = 100.0
    engine.characters[1].x = 500.0
    engine.coins = [Coin(500.0, 340.0, 60.0)]
    obs = np.stack([env._observation(i) for i in range(2)])
    assert tuple(np.argwhere(obs[0][1])[0]) == (CENTER, CENTER)
    assert tuple(np.argwhere(obs[1][1])[0]) == (CENTER, CENTER)
    # A moeda perto do agente 1 não aparece no recorte do agente 0.
    assert obs[0][3].sum() == 0
    assert obs[1][3].sum() == 1
    env.close()


def test_seed_reproducibility():
    cfg = ArenaConfig(level=4)
    first = ArenaEnv(cfg, opponent_policy="random")
    second = ArenaEnv(cfg, opponent_policy="random")
    obs_a, _ = first.reset(seed=11)
    obs_b, _ = second.reset(seed=11)
    assert np.array_equal(obs_a, obs_b)
    for _ in range(5):
        obs_a, _, _, _, _ = first.step(STOP)
        obs_b, _, _, _, _ = second.step(STOP)
    assert np.array_equal(obs_a, obs_b)
    first.close()
    second.close()


def test_observation_does_not_leak_global_state():
    env = ArenaEnv(ArenaConfig(level=4), opponent_policy="random")
    env.reset(seed=0)
    _place(env.engine, x=100.0)
    far_frame = _frame(env)
    # Colocar um adversário do outro lado do mapa não muda o recorte.
    env.engine.characters[1].x = 600.0
    env.engine.characters[1].y = 347.5
    assert np.array_equal(_frame(env), far_frame)
    # ... mas um adversário dentro da janela aparece no canal 2.
    env.engine.characters[1].x = 120.0
    assert _frame(env)[2].sum() == 1


def test_checkpoint_config_applies_level_view_size_and_super_coin():
    cfg = ArenaConfig(level=0, view_size=9)
    apply_env_config(cfg, {"config": {
        "level": 4, "view_size": 7, "super_coin_enabled": True}})
    assert cfg.level == 4
    assert cfg.view_size == 7
    assert cfg.super_coin_enabled is True
    env = ArenaEnv(cfg, opponent_policy="random")
    assert env.observation_space.shape == (SUPER_CHANNELS, 7, 7)
    env.close()

    default = ArenaConfig(level=0)
    apply_env_config(default, {"config": {"level": 4}})
    assert default.super_coin_enabled is None
    env = ArenaEnv(default, opponent_policy="random")
    assert env.observation_space.shape == (N_CHANNELS, VIEW, VIEW)
    env.close()


def test_train_and_play_cli_accept_level_four():
    train = parse_train_args(["--level", "4", "--view-size", "7"])
    assert train.level == 4 and train.view_size == 7
    assert train.super_coin is None
    assert parse_train_args(["--super-coin"]).super_coin is True
    assert parse_train_args(["--no-super-coin"]).super_coin is False
    play = parse_play_args(["--level", "4", "--view-size", "7"])
    assert play.level == 4 and play.view_size == 7
    assert play.super_coin is None


@pytest.mark.parametrize("size", [2, 4, 1, 0])
def test_train_cli_rejects_invalid_view_size(size):
    args = parse_train_args(["--level", "4", "--view-size", str(size)])
    with pytest.raises(SystemExit):
        validate_train_args(args)


def test_gpu_env_rejects_local_view():
    torch = pytest.importorskip("torch")  # noqa: F841
    from arenas.seek_coin.arena.gpu_env import GpuArenaVecEnv

    with pytest.raises(NotImplementedError):
        GpuArenaVecEnv(ArenaConfig(level=4, n_agents=2,
                                   team_ids=(0, 1)),
                       n_envs=1, n_agents=2, team_ids=(0, 1),
                       device="cpu")


def test_single_agent_smoke_runs_a_full_episode():
    cfg = ArenaConfig(level=4, episode_seconds=0.5, frame_skip=2)
    env = ArenaEnv(cfg, opponent_policy="rule")
    obs, _ = env.reset(seed=3)
    steps = 0
    while True:
        obs, _, terminated, truncated, info = env.step(STOP)
        steps += 1
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break
    assert steps > 0 and truncated
    env.close()

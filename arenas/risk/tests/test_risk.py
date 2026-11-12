"""Testes do tabuleiro, do motor, do bot e do ambiente de conquista."""
import numpy as np
import pytest

from arenas.risk.agents import RiskBot
from arenas.risk.board import (
    ADJACENCY,
    ADJACENCY_INDEX,
    CONTINENT_BONUS,
    CONTINENT_TERRITORIES,
    TERRITORIES,
    TERRITORY_INDEX,
)
from arenas.risk.env import RiskEnv
from arenas.risk.game import RiskGame


def _scenario(src_name="Alaska", dst_name="Yukon"):
    game = RiskGame(n_players=2, seed=0)
    src = TERRITORY_INDEX[src_name]
    dst = TERRITORY_INDEX[dst_name]
    game.owner[:] = 1
    game.owner[src] = 0
    game.armies[:] = 1
    game.armies[src] = 5
    game.armies[dst] = 2
    game.current_player = 0
    game.phase = "attack"
    game.reinforcements_left = 0
    game.winner = None
    game.fortified = False
    return game, src, dst


class FixedRng:
    """RNG falso que devolve dados na ordem dada."""

    def __init__(self, values):
        self.values = list(values)

    def integers(self, low, high, size=None):
        assert size is not None
        return np.array([self.values.pop(0) for _ in range(size)])


def test_board_adjacency_is_symmetric_and_connected():
    for name, neighbors in ADJACENCY.items():
        for neighbor in neighbors:
            assert name in ADJACENCY[neighbor]
    start = TERRITORIES[0]
    seen = {start}
    stack = [start]
    while stack:
        for neighbor in ADJACENCY[stack.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    assert seen == set(TERRITORIES)


def test_reinforcement_count_and_continent_bonus():
    game = RiskGame(n_players=2, seed=0)
    game.owner[:] = 0
    expected = max(3, len(TERRITORIES) // 3) + sum(CONTINENT_BONUS.values())
    assert game.reinforcement_count(0) == expected

    # Um território de um continente passa a ser do jogador 1: perde o bônus
    # (e o número de territórios cai, podendo reduzir a base de reforço).
    europa = CONTINENT_TERRITORIES["Europa"][0]
    game.owner[europa] = 1
    expected = max(3, (len(TERRITORIES) - 1) // 3) + sum(
        bonus for continent, bonus in CONTINENT_BONUS.items()
        if continent != "Europa")
    assert game.reinforcement_count(0) == expected


def test_legal_actions_follow_phases():
    game = RiskGame(n_players=2, seed=0)
    owned = set(game.territories_owned(0))
    assert set(game.legal_actions()) == {1 + int(t) for t in owned}

    initial_reinforcements = game.reinforcements_left
    placed = 0
    while game.phase == "reinforce":
        game.apply_action(game.legal_actions()[0])
        placed += 1
    assert placed == initial_reinforcements
    assert game.phase == "attack"
    attack_actions = game.legal_actions()
    assert 0 in attack_actions
    assert any(action != 0 for action in attack_actions)

    game.apply_action(0)
    assert game.phase == "fortify"
    assert 0 in game.legal_actions()
    game.apply_action(0)
    assert game.current_player == 1
    assert game.phase == "reinforce"


def test_combat_dice_and_conquest():
    game, src, dst = _scenario()
    game.rng = FixedRng([6, 5, 1, 1, 2])
    battle = game.attack(src, dst)
    assert battle["attacker_dice"] == [6, 5, 1]
    assert battle["defender_dice"] == [2, 1]
    assert battle["attacker_losses"] == 0
    assert battle["defender_losses"] == 2
    assert battle["conquered"] is True
    assert game.owner[dst] == 0
    assert game.armies[dst] == 3
    assert game.armies[src] == 2


def test_combat_defender_wins_ties():
    game, src, dst = _scenario()
    game.armies[src] = 2
    game.rng = FixedRng([3, 3, 3])
    battle = game.attack(src, dst)
    assert battle["attacker_losses"] == 1
    assert battle["defender_losses"] == 0
    assert battle["conquered"] is False
    assert game.armies[src] == 1


def test_conquest_of_last_territory_wins():
    game, src, dst = _scenario()
    game.owner[:] = 0
    game.owner[dst] = 1
    game.armies[dst] = 1
    game.rng = FixedRng([6, 5, 1, 1])
    battle = game.attack(src, dst)
    assert battle["conquered"] is True
    assert game.winner == 0


def test_invalid_actions_are_rejected():
    game = RiskGame(n_players=2, seed=0)
    with pytest.raises(ValueError):
        game.apply_action(0)  # não se encerra a fase de reforço
    with pytest.raises(ValueError):
        game.attack(0, 0)
    with pytest.raises(ValueError):
        game.fortify(0, 0)


def test_bot_only_returns_legal_actions():
    game = RiskGame(n_players=4, seed=3)
    bot = RiskBot(seed=5, explore=0.2)
    for _ in range(600):
        if game.winner is not None:
            break
        legal = game.legal_actions()
        action = bot.act(game)
        assert action is None or action in legal
        if action is None:
            break
        game.apply_action(action)
    assert game.winner is not None or game.turn > 1


def test_env_contract_and_masking():
    env = RiskEnv(n_players=2, seed=7, max_turns=4)
    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,)
    assert mask.any()
    assert np.allclose(info["action_mask"], mask)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        steps += 1
    assert steps >= 1


def test_env_is_deterministic_with_same_seed():
    actions = []

    def rollout():
        env = RiskEnv(n_players=2, seed=11, max_turns=10)
        env.reset(seed=11)
        out = []
        for _ in range(10):
            mask = env.action_masks()
            if not mask.any():
                break
            action = int(np.flatnonzero(mask)[0])
            out.append(action)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return out

    first = rollout()
    second = rollout()
    assert first == second


def test_invalid_action_falls_back_with_penalty():
    env = RiskEnv(n_players=2, seed=3, max_turns=5)
    env.reset(seed=3)
    mask = env.action_masks()
    assert mask.any() and not mask.all()
    invalid = int(np.flatnonzero(~mask)[0])
    _, reward, terminated, truncated, info = env.step(invalid)
    assert info["invalid_action"] is True
    assert reward <= 0.0


def test_renderer_frame_shape():
    from arenas.risk.renderer import RiskRenderer

    game = RiskGame(seed=0)
    frame = RiskRenderer().frame(game)
    assert frame.shape == (700, 1200, 3)
    assert frame.dtype == np.uint8


def test_record_episode_writes_video(tmp_path):
    pytest.importorskip("imageio")
    from arenas.common.agent_factory import build_agent
    from arenas.risk.config import RiskConfig
    from arenas.risk.record import record_episode

    config = RiskConfig(max_turns=3, seed=1)
    env = RiskEnv(config=config)
    agent = build_agent(
        "dqn", env.observation_space.shape, env.action_space)
    path = tmp_path / "risk.mp4"
    result = record_episode(agent, config, path, seed=1, max_frames=2)
    assert path.exists() and path.stat().st_size > 0
    assert result["frames"] >= 1


def test_four_player_env_contract():
    env = RiskEnv(n_players=4, seed=2, max_turns=3)
    obs, info = env.reset(seed=2)
    assert obs.shape == env.observation_space.shape
    assert obs.shape[0] == len(TERRITORIES) * 5 + 6
    mask = env.action_masks()
    assert mask.any()
    _, reward, terminated, truncated, info = env.step(
        int(np.flatnonzero(mask)[0]))
    assert isinstance(reward, float)
    assert "winner" in info

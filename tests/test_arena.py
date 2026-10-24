import random

import numpy as np
import pytest
import torch

from agents.baseline import CoinRulePolicy, PolicyContext, PushRulePolicy, RandomPolicy, make_policy
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from dqn.agent import DQNAgent
from dqn.checkpoint import save_checkpoint, load_checkpoint

def test_reset_step_observation():
    env = ArenaEnv(ArenaConfig())
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    obs, reward, term, trunc, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    env.close()

def test_invalid_action():
    env = ArenaEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError): env.step(12)
    env.close()

def test_cnn_and_checkpoint(tmp_path):
    env = ArenaEnv()
    obs, _ = env.reset(seed=1)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, batch_size=2)
    assert agent.online(__import__("torch").as_tensor(obs).unsqueeze(0)).shape == (1, 12)
    p = tmp_path / "test.pt"
    save_checkpoint(p, agent)
    agent2 = DQNAgent(env.observation_space.shape, env.action_space.n)
    load_checkpoint(p, agent2, resume=True)
    env.close()

def test_one_episode_without_render():
    env = ArenaEnv(ArenaConfig(episode_seconds=.1))
    obs, _ = env.reset(seed=2)
    while True:
        obs, _, term, trunc, info = env.step(0)
        if term or trunc: break
    assert trunc
    env.close()


def test_death_penalty_and_random_platform_respawn():
    cfg = ArenaConfig(respawn_seconds=0.1)
    env = ArenaEnv(cfg)
    env.reset(seed=4)
    player = env.engine.characters[0]
    player.coins = 6
    player.y = cfg.height + 50
    # One step advances physics and triggers the fatal-region logic.
    env.step(0)
    assert player.coins == 0
    assert player.deaths == 1
    assert not player.alive
    # Advance beyond the respawn timer; the character should return to a platform.
    for _ in range(10):
        env.step(0)
    assert player.alive
    assert any(abs(player.y - py) < 1e-5 for _, py, _, _ in env.engine.platforms)
    env.close()


def test_jump_reaches_first_platform_tier():
    # O maior degrau entre plataformas adjacentes é 75px (chão 360 → plataforma 285);
    # o pulo precisa alcançá-lo para o agente conseguir subir na arena.
    env = ArenaEnv(ArenaConfig(opponent_count=0))
    env.reset(seed=1)
    p = env.engine.characters[0]
    p.x, p.y, p.vx, p.vy, p.on_ground = 100.0, 360.0, 0.0, 0.0, True
    env.step(3)
    for _ in range(90):
        env.step(3)
        if p.on_ground and p.y < 360:
            break
    assert p.y == 285.0
    env.close()


def test_human_policy_returns_valid_action_headless():
    from agents.human import HumanPolicy
    policy = HumanPolicy()
    ctx = PolicyContext(obs=np.zeros((5, 2, 2), dtype=np.float32), rng=np.random.default_rng(0))
    # Sem janela/teclas pressionadas, deve devolver uma ação válida (parado).
    assert 0 <= policy.act(ctx) <= 11


def test_make_policy_registry():
    assert isinstance(make_policy("rule"), PushRulePolicy)
    assert isinstance(make_policy("random"), RandomPolicy)
    assert isinstance(make_policy("coins"), CoinRulePolicy)
    ready = RandomPolicy(n_actions=3)
    assert make_policy(ready) is ready
    with pytest.raises(ValueError):
        make_policy("inexistente")


def test_random_opponent_keeps_historic_range():
    # O adversário "random" histórico sorteava 0..5 (sem empurrões); preserva.
    env = ArenaEnv(opponent_policy="random")
    env.reset(seed=1)
    ctx = PolicyContext(obs=np.zeros(env.observation_space.shape, dtype=np.float32), rng=env.np_random)
    for _ in range(100):
        assert 0 <= env.opponent_policy.act(ctx) < 6
    env.close()


def test_coins_opponent_runs_full_episode():
    env = ArenaEnv(ArenaConfig(episode_seconds=0.1), opponent_policy="coins")
    env.reset(seed=2)
    while True:
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert trunc
    env.close()


def _run_episodes(cfg, seed, n_episodes, save_at=None, path=None, resume_from=None):
    """Mini-loop espelhando train.py, para testar retomada bit a bit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = ArenaEnv(cfg, opponent_policy="rule")
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, batch_size=8)
    start, best = 1, -float("inf")
    if resume_from is not None:
        payload = load_checkpoint(resume_from, agent, resume=True, env=env)
        start = int(payload.get("extra", {}).get("episodes_done", 0)) + 1
        best = float(payload.get("extra", {}).get("best_reward", -float("inf")))
    trace = []
    for ep in range(start, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        total, steps = 0.0, 0
        while True:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, terminated, truncated)
            agent.learn()
            obs = next_obs
            total += reward
            steps += 1
            if terminated or truncated:
                break
        trace.append((ep, total, steps, agent.epsilon, agent.steps))
        if total > best:
            best = total
        if ep == save_at:
            save_checkpoint(path, agent, cfg,
                            {"episode": ep, "reward": total, "episodes_done": ep, "best_reward": best},
                            env=env)
            break
    return trace, agent, env


def test_resume_is_bit_identical(tmp_path):
    cfg = ArenaConfig(episode_seconds=0.5)
    seed = 123
    ckpt = tmp_path / "mid.pt"
    half, agent_half, env_half = _run_episodes(cfg, seed, 2, save_at=2, path=ckpt)
    resumed, agent_resumed, env_resumed = _run_episodes(cfg, seed, 4, resume_from=ckpt)
    full, agent_full, env_full = _run_episodes(cfg, seed, 4)
    # A retomada continua a numeração e reproduz exatamente os episódios 3 e 4.
    assert [row[0] for row in resumed] == [3, 4]
    assert resumed == full[2:]
    for p_r, p_f in zip(agent_resumed.online.parameters(), agent_full.online.parameters()):
        assert torch.equal(p_r, p_f)
    for p_r, p_f in zip(agent_resumed.target.parameters(), agent_full.target.parameters()):
        assert torch.equal(p_r, p_f)
    assert len(agent_resumed.buffer) == len(agent_full.buffer)
    sd_r, sd_f = agent_resumed.buffer.state_dict(), agent_full.buffer.state_dict()
    assert list(sd_r) == list(sd_f)
    for k in sd_r:
        assert np.array_equal(sd_r[k], sd_f[k])
    env_half.close()
    env_resumed.close()
    env_full.close()

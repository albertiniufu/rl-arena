"""Teste de integração do HAPPO portado para o monorepo."""
from arenas.algs.alg_registry import (
    available_algorithms,
    load_algorithm,
    save_checkpoint,
)
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario


def test_happo_discovered_and_runs_one_update(tmp_path):
    assert "happo" in available_algorithms()
    scenario = Scenario.all_vs_all(4)
    cfg = ArenaConfig(level=1, n_agents=4, team_ids=scenario.team_ids)
    env = ArenaMultiEnv(cfg, scenario=scenario)
    module = load_algorithm("happo")
    agent = module.Agent(
        env.observation_space.shape, env.action_space.n,
        n_agents=4, team_ids=scenario.team_ids,
        rollout_steps=4, batch_size=8, epochs=1, device="cpu",
    )
    obs, info = env.reset(seed=2)
    for _ in range(5):
        actions = agent.act_batch(obs)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        for i in range(4):
            agent.add_transition(
                i, obs[i], actions[i], rewards[i],
                next_obs[i], terminated[i])
        obs = next_obs
    assert agent.learn() is not None
    agent.end_episode(info)
    save_checkpoint(
        tmp_path / "happo.pt", agent, cfg,
        {"episodes_done": 1}, episodes=1)
    assert (tmp_path / "happo.pt").exists()
    env.close()

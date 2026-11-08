import argparse, csv
from pathlib import Path
import numpy as np
from arena.environment import ArenaEnv
from arena.multi_environment import ArenaMultiEnv
from arena.scenario import Scenario
from arena.config import ArenaConfig
from arena.skins import list_skins
from algs.alg_registry import algorithm_from_checkpoint, apply_env_config, load_checkpoint

def parse_args(argv=None):
    """Argumentos da CLI; separado de `main` para poder ser testado."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--opponent", choices=["rule", "random", "coins"], default="rule",
                    help="política do adversário na avaliação (padrão: rule)")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (arena/renderer_<skin>.py; padrão: clássico)")
    return ap.parse_args(argv)


def _evaluate_multi(args, alg, cfg):
    """Avaliação gulosa de um checkpoint multiagente (todos os agentes)."""
    scenario = Scenario.from_config(cfg)
    env = ArenaMultiEnv(cfg, scenario=scenario,
                        render_mode="human" if args.render else None,
                        skin=args.skin)
    agent_kwargs = dict(n_agents=scenario.n_agents,
                        team_ids=tuple(scenario.team_ids))
    if scenario.has_roles and getattr(alg.Agent, "accepts_roles", False):
        agent_kwargs["agent_roles"] = tuple(scenario.roles)
    agent = alg.Agent(env.observation_space.shape, env.action_space.n,
                      **agent_kwargs)
    load_checkpoint(args.checkpoint, agent, resume=False)
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        totals = np.zeros(env.n_agents, dtype=np.float64)
        steps = 0
        info = {}
        while True:
            actions = agent.act_batch(obs, greedy=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            totals += rewards
            steps += 1
            if bool(np.all(truncated)):
                break
        for i, agent_info in enumerate(info["agents"]):
            rows.append({
                "episode": ep + 1,
                "agent": i,
                "team": agent_info["team"],
                "reward": float(totals[i]),
                "coins": agent_info["coins"],
                "deaths": agent_info["deaths"],
                "duration_steps": steps,
                "duration_seconds": info.get("sim_time", steps / cfg.fps),
            })
    out = Path("logs/evaluation.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    rewards = np.array([r["reward"] for r in rows])
    print(f"Reward média={rewards.mean():.3f}, "
          f"desvio-padrão={rewards.std():.3f}")
    for team in sorted(set(r["team"] for r in rows)):
        vals = [r["reward"] for r in rows if r["team"] == team]
        print(f"  time {team}: reward média={np.mean(vals):.3f}")
    env.close()


def main():
    args = parse_args()
    # O checkpoint identifica o algoritmo: a config gravada (frame_stack/
    # frame_skip) define a observação e o pacote alg_* define a rede.
    try:
        alg, meta = algorithm_from_checkpoint(args.checkpoint)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Checkpoint inválido: {exc}") from exc
    cfg = ArenaConfig()
    apply_env_config(cfg, meta)
    if getattr(alg.Agent, "multi_agent", False) or cfg.n_agents > 1:
        _evaluate_multi(args, alg, cfg)
        return
    env = ArenaEnv(cfg, render_mode="human" if args.render else None,
                   opponent_policy=args.opponent, skin=args.skin)
    obs, _ = env.reset(seed=args.seed)
    agent = alg.Agent(env.observation_space.shape, env.action_space.n)
    load_checkpoint(args.checkpoint, agent, resume=False)
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total, steps, info = 0.0, 0, {}
        while True:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward; steps += 1
            if terminated or truncated: break
        row = {"episode": ep + 1, "reward": total, "coins": info.get("coins", 0),
               "deaths": info.get("deaths", 0),
               "duration_steps": steps, "duration_seconds": info.get("sim_time", steps / cfg.fps)}
        rows.append(row)
        print(row)
    out = Path("logs/evaluation.csv"); out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    rewards = np.array([r["reward"] for r in rows])
    print(f"Reward média={rewards.mean():.3f}, desvio-padrão={rewards.std():.3f}")
    env.close()

if __name__ == "__main__":
    main()

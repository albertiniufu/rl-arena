import argparse, csv
from pathlib import Path
import numpy as np
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.skins import list_skins
from alg_registry import algorithm_from_checkpoint, apply_env_config, load_checkpoint

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
               "deaths": info.get("deaths", 0), "eliminations": info.get("eliminations", 0),
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

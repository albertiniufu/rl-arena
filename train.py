import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.skins import list_skins
from dqn.agent import DQNAgent
from dqn.checkpoint import save_checkpoint, load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opponent", choices=["rule", "random", "coins"], default="rule")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (arena/renderer_<skin>.py; padrão: clássico)")
    args = ap.parse_args()
    # Os RNGs globais precisam ser semeados para o treino ser reproduzível:
    # a exploração usa o módulo `random` e o buffer amostra com `random.sample`.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = ArenaConfig()
    env = ArenaEnv(cfg, render_mode="human" if args.render else None, opponent_policy=args.opponent, skin=args.skin)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    start_ep, best = 1, -float("inf")
    if args.resume:
        payload = load_checkpoint(args.resume, agent, resume=True, env=env)
        start_ep = int(payload.get("extra", {}).get("episodes_done", 0)) + 1
        best = float(payload.get("extra", {}).get("best_reward", -float("inf")))
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    with log_path.open("w" if new_file else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "reward", "coins", "deaths", "eliminations", "mean_loss", "epsilon"])
        if new_file: writer.writeheader()
        last_ep = start_ep + args.episodes - 1
        last_total = 0.0
        for ep in range(start_ep, last_ep + 1):
            obs, _ = env.reset(seed=args.seed + ep)
            total, losses, steps = 0.0, [], 0
            info = {}
            while True:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.buffer.add(obs, action, reward, next_obs, terminated, truncated)
                loss = agent.learn()
                if loss is not None: losses.append(loss)
                obs = next_obs; total += reward; steps += 1
                if terminated or truncated: break
            last_total = total
            row = {"episode": ep, "steps": steps, "reward": total, "coins": info.get("coins", 0),
                   "deaths": info.get("deaths", 0), "eliminations": info.get("eliminations", 0),
                   "mean_loss": float(np.mean(losses)) if losses else "", "epsilon": agent.epsilon}
            writer.writerow(row); f.flush()
            if total > best:
                best = total
                save_checkpoint("checkpoints/best.pt", agent, cfg,
                                {"episode": ep, "reward": total, "episodes_done": ep, "best_reward": best}, env=env)
            if ep % args.save_every == 0:
                save_checkpoint("checkpoints/latest.pt", agent, cfg,
                                {"episode": ep, "reward": total, "episodes_done": ep, "best_reward": best}, env=env)
            print(f"episode={ep} reward={total:.2f} coins={row['coins']} deaths={row['deaths']} epsilon={agent.epsilon:.3f}")
    save_checkpoint("checkpoints/latest.pt", agent, cfg,
                    {"episode": last_ep, "reward": last_total, "episodes_done": last_ep, "best_reward": best}, env=env)
    env.close()

if __name__ == "__main__":
    main()

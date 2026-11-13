#!/usr/bin/env python3
"""Treinador PPO vetorizado na GPU usando `GpuArenaVecEnv`.

Todo o laço — ambiente, forward, coleta, GAE e updates — fica em tensores
na GPU; só os escalares de log/checkpoint voltam para a CPU. O checkpoint é
o formato padrão (`algorithm="ppo"`), com `n_agents`/`team_ids` na config,
então `evaluate.py`/`play.py` conseguem carregá-lo no modo multiagente.

Uso:
    python arenas/seek_coin/scripts/train_gpu_ppo.py \
        --envs 32 --rollout-steps 32 --updates 100 \
        --checkpoint-dir checkpoints/gpu_ppo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.distributions import Categorical

from arenas.algs.alg_ppo.agent import PPOAgent, compute_gae
from arenas.algs.alg_registry import save_checkpoint
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.gpu_env import GpuArenaVecEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=32,
                        help="passos de ambiente por update")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--coin-count", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=45.0)
    parser.add_argument("--obs-dtype",
                        choices=["float32", "float16", "bfloat16", "uint8"],
                        default="float32")
    parser.add_argument("--amp", action="store_true",
                        help="forward/update em autocast fp16/bf16")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile no modelo (exige compilador C)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="checkpoints/gpu_ppo")
    parser.add_argument("--log-dir", default="logs/gpu_ppo")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA indisponível; usando CPU.")

    team_ids = tuple(range(args.n_agents))
    cfg = ArenaConfig(
        level=args.level, n_agents=args.n_agents, team_ids=team_ids,
        frame_skip=args.frame_skip, frame_stack=args.frame_stack,
        coin_count=args.coin_count, episode_seconds=args.episode_seconds)
    env = GpuArenaVecEnv(
        config=cfg, n_envs=args.envs, n_agents=args.n_agents,
        team_ids=team_ids, device=device, seed=args.seed,
        obs_dtype=args.obs_dtype)
    obs, _ = env.reset(seed=args.seed)
    obs_shape = (env.channels * env.frame_stack,
                 env.grid_h, env.grid_w)
    agent = PPOAgent(
        obs_shape, 6, lr=args.lr, gamma=args.gamma,
        clip=args.clip, gae_lambda=args.gae_lambda,
        value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=device, n_agents=args.n_agents, team_ids=team_ids)
    model = agent.online.train()
    if args.compile:
        model = torch.compile(agent.online)
        model.train()
    optimizer = agent.optimizer
    amp_dtype = (torch.bfloat16 if args.obs_dtype == "bfloat16"
                 else torch.float16)
    amp_enabled = bool(args.amp and env.device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and amp_dtype == torch.float16)

    model_dtype = amp_dtype if amp_enabled else torch.float32

    def autocast():
        return torch.autocast(
            device_type=env.device.type, dtype=amp_dtype,
            enabled=amp_enabled)

    e, a = args.envs, args.n_agents
    n = e * a
    t_steps = args.rollout_steps
    states = torch.empty(
        (t_steps, n, *obs_shape), dtype=env.obs_dtype,
        device=env.device)
    actions = torch.empty(
        (t_steps, n), dtype=torch.long, device=env.device)
    log_probs = torch.empty((t_steps, n), device=env.device)
    values = torch.empty((t_steps, n), device=env.device)
    rewards = torch.empty((t_steps, n), device=env.device)
    next_values = torch.empty((t_steps, n), device=env.device)
    terminal_values = torch.zeros((t_steps, n), device=env.device)
    terminal_mask = torch.zeros(
        (t_steps, n), dtype=torch.bool, device=env.device)
    ends = torch.empty(
        (t_steps, n), dtype=torch.bool, device=env.device)

    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training.csv"
    log_path.write_text(
        "update,mean_reward,policy_loss,value_loss,entropy,steps_per_s\n")

    running = torch.zeros((e, a), device=env.device)
    episode_returns: list[float] = []
    best = -float("inf")
    total_env_steps = 0
    start = time.monotonic()
    print(f"gpu_ppo device={env.device} envs={e} agents={a} "
          f"obs={obs_shape} rollout={t_steps} updates={args.updates} "
          f"obs_dtype={env.obs_dtype} amp={amp_enabled} "
          f"compile={args.compile}")

    for update in range(1, args.updates + 1):
        terminal_values.zero_()
        terminal_mask.zero_()
        for t in range(t_steps):
            obs_flat = obs.reshape(n, *obs_shape)
            model_flat = obs_flat.to(model_dtype)
            with torch.no_grad(), autocast():
                logits, value = model(model_flat)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            next_obs, rew, terminated, truncated, info = env.step(
                action.view(e, a))
            states[t] = obs_flat
            actions[t] = action
            log_probs[t] = log_prob.float()
            values[t] = value.float()
            rewards[t] = rew.reshape(n)
            ends[t] = truncated.reshape(n)
            running += rew
            obs = next_obs
            total_env_steps += e
            if bool(truncated.all()):
                # V(s_T) do estado terminal, antes do reset.
                with torch.no_grad(), autocast():
                    _, terminal_value = model(
                        next_obs.reshape(n, *obs_shape).to(model_dtype))
                terminal_values[t] = terminal_value.float()
                terminal_mask[t] = True
                episode_returns.extend(
                    running.mean(dim=1).detach().cpu().tolist())
                running.zero_()
                obs, _ = env.reset(seed=args.seed + update * 1000 + t)

        # next_values[t] = V(obs_{t+1}); só o passo terminal/último precisa
        # de um forward extra.
        if t_steps > 1:
            next_values[:-1] = values[1:]
        with torch.no_grad(), autocast():
            _, final_value = model(
                obs.reshape(n, *obs_shape).to(model_dtype))
        next_values[-1] = final_value.float()
        next_values[terminal_mask] = terminal_values[terminal_mask]
        zeros_term = torch.zeros_like(ends)
        adv, ret = compute_gae(
            rewards.cpu(), values.cpu(), next_values.cpu(),
            zeros_term.cpu(), ends.cpu(), args.gamma, args.gae_lambda)
        adv = adv.to(env.device)
        ret = ret.to(env.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        states_flat = states.reshape(t_steps * n, *obs_shape)
        actions_flat = actions.reshape(-1)
        old_logp_flat = log_probs.reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)

        last_policy = last_value = last_entropy = 0.0
        for _ in range(args.epochs):
            order = torch.randperm(t_steps * n, device=env.device)
            for start_idx in range(0, t_steps * n, args.batch_size):
                idx = order[start_idx:start_idx + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits, value = model(
                        states_flat[idx].to(model_dtype))
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(actions_flat[idx])
                    ratio = torch.exp(new_logp - old_logp_flat[idx])
                    advantage = adv_flat[idx]
                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio, 1.0 - args.clip,
                        1.0 + args.clip) * advantage
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(
                        value, ret_flat[idx])
                    entropy = dist.entropy().mean()
                    loss = (policy_loss + args.value_coef * value_loss
                            - args.entropy_coef * entropy)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                last_policy = float(policy_loss.item())
                last_value = float(value_loss.item())
                last_entropy = float(entropy.item())

        agent.steps += 1
        mean_reward = (float(np.mean(episode_returns[-100:]))
                       if episode_returns
                       else float(rewards.mean().item()))
        elapsed = time.monotonic() - start
        steps_per_s = total_env_steps / max(elapsed, 1e-9)
        with log_path.open("a") as handle:
            handle.write(
                f"{update},{mean_reward:.4f},{last_policy:.4f},"
                f"{last_value:.4f},{last_entropy:.4f},"
                f"{steps_per_s:.1f}\n")
        if mean_reward > best or update == 1:
            best = mean_reward
            save_checkpoint(
                ckpt_dir / "best.pt", agent, cfg,
                {"update": update, "mean_reward": mean_reward,
                 "episodes_done": update, "best_reward": best},
                episodes=update)
        if update % max(1, args.save_every) == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt", agent, cfg,
                {"update": update, "mean_reward": mean_reward,
                 "episodes_done": update, "best_reward": best},
                episodes=update)
        print(f"update={update} reward={mean_reward:.2f} "
              f"pi={last_policy:.3f} v={last_value:.3f} "
              f"ent={last_entropy:.3f} steps/s={steps_per_s:.0f}")
        if args.max_minutes and elapsed >= args.max_minutes * 60.0:
            print("orçamento de tempo atingido; encerrando")
            break

    save_checkpoint(
        ckpt_dir / "latest.pt", agent, cfg,
        {"update": args.updates, "best_reward": best,
         "episodes_done": args.updates},
        episodes=args.updates)
    print(f"checkpoints em {ckpt_dir}")


if __name__ == "__main__":
    main()

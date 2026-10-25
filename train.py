import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from agents.baseline import make_policy
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.skins import list_skins
from dqn.agent import DQNAgent, CheckpointPolicy
from dqn.checkpoint import save_checkpoint, load_checkpoint


def resolve_resume_path(resume, checkpoint_dir):
    """Resolve `--resume` com nome simples dentro de `--checkpoint-dir`."""
    path = Path(resume)
    if path.parent == Path("."):
        return Path(checkpoint_dir) / path
    return path


def obs_to_image(obs):
    """Grade semântica (5, H, W) -> imagem RGB (H, W, 3) para o TensorBoard.

    Canais: 0 plataformas, 1 jogador, 2 oponentes, 3 moedas, 4 mortos/respawn.
    """
    h, w = obs.shape[1], obs.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        0: (120, 130, 140),
        1: (70, 190, 250),
        2: (240, 90, 95),
        3: (245, 195, 55),
        4: (180, 120, 200),
    }
    for channel, color in colors.items():
        img[obs[channel] > 0] = color
    return img


class AutoOpponent:
    """Currículo de adversários do `--opponent auto` (padrão).

    Por episódio global (1-based; independe de `--episodes` e de retomadas):
      - episódios 1..`random_episodes`: RandomPolicy
      - próximos `baseline_episodes`: alterna `coins` e `rule`
      - resto: joga contra o melhor checkpoint até o momento (guloso),
        com probabilidade `alt_prob` de alternar para `random`, `coins`,
        `rule` ou outro checkpoint disponível em `checkpoints/`.
    As fases usam contagens absolutas (não frações do total) justamente para
    a retomada reproduzível: o mesmo episódio global recebe a mesma política
    em qualquer segmentação da execução. A aleatoriedade vem do módulo
    `random` (seedado pelo --seed).
    """
    def __init__(self, env, checkpoint_dir="checkpoints",
                 random_episodes=50, baseline_episodes=100, alt_prob=0.1):
        self.env = env
        self.checkpoint_dir = Path(checkpoint_dir)
        self.random_episodes = random_episodes
        self.baseline_episodes = baseline_episodes
        self.alt_prob = alt_prob
        shape, n_actions = env.observation_space.shape, env.action_space.n
        self.checkpoint_agent = DQNAgent(shape, n_actions)
        self.alt_agent = DQNAgent(shape, n_actions)

    def init_from_disk(self):
        """Na retomada, carrega o melhor checkpoint existente como oponente."""
        best = self.checkpoint_dir / "best.pt"
        if best.exists():
            load_checkpoint(best, self.checkpoint_agent, resume=False)
        return self

    def update_best(self, agent):
        """Congela os pesos atuais do treino como 'melhor até o momento'."""
        self.checkpoint_agent.online.load_state_dict(agent.online.state_dict())

    def _checkpoints(self):
        if not self.checkpoint_dir.is_dir():
            return []
        return sorted(self.checkpoint_dir.glob("*.pt"))

    def choose(self, episode_number):
        """Política do adversário para o episódio global (1-based)."""
        n = episode_number
        if n <= self.random_episodes:
            return make_policy("random")
        if n <= self.random_episodes + self.baseline_episodes:
            return make_policy("coins" if n % 2 else "rule")
        # Fase de checkpoint: melhor até o momento, com alternância ocasional.
        if random.random() < self.alt_prob:
            options = ["random", "coins", "rule"]
            checkpoints = self._checkpoints()
            if checkpoints:
                options.append("checkpoint")
            pick = random.choice(options)
            if pick == "checkpoint":
                load_checkpoint(random.choice(checkpoints), self.alt_agent, resume=False)
                return CheckpointPolicy(self.alt_agent)
            return make_policy(pick)
        return CheckpointPolicy(self.checkpoint_agent)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opponent", choices=["rule", "random", "coins", "auto"], default="auto")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="diretório de checkpoints (saves e leitura do adversário auto)")
    ap.add_argument("--stats", action="store_true",
                    help="coleta estatísticas do treino da CNN (perda, gradiente, Q) "
                         "em logs/stats.csv e no terminal")
    ap.add_argument("--tensorboard", action="store_true",
                    help="grava escalares, histogramas e a observação em "
                         "logs/runs/seed<N> (abra com: tensorboard --logdir logs/runs)")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (arena/renderer_<skin>.py; padrão: clássico)")
    args = ap.parse_args()
    # Os RNGs globais precisam ser semeados para o treino ser reproduzível:
    # a exploração usa o módulo `random` e o buffer amostra com `random.sample`.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = ArenaConfig()
    ckpt_dir = Path(args.checkpoint_dir)
    resume_path = resolve_resume_path(args.resume, ckpt_dir) if args.resume else None
    # `auto` é um currículo gerenciado pelo AutoOpponent; o env inicia com random.
    env = ArenaEnv(cfg, render_mode="human" if args.render else None,
                   opponent_policy="random" if args.opponent == "auto" else args.opponent,
                   skin=args.skin)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    # O `auto` é criado antes do load para que a retomada já possa chamar
    # `init_from_disk` nele (carregar o melhor checkpoint como adversário).
    auto = AutoOpponent(env, checkpoint_dir=ckpt_dir) if args.opponent == "auto" else None
    start_ep, best = 1, -float("inf")
    if resume_path:
        payload = load_checkpoint(resume_path, agent, resume=True)
        episodes_done = payload.get("episodes")
        if episodes_done is None:
            # Checkpoints sem o campo (ex.: salvos sem `episodes`) caem para o extra histórico.
            episodes_done = payload.get("extra", {}).get("episodes_done", 0)
        start_ep = int(episodes_done) + 1
        best = float(payload.get("extra", {}).get("best_reward", -float("inf")))
        if auto is not None:
            auto.init_from_disk()
    last_ep = start_ep + args.episodes - 1
    Path("logs").mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    # Estatísticas adicionais, opcionais: não consomem RNG, então não alteram o treino.
    stats_f = None
    stats_writer = None
    if args.stats:
        stats_path = Path("logs/stats.csv")
        stats_new = not stats_path.exists() or not args.resume
        stats_f = stats_path.open("w" if stats_new else "a", newline="")
        stats_writer = csv.DictWriter(stats_f, fieldnames=[
            "episode", "steps", "reward", "mean_loss", "min_loss", "max_loss",
            "grad_mean", "grad_max", "q_mean", "q_max", "epsilon", "buffer_size"])
        if stats_new:
            stats_writer.writeheader()
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise SystemExit("TensorBoard não está instalado. Instale com: pip install tensorboard") from exc
        # Um diretório por seed: retomar com o mesmo seed continua as curvas.
        tb_writer = SummaryWriter(str(Path("logs/runs") / f"seed{args.seed}"))
    with log_path.open("w" if new_file else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "reward", "coins", "deaths", "eliminations", "mean_loss", "epsilon"])
        if new_file: writer.writeheader()
        last_total = 0.0
        for ep in range(start_ep, last_ep + 1):
            if auto is not None:
                env.opponent_policy = auto.choose(ep)
            obs, _ = env.reset(seed=args.seed + ep)
            total, losses, steps = 0.0, [], 0
            q_means, q_maxes, grad_norms = [], [], []
            info = {}
            while True:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.buffer.add(obs, action, reward, next_obs, terminated, truncated)
                if args.stats or args.tensorboard:
                    qm, qx = agent.q_stats(obs)
                    q_means.append(qm)
                    q_maxes.append(qx)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                    if args.stats or args.tensorboard:
                        grad_norms.append(agent.last_grad_norm)
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss_step", loss, agent.steps)
                        tb_writer.add_scalar("train/grad_norm_step", agent.last_grad_norm, agent.steps)
                obs = next_obs; total += reward; steps += 1
                if terminated or truncated: break
            last_total = total
            row = {"episode": ep, "steps": steps, "reward": total, "coins": info.get("coins", 0),
                   "deaths": info.get("deaths", 0), "eliminations": info.get("eliminations", 0),
                   "mean_loss": float(np.mean(losses)) if losses else "", "epsilon": agent.epsilon}
            writer.writerow(row); f.flush()
            if args.stats or tb_writer is not None:
                agg = {
                    "mean_loss": float(np.mean(losses)) if losses else 0.0,
                    "min_loss": float(np.min(losses)) if losses else 0.0,
                    "max_loss": float(np.max(losses)) if losses else 0.0,
                    "grad_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                    "grad_max": float(np.max(grad_norms)) if grad_norms else 0.0,
                    "q_mean": float(np.mean(q_means)) if q_means else 0.0,
                    "q_max": float(np.max(q_maxes)) if q_maxes else 0.0,
                }
            if tb_writer is not None:
                tb_writer.add_scalar("train/reward", total, ep)
                tb_writer.add_scalar("train/coins", info.get("coins", 0), ep)
                tb_writer.add_scalar("train/deaths", info.get("deaths", 0), ep)
                tb_writer.add_scalar("train/eliminations", info.get("eliminations", 0), ep)
                for name in ("mean_loss", "min_loss", "max_loss", "grad_mean", "grad_max", "q_mean", "q_max"):
                    tb_writer.add_scalar(f"train/{name}", agg[name], ep)
                tb_writer.add_scalar("train/epsilon", agent.epsilon, ep)
                tb_writer.add_scalar("train/buffer_size", len(agent.buffer), ep)
                tb_writer.add_image("env/observation", obs_to_image(obs), ep, dataformats="HWC")
                tb_writer.add_histogram("weights/conv1", agent.online.features[0].weight, ep)
                tb_writer.add_histogram("weights/head", agent.online.head[-1].weight, ep)
                tb_writer.flush()
            if stats_writer is not None:
                stats_row = {
                    "episode": ep, "steps": steps, "reward": total,
                    **{k: agg[k] for k in ("mean_loss", "min_loss", "max_loss", "grad_mean", "grad_max", "q_mean", "q_max")},
                    "epsilon": agent.epsilon, "buffer_size": len(agent.buffer),
                }
                stats_writer.writerow(stats_row)
                stats_f.flush()
                print(f"  stats: loss={stats_row['mean_loss']:.5f} q_mean={stats_row['q_mean']:.2f} "
                      f"q_max={stats_row['q_max']:.2f} grad={stats_row['grad_mean']:.2f} "
                      f"buffer={stats_row['buffer_size']}")
            if total > best:
                best = total
                save_checkpoint(ckpt_dir / "best.pt", agent, cfg,
                                {"episode": ep, "reward": total, "episodes_done": ep, "best_reward": best},
                                episodes=ep)
                if auto is not None:
                    auto.update_best(agent)
            if ep % args.save_every == 0:
                save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                                {"episode": ep, "reward": total, "episodes_done": ep, "best_reward": best},
                                episodes=ep)
            print(f"episode={ep} reward={total:.2f} coins={row['coins']} deaths={row['deaths']} epsilon={agent.epsilon:.3f}")
    if stats_f is not None:
        stats_f.close()
    if tb_writer is not None:
        tb_writer.close()
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": last_ep, "reward": last_total, "episodes_done": last_ep, "best_reward": best},
                    episodes=last_ep)
    env.close()

if __name__ == "__main__":
    main()

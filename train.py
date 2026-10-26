import argparse, csv, random, warnings
from pathlib import Path
import numpy as np
import torch
from agents.baseline import make_policy
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.skins import list_skins
from arena.vector import make_vector_env
from alg_registry import (
    CheckpointPolicy,
    available_algorithms,
    checkpoint_meta,
    load_algorithm,
    load_checkpoint,
    save_checkpoint,
)


def resolve_resume_path(resume, checkpoint_dir):
    """Resolve `--resume` com nome simples dentro de `--checkpoint-dir`."""
    path = Path(resume)
    if path.parent == Path("."):
        return Path(checkpoint_dir) / path
    return path


def budget_snapshots(total_episodes):
    """Episódios (absolutos) em que vale salvar um snapshot de orçamento.

    Os orçamentos são pontos fixos de episódios treinados (50/100/150/300),
    usados no protocolo de A/B para comparar fases no mesmo orçamento.
    """
    return {b for b in (50, 100, 150, 300) if b <= total_episodes}


def obs_to_image(obs):
    """Grade semântica (5, H, W) -> imagem RGB (H, W, 3) para o TensorBoard.

    Com frame stack a observação tem `5 * frame_stack` canais; usa apenas os
    últimos 5 (o frame atual).
    Canais: 0 plataformas, 1 jogador, 2 oponentes, 3 moedas, 4 mortos/respawn.
    """
    frame = obs[-5:]
    h, w = frame.shape[1], frame.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        0: (120, 130, 140),
        1: (70, 190, 250),
        2: (240, 90, 95),
        3: (245, 195, 55),
        4: (180, 120, 200),
    }
    for channel, color in colors.items():
        img[frame[channel] > 0] = color
    return img


def _try_load_checkpoint(path, agent, frame_stack=None):
    """Carrega `path` em `agent`; devolve False (com aviso) se não der.

    Um `.pt` de outra arquitetura (ou corrompido) no diretório de checkpoints
    não pode quebrar o treino do adversário `auto`: o arquivo é pulado.
    Com `frame_stack` informado, checkpoints gravados com outro valor são
    pulados antes do load. O motivo não é só a mensagem de erro: como a
    primeira conv recebe `5 * frame_stack` canais, o `load_state_dict`
    estoura no meio e deixa os tensores anteriores já sobrescritos — pular
    antes evita esse carregamento parcial de pesos incompatíveis.
    """
    try:
        if frame_stack is not None:
            saved = int(checkpoint_meta(path)["config"].get("frame_stack", 1))
            if saved != frame_stack:
                warnings.warn(
                    f"Checkpoint ignorado ({path}): frame_stack={saved} difere do "
                    f"treino atual ({frame_stack})", UserWarning)
                return False
        load_checkpoint(path, agent, resume=False)
    except Exception as exc:  # arquitetura antiga, arquivo corrompido, etc.
        warnings.warn(f"Checkpoint ignorado ({path}): {exc}", UserWarning)
        return False
    return True


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
    def __init__(self, env, alg, checkpoint_dir="checkpoints",
                 random_episodes=50, baseline_episodes=100, alt_prob=0.1):
        self.env = env
        self.alg = alg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.random_episodes = random_episodes
        self.baseline_episodes = baseline_episodes
        self.alt_prob = alt_prob
        shape, n_actions = env.observation_space.shape, env.action_space.n
        # Os agentes oponentes usam o MESMO algoritmo do treino: o state_dict
        # do melhor checkpoint é copiado para cá em `update_best`, e as chaves
        # precisam casar.
        self.checkpoint_agent = alg.Agent(shape, n_actions)
        self.alt_agent = alg.Agent(shape, n_actions)

    def init_from_disk(self):
        """Na retomada, carrega o melhor checkpoint existente como oponente.

        Checkpoint incompatível é ignorado com aviso; o agente fica com os
        pesos iniciais até o primeiro `update_best`.
        """
        best = self.checkpoint_dir / "best.pt"
        if best.exists():
            _try_load_checkpoint(best, self.checkpoint_agent,
                                 frame_stack=self.env.cfg.frame_stack)
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
                # Tenta os arquivos em ordem sorteada; incompatíveis são
                # pulados com aviso. Se nenhum carregar, cai para random.
                for path in random.sample(checkpoints, len(checkpoints)):
                    if _try_load_checkpoint(path, self.alt_agent,
                                            frame_stack=self.env.cfg.frame_stack):
                        return CheckpointPolicy(self.alt_agent)
                return make_policy("random")
            return make_policy(pick)
        return CheckpointPolicy(self.checkpoint_agent)

def parse_args(argv=None):
    """Argumentos da CLI; separado de `main` para poder ser testado."""
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
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="frames empilhados na observação (arena/environment.py)")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="repetições da ação por step do agente")
    ap.add_argument("--alg", choices=available_algorithms(), default="dqn",
                    help="algoritmo de RL: pacote alg_<nome> na raiz do projeto "
                         "(dqn = DQN padrão; doubledqn_per_dueling = Double DQN "
                         "+ PER + dueling; ppo = PPO; a2c = A2C)")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="taxa de aprendizado do Adam (na retomada, o otimizador "
                         "restaurado do checkpoint sobrepõe este valor)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="tamanho do batch amostrado do replay buffer por learn()")
    ap.add_argument("--envs", type=int, default=1,
                    help="ambientes vetorizados na coleta (Fase 5; incompatível "
                         "com --opponent auto, --render, --stats e --tensorboard)")
    return ap.parse_args(argv)


def validate_train_args(args):
    """Restrições do treino vetorizado (`--envs N`).

    A Fase 5 roda com adversário fixo: o currículo por episódio do `auto`, o
    render e a telemetria (`--stats`/`--tensorboard`) são por processo e não
    mapeiam em workers concorrentes. Na CLI, o erro vira `SystemExit`.
    """
    if args.envs < 1:
        raise SystemExit(f"--envs deve ser >= 1 (recebido {args.envs})")
    if args.envs == 1:
        return
    # Os algoritmos on-policy (PPO/A2C) precisam do rollout contíguo por
    # worker e do gancho de fim de episódio; o loop vetorizado descarta
    # transições de autoreset e não mapeia essa fronteira. O caminho de
    # 1 ambiente é o suportado para eles.
    if getattr(load_algorithm(args.alg).Agent, "on_policy", False):
        raise SystemExit(
            f"--alg {args.alg} é on-policy e não é compatível com --envs "
            f"{args.envs}; use --envs 1")
    incompat = [
        ("--opponent auto", args.opponent == "auto",
         "o currículo escolhe a política por episódio; no vetor ela é fixa"),
        ("--render", args.render,
         "o render é por processo e não há uma janela única para os workers"),
        ("--stats", args.stats,
         "a telemetria por episódio não mapeia em workers concorrentes"),
        ("--tensorboard", args.tensorboard,
         "a telemetria por episódio não mapeia em workers concorrentes"),
    ]
    for flag, used, reason in incompat:
        if used:
            raise SystemExit(
                f"--envs {args.envs} é incompatível com {flag}: {reason}")


def _final_observation(infos, index):
    """Observação terminal do worker `index`, se o VectorEnv a expôs.

    O gymnasium >= 1.0 usa a chave `final_obs`; versões/forks antigos usam
    `final_observation`. Devolve `None` quando não há (autoreset NEXT_STEP,
    em que o próprio `next_obs` do passo terminal já é o estado final).
    """
    for key in ("final_observation", "final_obs"):
        array = infos.get(key)
        if array is not None and array[index] is not None:
            return array[index]
    return None


def _worker_final_info(infos, index):
    """Info final do worker `index`, cobrindo os formatos do gymnasium.

    Com autoreset NEXT_STEP (padrão do gymnasium >= 1.0), o `info` do passo
    terminal já é o info final do episódio. Com SAME_STEP (ou versões
    antigas), o info original vem em `final_info`, como dict de arrays por
    worker ou array de objetos. As chaves internas de máscara (`_key`) e o
    próprio `final_info` não fazem parte do info do episódio.
    """
    final = infos.get("final_info")
    if isinstance(final, dict):
        return {key: value[index] for key, value in final.items()
                if not str(key).startswith("_")}
    if final is not None and isinstance(final[index], dict):
        return final[index]
    return {key: value[index] for key, value in infos.items()
            if not str(key).startswith("_") and key != "final_info"
            and hasattr(value, "__getitem__")}


def _train_vectorized(args, cfg, agent, ckpt_dir, start_ep, last_ep, best, budgets):
    """Loop de treino com coleta vetorizada (`--envs > 1`, Fase 5).

    Cada passo do vetor coleta uma transição de cada worker e `learn()` roda a
    cada 4 passos do vetor. Os RNGs (`random`/`numpy`/`torch`) já foram
    semeados no `main`; o `reset(seed)` do VectorEnv semeia o worker `i` com
    `seed + i`, e os autoresets seguintes continuam esses RNGs. Episódios são
    contados por `terminated | truncated` de cada worker; como os workers
    correm em paralelo, o total pode passar de `last_ep` em até
    `--envs - 1` episódios.

    Escalas com N workers: o `learn()` a cada 4 passos do vetor consome ~4N
    transições por update (contra 1 no loop histórico de 1 env), e epsilon e
    sincronização da target avançam por update — portanto, por transição,
    ambos andam ~4N vezes mais devagar que no treino de 1 env. Isso é a
    decisão da Fase 5 (K=4 mantido), não um schedule equivalente.
    """
    n_envs = args.envs
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    total_episodes = start_ep - 1
    last_total = 0.0
    # Orçamentos já cruzados antes desta execução (retomada) não são regravados:
    # o snapshot representa o total de episódios do próprio rótulo.
    fired_budgets = {b for b in budgets if b <= start_ep - 1}
    losses_since_row = []
    vector_steps = 0
    venv = make_vector_env(n_envs, cfg, args.opponent, asynchronous=True)
    try:
        obs, _ = venv.reset(seed=args.seed)
        worker_rewards = np.zeros(n_envs, dtype=np.float64)
        worker_steps = np.zeros(n_envs, dtype=np.int64)
        # Com autoreset NEXT_STEP, o passo seguinte ao fim de um episódio só
        # devolve a observação resetada; ele não vira transição de treino.
        pending_reset = np.zeros(n_envs, dtype=bool)
        with log_path.open("w" if new_file else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "steps", "reward", "coins", "deaths", "eliminations",
                "mean_loss", "epsilon"])
            if new_file:
                writer.writeheader()
            while total_episodes < last_ep:
                actions = agent.act_batch(obs)
                next_obs, rewards, terminations, truncations, infos = venv.step(actions)
                vector_steps += 1
                transitions, completed = [], []
                for i in range(n_envs):
                    if pending_reset[i]:
                        pending_reset[i] = False
                        continue
                    done = bool(terminations[i]) or bool(truncations[i])
                    final_obs = _final_observation(infos, i) if done else None
                    # O VectorEnv auto-reseta: sem a observação terminal, a
                    # transição perderia o estado final do episódio.
                    next_state = final_obs if final_obs is not None else next_obs[i]
                    # `truncated` não entra no buffer: o alvo bootstrapa no
                    # fim por tempo (só terminais verdadeiros zeram o futuro).
                    transitions.append((
                        obs[i], int(actions[i]), float(rewards[i]), next_state,
                        bool(terminations[i])))
                    worker_rewards[i] += float(rewards[i])
                    worker_steps[i] += 1
                    if done:
                        completed.append(i)
                        if final_obs is None:
                            pending_reset[i] = True
                agent.buffer.add_batch(transitions)
                # Um learn por passo do vetor é caro demais; a cada 4 passos
                # mantém a atualização frequente sem dominar o wall-clock.
                if vector_steps % 4 == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses_since_row.append(loss)
                # Episódios que terminam no mesmo passo compartilham a janela
                # de perdas desde a última linha.
                mean_loss = float(np.mean(losses_since_row)) if losses_since_row else ""
                if completed:
                    losses_since_row = []
                for i in completed:
                    total_episodes += 1
                    info = _worker_final_info(infos, i)
                    reward = float(worker_rewards[i])
                    row = {
                        "episode": total_episodes,
                        "steps": int(worker_steps[i]),
                        "reward": reward,
                        "coins": info.get("coins", 0),
                        "deaths": info.get("deaths", 0),
                        "eliminations": info.get("eliminations", 0),
                        "mean_loss": mean_loss,
                        "epsilon": agent.epsilon,
                    }
                    writer.writerow(row)
                    f.flush()
                    last_total = reward
                    if reward > best:
                        best = reward
                        save_checkpoint(ckpt_dir / "best.pt", agent, cfg,
                                        {"episode": total_episodes, "reward": reward,
                                         "episodes_done": total_episodes,
                                         "best_reward": best},
                                        episodes=total_episodes)
                    if total_episodes % args.save_every == 0:
                        save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                                        {"episode": total_episodes, "reward": reward,
                                         "episodes_done": total_episodes,
                                         "best_reward": best},
                                        episodes=total_episodes)
                    # Snapshot ao CRUZAR o orçamento: o total pode pular o
                    # valor exato, então não dá para comparar por igualdade.
                    for budget in sorted(budgets - fired_budgets):
                        if total_episodes >= budget:
                            save_checkpoint(ckpt_dir / "budget" / f"ep{budget:03d}.pt",
                                            agent, cfg, {"episode": budget},
                                            episodes=budget)
                            fired_budgets.add(budget)
                    print(f"episode={total_episodes} reward={reward:.2f} "
                          f"coins={row['coins']} deaths={row['deaths']} "
                          f"epsilon={agent.epsilon:.3f}")
                    worker_rewards[i] = 0.0
                    worker_steps[i] = 0
                obs = next_obs
    finally:
        venv.close()
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": total_episodes, "reward": last_total,
                     "episodes_done": total_episodes, "best_reward": best},
                    episodes=total_episodes)


def main():
    args = parse_args()
    validate_train_args(args)
    alg = load_algorithm(args.alg)
    # Os RNGs globais precisam ser semeados para o treino ser reproduzível:
    # a exploração usa o módulo `random`; o replay uniforme amostra com
    # `random.sample` e o priorizado sorteia com `np.random.choice`.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = ArenaConfig()
    cfg.frame_stack, cfg.frame_skip = args.frame_stack, args.frame_skip
    ckpt_dir = Path(args.checkpoint_dir)
    resume_path = resolve_resume_path(args.resume, ckpt_dir) if args.resume else None
    # `auto` é um currículo gerenciado pelo AutoOpponent; o env inicia com random.
    env = ArenaEnv(cfg, render_mode="human" if args.render else None,
                   opponent_policy="random" if args.opponent == "auto" else args.opponent,
                   skin=args.skin)
    agent = alg.Agent(env.observation_space.shape, env.action_space.n,
                      lr=args.lr, batch_size=args.batch_size)
    # O `auto` é criado antes do load para que a retomada já possa chamar
    # `init_from_disk` nele (carregar o melhor checkpoint como adversário).
    auto = (AutoOpponent(env, alg, checkpoint_dir=ckpt_dir)
            if args.opponent == "auto" else None)
    start_ep, best = 1, -float("inf")
    if resume_path:
        try:
            payload = load_checkpoint(resume_path, agent, resume=True)
        except ValueError as exc:
            raise SystemExit(f"Não foi possível retomar: {exc}") from exc
        episodes_done = payload.get("episodes")
        if episodes_done is None:
            # Checkpoints sem o campo (ex.: salvos sem `episodes`) caem para o extra histórico.
            episodes_done = payload.get("extra", {}).get("episodes_done", 0)
        start_ep = int(episodes_done) + 1
        best = float(payload.get("extra", {}).get("best_reward", -float("inf")))
        if auto is not None:
            auto.init_from_disk()
    last_ep = start_ep + args.episodes - 1
    # Orçamentos contam episódios absolutos (retomada continua a numeração),
    # então o snapshot compara fases com o mesmo total de treino.
    budgets = budget_snapshots(last_ep)
    Path("logs").mkdir(exist_ok=True)
    # parents=True: o protocolo de A/B usa caminhos aninhados
    # (ex.: checkpoints/ab/fase0-branch/seed7-100).
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    if args.envs > 1:
        # O env de episódio único serviu só para montar espaços/agente: o loop
        # vetorizado cria os próprios workers. Caminho histórico fica intacto
        # com --envs 1 (default).
        env.close()
        _train_vectorized(args, cfg, agent, ckpt_dir, start_ep, last_ep, best, budgets)
        return
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
                # `truncated` não entra no buffer: o alvo bootstrapa no fim
                # por tempo; só `terminated` zera o futuro.
                agent.buffer.add(obs, action, reward, next_obs, terminated)
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
            # Algoritmos on-policy marcam a fronteira do episódio no rollout
            # e podem fazer um update parcial aqui (A2C).
            end_loss = agent.end_episode(info) if hasattr(agent, "end_episode") else None
            if end_loss is not None:
                losses.append(end_loss)
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
                # Cada algoritmo expõe a última camada que faz o papel de
                # cabeça (no dueling, o ramo de valor escalar).
                tb_writer.add_histogram("weights/head", agent.head_weights(), ep)
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
            if ep in budgets:
                # Snapshot de orçamento só com pesos: fica fora do glob
                # não-recursivo do `auto` e serve ao protocolo de A/B.
                save_checkpoint(ckpt_dir / "budget" / f"ep{ep:03d}.pt", agent, cfg,
                                {"episode": ep}, episodes=ep)
            print(f"episode={ep} reward={total:.2f} coins={row['coins']} deaths={row['deaths']} epsilon={agent.epsilon:.3f}")
    if stats_f is not None:
        stats_f.close()
    if tb_writer is not None:
        tb_writer.close()
    # O PPO pode terminar com um rollout parcial; consumi-lo antes do
    # checkpoint final evita desperdiçar as últimas transições.
    if hasattr(agent, "finish_training"):
        tail_loss = agent.finish_training()
        if tail_loss is not None:
            print(f"finish_training loss={tail_loss:.5f}")
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": last_ep, "reward": last_total, "episodes_done": last_ep, "best_reward": best},
                    episodes=last_ep)
    env.close()

if __name__ == "__main__":
    main()

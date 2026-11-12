import sys
from pathlib import Path

# Permite rodar o arquivo diretamente (python arenas/seek_coin/train.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse, copy, csv, os, queue, random, threading, time, warnings
from pathlib import Path
import numpy as np
import torch
from arenas.seek_coin.agents.baseline import make_policy
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.skins import list_skins
from arenas.seek_coin.arena.vector import make_vector_env
from arenas.algs.alg_registry import (
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


def time_budget_reached(start, max_hours, now=None):
    """O orçamento wall-clock do treino terminou?

    `max_hours=None` desliga o orçamento. O relógio é monotônico e pode ser
    injetado (`now`) para os testes; o treino checa ao fim de cada episódio.
    """
    if max_hours is None:
        return False
    now = time.monotonic() if now is None else now
    return (now - start) >= max_hours * 3600.0


def obs_to_image(obs, n_channels=5):
    """Grade semântica (C, H, W) -> imagem RGB (H, W, 3) para o TensorBoard.

    Com frame stack a observação tem `C * frame_stack` canais; usa apenas os
    últimos `n_channels` (o frame atual). `n_channels=6` no modo multiagente
    (IPPO/MAPPO), em que o canal 5 marca os aliados.
    Canais: 0 plataformas, 1 jogador, 2 oponentes, 3 moedas, 4 mortos/respawn,
    5 aliados (multiagente).
    """
    frame = obs[-n_channels:]
    h, w = frame.shape[1], frame.shape[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        0: (120, 130, 140),
        1: (70, 190, 250),
        2: (240, 90, 95),
        3: (245, 195, 55),
        4: (180, 120, 200),
        5: (120, 230, 140),
    }
    for channel, color in colors.items():
        if channel < frame.shape[0]:
            img[frame[channel] > 0] = color
    return img


def greedy_evaluation(agent, env, episodes, seed):
    """Roda `episodes` episódios gulosos e devolve médias/desvios.

    O ambiente é dedicado e as seeds são fixas (`seed + i`, como no
    `evaluate.py`), então a avaliação é comparável entre algoritmos, seeds de
    treino e retomadas. `act(..., greedy=True)` não usa ε nem atualiza pesos;
    `reset(seed=...)` semeia os RNGs do próprio ambiente, de modo que os RNGs
    do treino não são consumidos.
    """
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, info = 0.0, {}
        while True:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rows.append({
            "reward": total,
            "coins": info.get("coins", 0),
            "deaths": info.get("deaths", 0),
        })
    rewards = np.array([r["reward"] for r in rows], dtype=float)
    coins = np.array([r["coins"] for r in rows], dtype=float)
    deaths = np.array([r["deaths"] for r in rows], dtype=float)
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "coins_mean": float(coins.mean()),
        "coins_std": float(coins.std()),
        "deaths_mean": float(deaths.mean()),
        "deaths_std": float(deaths.std()),
    }


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

def epsilon_for_episode(ep, ramp_end, floor):
    """Rampa linear de epsilon por episódio: 1 → `floor` em `ramp_end`.

    `ep` é o número absoluto do episódio (a retomada continua a numeração),
    então a rampa retoma de onde parou. Ancorar em episódios — e não em
    `--episodes` — mantém o comportamento de treinos por tempo de parede
    (`--max-hours` com `--episodes 1000000`), em que a rampa relativa nunca
    terminaria.
    """
    return max(floor, 1.0 - (1.0 - floor) * ep / ramp_end)


def schedule_epsilon(agent, ep, ramp_episodes, floor):
    """Aplica a rampa de epsilon do `train.py` ao agente, se ele usar epsilon.

    Algoritmos on-policy (PPO/A2C) têm `on_policy = True` e não exploram por
    epsilon. Com a rampa ligada, o decay por passo do agente é desligado:
    quem controla ε é o episódio, não o número de `learn()`.
    """
    if getattr(agent, "on_policy", False) or ramp_episodes <= 0:
        return
    agent.epsilon_decay = 1.0
    agent.epsilon = epsilon_for_episode(ep, ramp_episodes, floor)


class _DefaultHelpFormatter(argparse.HelpFormatter):
    """Mostra em `-h` o valor que o treino usa quando a opção é omitida.

    O `ArgumentDefaultsHelpFormatter` só acrescenta o default em opções com
    `help` e imprime `None` literal. Aqui o rótulo é sempre incluído e os
    defaults dinâmicos (calculados depois do parse) recebem um texto próprio.
    """

    _LABELS = {
        "resume": "nenhum",
        "run_name": "gerado da config",
        "max_hours": "sem limite",
        "skin": "clássico",
        "learn_every": "4 (com --envs 1) ou 1 (vetorizado)",
    }

    def _get_help_string(self, action):
        text = action.help or ""
        if action.default is argparse.SUPPRESS:
            return text
        label = self._LABELS.get(action.dest)
        if label is None:
            label = action.default
            if isinstance(label, bool):
                label = "ligado" if label else "desligado"
            elif label is None:
                label = "nenhum"
        if "(padrão:" in text:  # o texto já explica o default
            return text
        return f"{text} (padrão: {label})" if text else f"(padrão: {label})"


def parse_args(argv=None):
    """Argumentos da CLI; separado de `main` para poder ser testado."""
    ap = argparse.ArgumentParser(formatter_class=_DefaultHelpFormatter)
    ap.add_argument("--episodes", type=int, default=300,
                    help="episódios de treino (adicionais ao retomar)")
    ap.add_argument("--level", type=int, default=0,
                    help="nível do mapa (0 = mapa histórico; 1 = plataforma móvel)")
    ap.add_argument("--render", action="store_true",
                    help="abre a janela do renderer durante o treino")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint de onde retomar (pesos, otimizador e contadores)")
    ap.add_argument("--save-every", type=int, default=10,
                    help="salva latest.pt a cada N episódios")
    ap.add_argument("--seed", type=int, default=7,
                    help="seed de random/numpy/torch; cada episódio usa seed+ep")
    ap.add_argument("--opponent", choices=["rule", "random", "coins", "auto"], default="auto",
                    help="política do adversário (auto = currículo que alterna "
                         "random, coins, rule e checkpoints)")
    ap.add_argument("--agents", type=int, default=None,
                    help="número de agentes no modo multiagente (IPPO/MAPPO/HAPPO); "
                         "omitido = 4 com um algoritmo multiagente e 1 nos demais")
    ap.add_argument("--scenario", choices=["all-vs-all", "teams"],
                    default="all-vs-all",
                    help="cenário multiagente: todos-contra-todos (padrão) ou times")
    ap.add_argument("--teams", type=int, default=2,
                    help="número de times com --scenario teams (padrão 2)")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="diretório de checkpoints (saves e leitura do adversário auto)")
    ap.add_argument("--stats", action="store_true",
                    help="coleta estatísticas do treino da CNN (perda, gradiente, Q) "
                         "em logs/stats.csv e no terminal")
    ap.add_argument("--tensorboard", action="store_true",
                    help="grava escalares, histogramas e a observação em "
                         "logs/runs/<run-name> (abra com: tensorboard --logdir logs/runs)")
    ap.add_argument("--run-name", default=None,
                    help="nome do experimento no TensorBoard; omitido usa um nome "
                         "derivado de algoritmo, adversário, frame stack/skip, lr, "
                         "batch e seed")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="avalia a política gulosa a cada N episódios e grava "
                         "eval/* no TensorBoard; 0 desliga (só com --envs 1)")
    ap.add_argument("--eval-episodes", type=int, default=5,
                    help="episódios por avaliação gulosa periódica")
    ap.add_argument("--eval-seed", type=int, default=1000,
                    help="seed base da avaliação periódica (episódios seed+i, "
                         "como no evaluate.py)")
    ap.add_argument("--eval-opponent", choices=["rule", "random", "coins"],
                    default="rule",
                    help="adversário fixo da avaliação periódica; independente "
                         "do --opponent do treino")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="encerra o treino de forma limpa após N horas de parede; "
                         "salva o checkpoint final e consome o rollout on-policy")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (arena/renderer_<skin>.py); omitido "
                         "usa a skin clássica")
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="frames empilhados na observação (arena/environment.py)")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="repetições da ação por step do agente")
    ap.add_argument("--push-coin-transfer", type=int, default=None,
                    help="moedas que um empurrão transfere do alvo para quem "
                         "empurrou (0 desliga; omitido usa o default do "
                         "ArenaConfig, 5)")
    ap.add_argument("--push-coin-reward", type=float, default=None,
                    help="peso de recompensa por moeda transferida no empurrão "
                         "(a mecânica de transferência não muda; omitido usa "
                         "o ArenaConfig, 1.0)")
    ap.add_argument("--pbrs-coef", type=float, default=None,
                    help="coeficiente do PBRS sobre a diferença de moedas "
                         "(0 desliga; omitido usa o ArenaConfig, 0)")
    ap.add_argument("--save-every-minutes", type=float, default=0.0,
                    help="checkpoint extra a cada N minutos de parede "
                         "(0 desliga; multiagente)")
    ap.add_argument("--video-every-minutes", type=float, default=0.0,
                    help="grava um episódio guloso em MP4 a cada N minutos "
                         "de parede (0 desliga; multiagente)")
    ap.add_argument("--alg",
                    choices=[name for name in available_algorithms()
                             if name != "ppo_continuous"],
                    default="dqn",
                    help="algoritmo de RL: pacote alg_<nome> em algs/ "
                         "(dqn = DQN enxuto; doubledqn_per_dueling = Double DQN "
                         "+ PER + dueling; ppo = PPO; a2c = A2C; "
                         "ippo/mappo/happo = multiagente)")
    ap.add_argument("--lr", type=float, default=2.5e-4,
                    help="taxa de aprendizado do Adam (na retomada, o otimizador "
                         "restaurado do checkpoint sobrepõe este valor)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="tamanho do batch amostrado do replay buffer por learn()")
    ap.add_argument("--epsilon-ramp-episodes", type=int, default=150,
                    help="rampa linear de epsilon (1 → --epsilon-min) ao longo "
                         "destes episódios absolutos; 0 desliga a rampa e usa o "
                         "decay por passo do agente")
    ap.add_argument("--epsilon-min", type=float, default=0.1,
                    help="piso de epsilon na rampa (e no decay do agente)")
    ap.add_argument("--warmup", type=int, default=1000,
                    help="transições no buffer antes do primeiro learn() "
                         "(só algoritmos off-policy)")
    ap.add_argument("--learn-every", type=int, default=None,
                    help="um update a cada N passos do agente (só com --envs 1); "
                         "4 no modo de 1 ambiente (valor validado, ~3x mais passos "
                         "por hora) e 1 no modo vetorizado, onde o update já é por "
                         "passo do vetor; 1 = um update por transição")
    ap.add_argument("--envs", type=int, default=1,
                    help="ambientes vetorizados na coleta (age em lote; exige "
                         "adversário fixo e é incompatível com --opponent auto, "
                         "--render, --stats, --tensorboard e --eval-every)")
    ap.add_argument("--overlap", action="store_true",
                    help="sobrepõe coleta (CPU, workers) e update (GPU) num "
                         "treino on-policy: a coleta do próximo rollout roda "
                         "enquanto a GPU atualiza o anterior (política "
                         "defasada em 1 rollout; exige PPO/A2C)")
    args = ap.parse_args(argv)
    # O número de agentes depende do algoritmo: IPPO/MAPPO são multiagente.
    if args.agents is None:
        args.agents = (4 if getattr(load_algorithm(args.alg).Agent,
                                    "multi_agent", False) else 1)
    # O default depende do modo: K=4 só faz sentido com 1 ambiente; no loop
    # vetorizado o update já acontece a cada passo do vetor. Um valor
    # explícito continua sendo validado em `validate_train_args`.
    if args.learn_every is None:
        args.learn_every = 1 if args.envs > 1 else 4
    return args


def tensorboard_run_name(args):
    """Nome do experimento no TensorBoard: algoritmo + configurações.

    O nome precisa ser estável para retomar o mesmo treino. Por isso ele
    inclui algoritmo, adversário, frame stack/skip, lr, batch e seed — e não
    inclui o número de episódios, que muda ao retomar. `--run-name` permite
    um nome explícito.
    """
    if args.run_name:
        name = args.run_name.strip()
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise SystemExit("--run-name deve ser um nome simples, sem barras")
        return name
    scenario = ""
    if getattr(args, "agents", 1) > 1:
        scenario = f"_{args.scenario}_a{args.agents}"
        if args.scenario == "teams":
            scenario += f"_t{args.teams}"
    return (
        f"{args.alg}_configs_{args.opponent}{scenario}"
        f"_fs{args.frame_stack}_fk{args.frame_skip}"
        f"_lr{args.lr:g}_b{args.batch_size}_seed{args.seed}"
    )


def validate_train_args(args):
    """Restrições dos modos vetorizado (`--envs N`) e sobreposto (`--overlap`).

    Os dois modos rodam com adversário fixo: o currículo por episódio do
    `auto`, o render e a telemetria (`--stats`/`--tensorboard`) são por
    processo e não mapeiam em workers concorrentes. Na CLI, o erro vira
    `SystemExit`.
    """
    if args.envs < 1:
        raise SystemExit(f"--envs deve ser >= 1 (recebido {args.envs})")
    max_hours = getattr(args, "max_hours", None)
    if max_hours is not None and max_hours <= 0:
        raise SystemExit(f"--max-hours deve ser > 0 (recebido {max_hours})")
    if args.learn_every < 1:
        raise SystemExit(
            f"--learn-every deve ser >= 1 (recebido {args.learn_every})")
    eval_every = getattr(args, "eval_every", 0)
    if eval_every < 0:
        raise SystemExit(
            f"--eval-every deve ser >= 0 (recebido {eval_every})")
    eval_episodes = getattr(args, "eval_episodes", 5)
    if eval_every > 0 and eval_episodes < 1:
        raise SystemExit(
            f"--eval-episodes deve ser >= 1 (recebido {eval_episodes})")
    # Modo multiagente (IPPO/MAPPO): cenário fixo e sem coleta
    # vetorizada/sobreposta (o loop é único e todos os agentes aprendem).
    agents = getattr(args, "agents", 1)
    scenario = getattr(args, "scenario", "all-vs-all")
    teams = getattr(args, "teams", 2)
    multi_agent_alg = getattr(load_algorithm(args.alg).Agent,
                              "multi_agent", False)
    if agents < 1:
        raise SystemExit(f"--agents deve ser >= 1 (recebido {agents})")
    if agents > 1 and not multi_agent_alg:
        raise SystemExit(
            "--agents > 1 exige um algoritmo multiagente (--alg ippo/mappo/happo)")
    if multi_agent_alg:
        if agents < 2:
            raise SystemExit(
                f"--alg {args.alg} exige --agents >= 2")
        if args.envs > 1:
            raise SystemExit(
                "--envs > 1 ainda não é suportado no modo multiagente")
        if getattr(args, "overlap", False):
            raise SystemExit("--overlap não se aplica ao modo multiagente")
        if args.opponent != "auto":
            raise SystemExit(
                "--opponent não se aplica ao modo multiagente; "
                "todos os agentes aprendem")
        if scenario == "teams":
            if teams < 2:
                raise SystemExit(
                    f"--teams deve ser >= 2 com --scenario teams (recebido {teams})")
            if agents % teams:
                raise SystemExit(
                    f"--agents ({agents}) deve ser divisível por --teams ({teams})")
    elif scenario != "all-vs-all":
        raise SystemExit(
            "--scenario só se aplica ao modo multiagente (--alg ippo/mappo/happo)")
    if getattr(args, "save_every_minutes", 0) < 0:
        raise SystemExit("--save-every-minutes deve ser >= 0")
    if getattr(args, "video_every_minutes", 0) < 0:
        raise SystemExit("--video-every-minutes deve ser >= 0")
    pbrs_coef = getattr(args, "pbrs_coef", None)
    if pbrs_coef is not None and pbrs_coef < 0:
        raise SystemExit("--pbrs-coef deve ser >= 0")
    push_coin_reward = getattr(args, "push_coin_reward", None)
    if push_coin_reward is not None and push_coin_reward < 0:
        raise SystemExit("--push-coin-reward deve ser >= 0")
    overlap = getattr(args, "overlap", False)
    if overlap and not getattr(load_algorithm(args.alg).Agent, "on_policy", False):
        raise SystemExit(
            f"--overlap só se aplica a algoritmos on-policy (PPO/A2C); "
            f"--alg {args.alg} é off-policy")
    if args.envs == 1 and not overlap:
        return
    if args.envs > 1 and args.learn_every != 1:
        raise SystemExit(
            "--learn-every não se combina com --envs > 1: o loop vetorizado "
            "já faz um update a cada 4 passos do vetor")
    incompat = [
        ("--opponent auto", args.opponent == "auto",
         "o currículo escolhe a política por episódio; no vetor ela é fixa"),
        ("--render", args.render,
         "o render é por processo e não há uma janela única para os workers"),
        ("--stats", args.stats,
         "a telemetria por episódio não mapeia em workers concorrentes"),
        ("--tensorboard", args.tensorboard,
         "a telemetria por episódio não mapeia em workers concorrentes"),
        ("--eval-every", getattr(args, "eval_every", 0) > 0,
         "a avaliação periódica usa um ambiente dedicado e não mapeia em "
         "workers concorrentes"),
    ]
    mode = "--overlap" if overlap else f"--envs {args.envs}"
    for flag, used, reason in incompat:
        if used:
            raise SystemExit(f"{mode} é incompatível com {flag}: {reason}")


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


def _train_vectorized(args, cfg, agent, ckpt_dir, start_ep, last_ep, best, budgets,
                      start_time):
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
        # Início (relógio de parede) do episódio corrente de cada worker; os
        # workers correm em paralelo, então "wall time" é o tempo de coleta
        # compartilhado até o fim do episódio daquele worker.
        worker_ep_start = np.full(n_envs, time.monotonic(), dtype=np.float64)
        # Com autoreset NEXT_STEP, o passo seguinte ao fim de um episódio só
        # devolve a observação resetada; ele não vira transição de treino.
        pending_reset = np.zeros(n_envs, dtype=bool)
        with log_path.open("w" if new_file else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "steps", "reward", "coins", "deaths",
                "mean_loss", "epsilon"])
            if new_file:
                writer.writeheader()
            while total_episodes < last_ep:
                schedule_epsilon(agent, total_episodes + 1,
                                 args.epsilon_ramp_episodes, args.epsilon_min)
                actions = agent.act_batch(obs)
                next_obs, rewards, terminations, truncations, infos = venv.step(actions)
                vector_steps += 1
                transitions, completed = [], []
                for i in range(n_envs):
                    if pending_reset[i]:
                        pending_reset[i] = False
                        worker_ep_start[i] = time.monotonic()
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
                    ep_seconds = time.monotonic() - worker_ep_start[i]
                    print(f"episode={total_episodes} reward={reward:.2f} "
                          f"coins={row['coins']} deaths={row['deaths']} "
                          f"epsilon={agent.epsilon:.3f} time={ep_seconds:.2f}s")
                    worker_rewards[i] = 0.0
                    worker_steps[i] = 0
                    worker_ep_start[i] = time.monotonic()
                obs = next_obs
                if time_budget_reached(start_time, args.max_hours):
                    print(f"orçamento de {args.max_hours:g} h atingido; "
                          f"encerrando após o episódio {total_episodes}")
                    break
    finally:
        venv.close()
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": total_episodes, "reward": last_total,
                     "episodes_done": total_episodes, "best_reward": best},
                    episodes=total_episodes)


def _rollout_target(agent):
    """Transições por update: `rollout_steps` no PPO, `n_steps` no A2C."""
    target = getattr(agent, "rollout_steps", None) or getattr(agent, "n_steps", 1)
    return max(1, int(target))


def _append_rollout_segment(agent, segment):
    """Adiciona um trecho contíguo de um worker ao rollout on-policy.

    O buffer é uma lista plana; para o GAE/n-step não atravessar de um worker
    para outro, o último passo de cada trecho é marcado como fim de episódio.
    O alvo ainda faz bootstrap pelo `next_state` guardado, como num
    truncamento — só a propagação da vantagem é interrompida.
    """
    last = len(segment) - 1
    for k, (state, action, reward, next_state, terminated, log_prob, done) in \
            enumerate(segment):
        agent.buffer.add(state, action, reward, next_state, terminated,
                         log_prob)
        if done or k == last:
            agent.buffer.mark_episode_end()


def _record_episode(f, writer, args, agent, cfg, ckpt_dir, budgets,
                    fired_budgets, best, episode, reward, steps, coins,
                    deaths, mean_loss, ep_seconds):
    """Grava a linha do episódio, salva checkpoints e imprime o resumo.

    Usado pelos loops de coleta concorrentes (vetorizado e `--overlap`); os
    checkpoints são salvos sempre na thread principal, nunca na de coleta.
    """
    row = {
        "episode": episode,
        "steps": steps,
        "reward": reward,
        "coins": coins,
        "deaths": deaths,
        "mean_loss": mean_loss,
        "epsilon": agent.epsilon,
    }
    writer.writerow(row)
    f.flush()
    if reward > best:
        best = reward
        save_checkpoint(ckpt_dir / "best.pt", agent, cfg,
                        {"episode": episode, "reward": reward,
                         "episodes_done": episode, "best_reward": best},
                        episodes=episode)
    if episode % args.save_every == 0:
        save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                        {"episode": episode, "reward": reward,
                         "episodes_done": episode, "best_reward": best},
                        episodes=episode)
    for budget in sorted(budgets - fired_budgets):
        if episode >= budget:
            save_checkpoint(ckpt_dir / "budget" / f"ep{budget:03d}.pt",
                            agent, cfg, {"episode": budget}, episodes=budget)
            fired_budgets.add(budget)
    print(f"episode={episode} reward={reward:.2f} coins={coins} "
          f"deaths={deaths} epsilon={agent.epsilon:.3f} "
          f"time={ep_seconds:.2f}s")
    return best


def multi_greedy_evaluation(agent, env, episodes, seed):
    """Avaliação gulosa multiagente com seeds fixas (`seed + i`).

    Devolve médias/desvios globais e médias por time, no mesmo espírito de
    `greedy_evaluation`, mas consumindo ações conjuntas.
    """
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        totals = np.zeros(env.n_agents, dtype=np.float64)
        agents = []
        while True:
            actions = agent.act_batch(obs, greedy=True)
            obs, rewards, terminated, truncated, info = env.step(actions)
            totals += rewards
            agents = info.get("agents", [])
            if bool(np.all(truncated)):
                break
        rows.append({
            "reward": totals,
            "coins": np.array([a["coins"] for a in agents], dtype=float),
            "deaths": np.array([a["deaths"] for a in agents], dtype=float),
        })
    rewards = np.stack([r["reward"] for r in rows])
    coins = np.stack([r["coins"] for r in rows])
    deaths = np.stack([r["deaths"] for r in rows])
    metrics = {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "coins_mean": float(coins.mean()),
        "coins_std": float(coins.std()),
        "deaths_mean": float(deaths.mean()),
        "deaths_std": float(deaths.std()),
    }
    for team in sorted(set(env.scenario.team_ids)):
        idx = [i for i, t in enumerate(env.scenario.team_ids) if t == team]
        metrics[f"team{team}_reward_mean"] = float(rewards[:, idx].mean())
        metrics[f"team{team}_coins_mean"] = float(coins[:, idx].mean())
        metrics[f"team{team}_deaths_mean"] = float(deaths[:, idx].mean())
    return metrics


def record_multi_episode(agent, env, path, seed):
    """Grava um episódio guloso em MP4 e devolve o caminho.

    Usa um ambiente dedicado e `act_batch(greedy=True)`: não consome o RNG do
    treino. Headless com `SDL_VIDEODRIVER=dummy` (definido pelo chamador).
    """
    try:
        import imageio
        import pygame
        import pygame.surfarray  # noqa: F401 — submódulo precisa de import explícito
    except ImportError as exc:
        raise SystemExit(
            "Gravar vídeo requer: pip install imageio imageio-ffmpeg") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path), fps=env.cfg.fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "23",
                       "-movflags", "+faststart"])

    def append_frame():
        frame = np.transpose(
            pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
        writer.append_data(frame)

    try:
        obs, _ = env.reset(seed=seed)
        env.render()
        append_frame()
        while True:
            actions = agent.act_batch(obs, greedy=True)
            obs, _, _, truncated, _ = env.step(actions)
            append_frame()
            if bool(np.all(truncated)):
                break
    finally:
        writer.close()
    return path


def _train_multi_agent(args, cfg, scenario, env, agent, ckpt_dir, start_ep,
                       last_ep, best, budgets, start_time):
    """Loop de treino multiagente (IPPO/MAPPO): ações conjuntas.

    O ambiente é único e todos os agentes agem a cada step; `agent.learn()`
    atualiza a rede quando o rollout de todos os agentes enche (no MAPPO, o
    crítico centralizado consome o passo conjunto montado pelo buffer).
    A telemetria usa médias por agente para manter as colunas históricas do
    `logs/training.csv`.
    """
    n_agents = scenario.n_agents
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    eval_env = None
    if args.eval_every > 0:
        eval_env = ArenaMultiEnv(cfg, scenario=scenario, render_mode=None)
    # Vídeos periódicos usam um ambiente dedicado e render headless. O
    # driver é definido antes de o pygame ser inicializado pelo renderer.
    video_env = None
    if args.video_every_minutes > 0:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        video_env = ArenaMultiEnv(cfg, scenario=scenario, render_mode="human",
                                  skin=args.skin)
    last_clock_save = time.monotonic()
    last_video_at = time.monotonic()
    stats_f = stats_writer = None
    if args.stats:
        stats_path = Path("logs/stats.csv")
        stats_new = not stats_path.exists() or not args.resume
        stats_f = stats_path.open("w" if stats_new else "a", newline="")
        stats_writer = csv.DictWriter(stats_f, fieldnames=[
            "episode", "steps", "reward", "mean_loss", "min_loss", "max_loss",
            "grad_mean", "grad_max", "q_mean", "q_max", "epsilon",
            "buffer_size"])
        if stats_new:
            stats_writer.writeheader()
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise SystemExit(
                "TensorBoard não está instalado. Instale com: "
                "pip install tensorboard") from exc
        tb_writer = SummaryWriter(
            str(Path("logs/runs") / tensorboard_run_name(args)))

    total_episodes = start_ep - 1
    last_total = 0.0
    fired_budgets = {b for b in budgets if b <= start_ep - 1}
    try:
        with log_path.open("w" if new_file else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "steps", "reward", "coins", "deaths",
                "mean_loss", "epsilon"])
            if new_file:
                writer.writeheader()
            for ep in range(start_ep, last_ep + 1):
                ep_start = time.monotonic()
                obs, info = env.reset(seed=args.seed + ep)
                totals = np.zeros(n_agents, dtype=np.float64)
                pbrs_totals = np.zeros(n_agents, dtype=np.float64)
                losses, q_means, q_maxes, grad_norms = [], [], [], []
                steps = 0
                while True:
                    actions = agent.act_batch(obs)
                    next_obs, rewards, terminated, truncated, info = \
                        env.step(actions)
                    pbrs_totals += np.array(
                        [a["pbrs_this_step"] for a in info["agents"]],
                        dtype=np.float64)
                    for i in range(n_agents):
                        agent.add_transition(
                            i, obs[i], int(actions[i]), float(rewards[i]),
                            next_obs[i], bool(terminated[i]))
                    if args.stats or tb_writer is not None:
                        qm, qx = agent.q_stats(obs[0])
                        q_means.append(qm)
                        q_maxes.append(qx)
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                        if args.stats or tb_writer is not None:
                            grad_norms.append(agent.last_grad_norm)
                        if tb_writer is not None:
                            tb_writer.add_scalar(
                                "train/loss_step", loss, agent.steps)
                            tb_writer.add_scalar(
                                "train/grad_norm_step",
                                agent.last_grad_norm, agent.steps)
                    obs = next_obs
                    totals += rewards
                    steps += 1
                    if bool(np.all(truncated)):
                        break
                end_loss = agent.end_episode(info)
                if end_loss is not None:
                    losses.append(end_loss)
                ep_seconds = time.monotonic() - ep_start
                reward_mean = float(totals.mean())
                coins_mean = float(np.mean(
                    [a["coins"] for a in info["agents"]]))
                deaths_mean = float(np.mean(
                    [a["deaths"] for a in info["agents"]]))
                mean_loss = float(np.mean(losses)) if losses else 0.0
                last_total = reward_mean
                total_episodes = ep
                best = _record_episode(
                    f, writer, args, agent, cfg, ckpt_dir, budgets,
                    fired_budgets, best, ep, reward_mean, steps, coins_mean,
                    deaths_mean, mean_loss, ep_seconds)
                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "train/reward_mean", reward_mean, ep)
                    tb_writer.add_scalar("train/coins_mean", coins_mean, ep)
                    tb_writer.add_scalar("train/deaths_mean", deaths_mean, ep)
                    tb_writer.add_scalar("train/mean_loss", mean_loss, ep)
                    tb_writer.add_scalar("train/epsilon", agent.epsilon, ep)
                    tb_writer.add_scalar(
                        "train/buffer_size", len(agent.buffer), ep)
                    tb_writer.add_scalar(
                        "train/pbrs_mean", float(pbrs_totals.mean()), ep)
                    for team in sorted(set(scenario.team_ids)):
                        idx = [i for i, t in enumerate(scenario.team_ids)
                               if t == team]
                        tb_writer.add_scalar(
                            f"train/team{team}/pbrs",
                            float(pbrs_totals[idx].mean()), ep)
                    for team in info.get("teams", []):
                        t = team["team"]
                        tb_writer.add_scalar(
                            f"train/team{t}/reward", team["reward"], ep)
                        tb_writer.add_scalar(
                            f"train/team{t}/coins", team["coins"], ep)
                        tb_writer.add_scalar(
                            f"train/team{t}/deaths", team["deaths"], ep)
                    tb_writer.add_image(
                        "env/observation",
                        obs_to_image(obs[0], n_channels=6), ep,
                        dataformats="HWC")
                    tb_writer.add_histogram(
                        "weights/conv1", agent.online.features[0].weight, ep)
                    tb_writer.add_histogram(
                        "weights/head", agent.head_weights(), ep)
                    tb_writer.flush()
                # Snapshots e vídeos por relógio de parede (15 min no run de
                # 3 h), sempre na fronteira de um episódio.
                if args.save_every_minutes > 0 and (
                        time.monotonic() - last_clock_save
                        >= args.save_every_minutes * 60.0):
                    save_checkpoint(
                        ckpt_dir / "timed" / f"ep{ep:05d}.pt", agent, cfg,
                        {"episode": ep, "reward": reward_mean,
                         "episodes_done": ep, "best_reward": best},
                        episodes=ep)
                    last_clock_save = time.monotonic()
                    print(f"checkpoint por tempo: ep{ep:05d}.pt")
                if video_env is not None and (
                        time.monotonic() - last_video_at
                        >= args.video_every_minutes * 60.0):
                    video_path = record_multi_episode(
                        agent, video_env,
                        ckpt_dir / "videos" / f"ep{ep:05d}.mp4",
                        seed=args.eval_seed + ep)
                    last_video_at = time.monotonic()
                    print(f"vídeo salvo em {video_path}")
                    if tb_writer is not None:
                        tb_writer.add_text("videos/ultimo", str(video_path), ep)
                        tb_writer.flush()
                if stats_writer is not None:
                    agg = {
                        "mean_loss": mean_loss,
                        "min_loss": float(np.min(losses)) if losses else 0.0,
                        "max_loss": float(np.max(losses)) if losses else 0.0,
                        "grad_mean": (float(np.mean(grad_norms))
                                      if grad_norms else 0.0),
                        "grad_max": (float(np.max(grad_norms))
                                     if grad_norms else 0.0),
                        "q_mean": float(np.mean(q_means)) if q_means else 0.0,
                        "q_max": float(np.max(q_maxes)) if q_maxes else 0.0,
                    }
                    stats_row = {
                        "episode": ep, "steps": steps, "reward": reward_mean,
                        **agg, "epsilon": agent.epsilon,
                        "buffer_size": len(agent.buffer),
                    }
                    stats_writer.writerow(stats_row)
                    stats_f.flush()
                    print(f"  stats: loss={agg['mean_loss']:.5f} "
                          f"q_mean={agg['q_mean']:.2f} "
                          f"q_max={agg['q_max']:.2f} "
                          f"grad={agg['grad_mean']:.2f} "
                          f"buffer={len(agent.buffer)}")
                if eval_env is not None and ep % args.eval_every == 0:
                    metrics = multi_greedy_evaluation(
                        agent, eval_env, args.eval_episodes, args.eval_seed)
                    print(f"  eval ep={ep} "
                          f"reward={metrics['reward_mean']:.2f}"
                          f"±{metrics['reward_std']:.2f} "
                          f"coins={metrics['coins_mean']:.2f} "
                          f"deaths={metrics['deaths_mean']:.2f}")
                    if tb_writer is not None:
                        for name in ("reward_mean", "reward_std",
                                     "coins_mean", "coins_std",
                                     "deaths_mean", "deaths_std"):
                            tb_writer.add_scalar(
                                f"eval/{name}", metrics[name], ep)
                        for key, value in metrics.items():
                            if key.startswith("team"):
                                tb_writer.add_scalar(
                                    f"eval/{key}", value, ep)
                        tb_writer.flush()
                if time_budget_reached(start_time, args.max_hours):
                    print(f"orçamento de {args.max_hours:g} h atingido; "
                          f"encerrando após o episódio {ep}")
                    break
    finally:
        if stats_f is not None:
            stats_f.close()
        if tb_writer is not None:
            tb_writer.close()
        if eval_env is not None:
            eval_env.close()
        if video_env is not None:
            video_env.close()
        env.close()
    # O rollout parcial final é consumido antes de gravar o checkpoint.
    if hasattr(agent, "finish_training"):
        tail = agent.finish_training()
        if tail is not None:
            print(f"finish_training loss={tail:.5f}")
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": total_episodes, "reward": last_total,
                     "episodes_done": total_episodes, "best_reward": best},
                    episodes=total_episodes)


def _train_vectorized_on_policy(args, cfg, agent, ckpt_dir, start_ep, last_ep,
                                best, budgets, start_time):
    """Loop vetorizado para PPO/A2C (`--envs > 1`).

    A coleta lê `--envs` ambientes em paralelo e age em lote (`act_batch`), o
    que troca N forwards de batch 1 por um só — a GPU passa a ser usada de
    verdade e os workers aproveitam os núcleos de CPU. Para o cálculo de
    retornos não atravessar workers, o rollout é montado em blocos por worker
    (`_append_rollout_segment`); o tamanho do bloco é o alvo do algoritmo
    dividido por `--envs`, então cada update consome um rollout completo.
    """
    n_envs = args.envs
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    total_episodes = start_ep - 1
    last_total = 0.0
    fired_budgets = {b for b in budgets if b <= start_ep - 1}
    chunk_steps = max(1, _rollout_target(agent) // n_envs)
    losses_since_row = []
    venv = make_vector_env(n_envs, cfg, args.opponent, asynchronous=True)
    try:
        obs, _ = venv.reset(seed=args.seed)
        worker_rewards = np.zeros(n_envs, dtype=np.float64)
        worker_steps = np.zeros(n_envs, dtype=np.int64)
        worker_ep_start = np.full(n_envs, time.monotonic(), dtype=np.float64)
        pending_reset = np.zeros(n_envs, dtype=bool)
        with log_path.open("w" if new_file else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "steps", "reward", "coins", "deaths",
                "mean_loss", "epsilon"])
            if new_file:
                writer.writeheader()
            while total_episodes < last_ep:
                # Trechos por worker: a ordem de inserção no buffer será
                # worker-major, para cada trajetória ficar contígua.
                segments = [[] for _ in range(n_envs)]
                stop = False
                for _ in range(chunk_steps):
                    actions = agent.act_batch(obs)
                    next_obs, rewards, terminations, truncations, infos = \
                        venv.step(actions)
                    for i in range(n_envs):
                        if pending_reset[i]:
                            pending_reset[i] = False
                            worker_ep_start[i] = time.monotonic()
                            continue
                        done = bool(terminations[i]) or bool(truncations[i])
                        final_obs = _final_observation(infos, i) if done else None
                        next_state = final_obs if final_obs is not None else next_obs[i]
                        segments[i].append((
                            obs[i], int(actions[i]), float(rewards[i]),
                            next_state, bool(terminations[i]), None, done))
                        worker_rewards[i] += float(rewards[i])
                        worker_steps[i] += 1
                        if not done:
                            continue
                        if final_obs is None:
                            pending_reset[i] = True
                        total_episodes += 1
                        info = _worker_final_info(infos, i)
                        reward = float(worker_rewards[i])
                        mean_loss = (float(np.mean(losses_since_row))
                                     if losses_since_row else "")
                        last_total = reward
                        best = _record_episode(
                            f, writer, args, agent, cfg, ckpt_dir, budgets,
                            fired_budgets, best, total_episodes, reward,
                            int(worker_steps[i]), info.get("coins", 0),
                            info.get("deaths", 0), mean_loss,
                            time.monotonic() - worker_ep_start[i])
                        worker_rewards[i] = 0.0
                        worker_steps[i] = 0
                        worker_ep_start[i] = time.monotonic()
                    obs = next_obs
                    if total_episodes >= last_ep:
                        stop = True
                        break
                for i in range(n_envs):
                    _append_rollout_segment(agent, segments[i])
                if total_episodes >= last_ep:
                    stop = True
                loss = agent.learn()
                if loss is not None:
                    losses_since_row = [loss]
                if stop:
                    break
                if time_budget_reached(start_time, args.max_hours):
                    print(f"orçamento de {args.max_hours:g} h atingido; "
                          f"encerrando após o episódio {total_episodes}")
                    break
    finally:
        venv.close()
    # O último rollout pode ter ficado incompleto (fim dos episódios/max-hours).
    if hasattr(agent, "finish_training"):
        tail = agent.finish_training()
        if tail is not None:
            print(f"finish_training loss={tail:.5f}")
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": total_episodes, "reward": last_total,
                     "episodes_done": total_episodes, "best_reward": best},
                    episodes=total_episodes)


@torch.no_grad()
def _act_batch_on_cpu(net, obs, generator):
    """Ações e log-probs de um lote, pela cópia CPU da política.

    O coletor roda numa thread separada e não toca na rede da GPU: assim a
    coleta (CPU) e o update (GPU) acontecem de fato ao mesmo tempo. O
    `generator` é exclusivo do coletor — o RNG default do torch é global
    entre threads, e usar um gerador próprio evita que a amostragem da
    coleta corrompa/consuma o RNG do update.
    """
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    logits, _ = net(x)
    probs = torch.softmax(logits, dim=1)
    actions = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    log_probs = torch.log_softmax(logits, dim=1).gather(
        1, actions.unsqueeze(1)).squeeze(1)
    return actions.numpy(), log_probs.numpy()


def _train_on_policy_overlap(args, cfg, agent, ckpt_dir, start_ep, last_ep,
                             best, budgets, start_time):
    """Coleta (CPU) sobreposta ao update (GPU) para PPO/A2C.

    Uma thread coleta rollouts com uma cópia CPU da política e os entrega por
    uma fila de tamanho 1; a thread principal aplica o rollout ao buffer e faz
    o update na GPU. Enquanto o update roda, o coletor já está no próximo
    rollout, usando os pesos de antes do update — por isso a política fica
    defasada em uma atualização. O PPO compensa isso guardando o log-prob da
    ação no momento da coleta (`RolloutBuffer.log_probs_array`), então o ratio
    compara com a política de comportamento correta; no A2C o único update por
    rollout já é aproximadamente on-policy com a defasagem.

    A fila limitada a 1 garante que o coletor fique no máximo um rollout à
    frente: o ganho é a sobreposição, sem deixar a política envelhecer sem
    limite.
    """
    n_envs = args.envs
    chunk_steps = max(1, _rollout_target(agent) // n_envs)
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    total_episodes = start_ep - 1
    last_total = 0.0
    fired_budgets = {b for b in budgets if b <= start_ep - 1}

    venv = make_vector_env(n_envs, cfg, args.opponent, asynchronous=True)
    # A rede de coleta vive na CPU; a rede de treino continua no device
    # escolhido (GPU quando disponível). `eval()` desliga dropout/BN (não há),
    # mas documenta que esta cópia nunca é treinada.
    rollout_net = copy.deepcopy(agent.online).cpu().eval()
    actor_net = copy.deepcopy(rollout_net)
    weights_lock = threading.Lock()
    payloads = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    # Gerador da amostragem da coleta, separado do RNG global do update.
    action_gen = torch.Generator().manual_seed(args.seed + 1)

    def collect():
        try:
            obs, _ = venv.reset(seed=args.seed)
            worker_rewards = np.zeros(n_envs, dtype=np.float64)
            worker_steps = np.zeros(n_envs, dtype=np.int64)
            worker_ep_start = np.full(n_envs, time.monotonic(),
                                      dtype=np.float64)
            pending_reset = np.zeros(n_envs, dtype=bool)
            total = start_ep - 1
            while not stop_event.is_set():
                # Pesos do início do rollout: uma sincronização do update
                # pode terminar no meio, mas só vale para o próximo rollout.
                with weights_lock:
                    rollout_net.load_state_dict(actor_net.state_dict())
                segments = [[] for _ in range(n_envs)]
                episodes = []
                reason = None
                for _ in range(chunk_steps):
                    actions, log_probs = _act_batch_on_cpu(
                        rollout_net, obs, action_gen)
                    next_obs, rewards, terminations, truncations, infos = \
                        venv.step(actions)
                    for i in range(n_envs):
                        if pending_reset[i]:
                            pending_reset[i] = False
                            worker_ep_start[i] = time.monotonic()
                            continue
                        done = bool(terminations[i]) or bool(truncations[i])
                        final_obs = _final_observation(infos, i) if done else None
                        next_state = final_obs if final_obs is not None else next_obs[i]
                        segments[i].append((
                            obs[i], int(actions[i]), float(rewards[i]),
                            next_state, bool(terminations[i]),
                            float(log_probs[i]), done))
                        worker_rewards[i] += float(rewards[i])
                        worker_steps[i] += 1
                        if not done:
                            continue
                        if final_obs is None:
                            pending_reset[i] = True
                        total += 1
                        info = _worker_final_info(infos, i)
                        episodes.append({
                            "episode": total,
                            "reward": float(worker_rewards[i]),
                            "steps": int(worker_steps[i]),
                            "coins": info.get("coins", 0),
                            "deaths": info.get("deaths", 0),
                            "seconds": time.monotonic() - worker_ep_start[i],
                        })
                        worker_rewards[i] = 0.0
                        worker_steps[i] = 0
                        worker_ep_start[i] = time.monotonic()
                    obs = next_obs
                    if total >= last_ep:
                        reason = "episodes"
                        break
                    if time_budget_reached(start_time, args.max_hours):
                        reason = "budget"
                        break
                payloads.put({"segments": segments, "episodes": episodes,
                              "total": total, "reason": reason})
                if reason is not None:
                    break
        finally:
            venv.close()

    thread = threading.Thread(target=collect, name="coleta", daemon=True)
    thread.start()
    last_loss = None
    with log_path.open("w" if new_file else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "steps", "reward", "coins", "deaths",
            "mean_loss", "epsilon"])
        if new_file:
            writer.writeheader()
        try:
            while True:
                payload = payloads.get()
                for i in range(n_envs):
                    _append_rollout_segment(agent, payload["segments"][i])
                loss = agent.learn()
                if loss is not None:
                    last_loss = float(loss)
                total_episodes = payload["total"]
                for rec in payload["episodes"]:
                    last_total = rec["reward"]
                    best = _record_episode(
                        f, writer, args, agent, cfg, ckpt_dir, budgets,
                        fired_budgets, best, rec["episode"], rec["reward"],
                        rec["steps"], rec["coins"], rec["deaths"],
                        last_loss if last_loss is not None else "",
                        rec["seconds"])
                # Publica os pesos pós-update para o próximo rollout.
                with weights_lock:
                    actor_net.load_state_dict(
                        {key: value.detach().cpu()
                         for key, value in agent.online.state_dict().items()})
                if payload["reason"] is not None:
                    if payload["reason"] == "budget":
                        print(f"orçamento de {args.max_hours:g} h atingido; "
                              f"encerrando após o episódio {total_episodes}")
                    break
        finally:
            stop_event.set()
            thread.join(timeout=5)
    if hasattr(agent, "finish_training"):
        tail = agent.finish_training()
        if tail is not None:
            print(f"finish_training loss={tail:.5f}")
    save_checkpoint(ckpt_dir / "latest.pt", agent, cfg,
                    {"episode": total_episodes, "reward": last_total,
                     "episodes_done": total_episodes, "best_reward": best},
                    episodes=total_episodes)


def main():
    args = parse_args()
    validate_train_args(args)
    start_time = time.monotonic()
    alg = load_algorithm(args.alg)
    # Os RNGs globais precisam ser semeados para o treino ser reproduzível:
    # a exploração usa o módulo `random`; o replay uniforme amostra com
    # `random.sample` e o priorizado sorteia com `np.random.choice`.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = ArenaConfig()
    cfg.level = args.level
    cfg.frame_stack, cfg.frame_skip = args.frame_stack, args.frame_skip
    if args.push_coin_transfer is not None:
        cfg.push_coin_transfer = args.push_coin_transfer
    if args.push_coin_reward is not None:
        cfg.push_coin_reward = args.push_coin_reward
    if args.pbrs_coef is not None:
        cfg.pbrs_coef = args.pbrs_coef
    ckpt_dir = Path(args.checkpoint_dir)
    resume_path = resolve_resume_path(args.resume, ckpt_dir) if args.resume else None
    multi_agent = getattr(alg.Agent, "multi_agent", False)
    scenario = None
    if multi_agent:
        # O cenário vai para o ArenaConfig (e portanto para o checkpoint).
        scenario = (Scenario.all_vs_all(args.agents)
                    if args.scenario == "all-vs-all"
                    else Scenario.teams(args.agents, args.teams))
        cfg.n_agents = scenario.n_agents
        cfg.team_ids = tuple(scenario.team_ids)
        env = ArenaMultiEnv(cfg, render_mode="human" if args.render else None,
                            skin=args.skin)
    else:
        # `auto` é um currículo gerenciado pelo AutoOpponent; o env inicia com random.
        env = ArenaEnv(
            cfg, render_mode="human" if args.render else None,
            opponent_policy="random" if args.opponent == "auto" else args.opponent,
            skin=args.skin)
    agent_kwargs = dict(lr=args.lr, batch_size=args.batch_size)
    if multi_agent:
        agent_kwargs.update(n_agents=scenario.n_agents,
                            team_ids=tuple(scenario.team_ids))
    elif not getattr(alg.Agent, "on_policy", False):
        # Só os agentes off-policy usam epsilon e warmup; PPO/A2C não
        # aceitam esses parâmetros (a exploração deles é a entropia).
        agent_kwargs.update(epsilon_min=args.epsilon_min, warmup=args.warmup)
    agent = alg.Agent(env.observation_space.shape, env.action_space.n,
                      **agent_kwargs)
    # Deixa explícito onde a rede roda: sem CUDA no torch, todo o treino cai
    # na CPU e a GPU do sistema fica ociosa.
    print(f"device={agent.device} alg={args.alg} envs={args.envs}")
    # O `auto` é criado antes do load para que a retomada já possa chamar
    # `init_from_disk` nele (carregar o melhor checkpoint como adversário).
    auto = (AutoOpponent(env, alg, checkpoint_dir=ckpt_dir)
            if args.opponent == "auto" and not multi_agent else None)
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
        saved_level = int((payload.get("config") or {}).get("level", 0))
        if saved_level != int(cfg.level):
            raise SystemExit(
                f"Nível do checkpoint ({saved_level}) difere do --level "
                f"({cfg.level}); retome com o mesmo nível.")
        if multi_agent:
            # O cenário não pode mudar entre execuções: os buffers e a
            # interpretação dos canais de aliados/inimigos dependem dele.
            saved = payload.get("config") or {}
            saved_n = int(saved.get("n_agents", 0) or 0)
            saved_teams = tuple(saved.get("team_ids") or ())
            if (saved_n != scenario.n_agents
                    or saved_teams != tuple(scenario.team_ids)):
                raise SystemExit(
                    f"Cenário do checkpoint incompatível (n_agents={saved_n}, "
                    f"team_ids={saved_teams}); esperado "
                    f"n_agents={scenario.n_agents}, "
                    f"team_ids={tuple(scenario.team_ids)}.")
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
    if multi_agent:
        _train_multi_agent(args, cfg, scenario, env, agent, ckpt_dir,
                           start_ep, last_ep, best, budgets, start_time)
        return
    log_path = Path("logs/training.csv")
    new_file = not log_path.exists() or not args.resume
    if getattr(args, "overlap", False):
        # Coleta e update sobrepostos: o env de episódio único serviu só para
        # montar espaços/agente; o coletor cria os próprios workers.
        env.close()
        _train_on_policy_overlap(args, cfg, agent, ckpt_dir, start_ep, last_ep,
                                 best, budgets, start_time)
        return
    if args.envs > 1:
        # O env de episódio único serviu só para montar espaços/agente: o loop
        # vetorizado cria os próprios workers. Caminho histórico fica intacto
        # com --envs 1 (default).
        env.close()
        if getattr(alg.Agent, "on_policy", False):
            _train_vectorized_on_policy(
                args, cfg, agent, ckpt_dir, start_ep, last_ep, best,
                budgets, start_time)
        else:
            _train_vectorized(args, cfg, agent, ckpt_dir, start_ep, last_ep,
                              best, budgets, start_time)
        return
    # Avaliação gulosa periódica: ambiente separado, com adversário e seeds
    # fixos. Criar depois do early return garante que ele só existe no caminho
    # suportado (--envs 1).
    eval_env = None
    if args.eval_every > 0:
        eval_env = ArenaEnv(cfg, render_mode=None,
                            opponent_policy=args.eval_opponent, skin=None)
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
        # Um diretório por experimento: o nome inclui algoritmo e configs,
        # então retomar com os mesmos argumentos continua as curvas.
        tb_writer = SummaryWriter(
            str(Path("logs/runs") / tensorboard_run_name(args)))
    with log_path.open("w" if new_file else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "steps", "reward", "coins", "deaths", "mean_loss", "epsilon"])
        if new_file: writer.writeheader()
        last_total = 0.0
        last_done = start_ep - 1
        # Contador global de passos: com --learn-every K o update acontece a
        # cada K transições, sem reiniciar a fase a cada episódio.
        learn_counter = 0
        for ep in range(start_ep, last_ep + 1):
            # Tempo de parede do episódio (coleta + updates + ganchos), exibido
            # junto das demais estatísticas do episódio.
            ep_start = time.monotonic()
            schedule_epsilon(agent, ep, args.epsilon_ramp_episodes, args.epsilon_min)
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
                learn_counter += 1
                loss = (agent.learn()
                        if learn_counter % args.learn_every == 0 else None)
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
            ep_seconds = time.monotonic() - ep_start
            last_total = total
            row = {"episode": ep, "steps": steps, "reward": total, "coins": info.get("coins", 0),
                   "deaths": info.get("deaths", 0),
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
            print(f"episode={ep} reward={total:.2f} coins={row['coins']} "
                  f"deaths={row['deaths']} epsilon={agent.epsilon:.3f} "
                  f"time={ep_seconds:.2f}s")
            if eval_env is not None and ep % args.eval_every == 0:
                metrics = greedy_evaluation(
                    agent, eval_env, args.eval_episodes, args.eval_seed)
                print(f"  eval ep={ep} coins={metrics['coins_mean']:.2f}"
                      f"±{metrics['coins_std']:.2f} "
                      f"reward={metrics['reward_mean']:.2f} "
                      f"deaths={metrics['deaths_mean']:.2f}")
                if tb_writer is not None:
                    for name in ("coins_mean", "coins_std", "reward_mean",
                                 "reward_std", "deaths_mean", "deaths_std"):
                        tb_writer.add_scalar(f"eval/{name}", metrics[name], ep)
                    tb_writer.flush()
            last_done = ep
            if time_budget_reached(start_time, args.max_hours):
                print(f"orçamento de {args.max_hours:g} h atingido; "
                      f"encerrando após o episódio {ep}")
                break
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
                    {"episode": last_done, "reward": last_total,
                     "episodes_done": last_done, "best_reward": best},
                    episodes=last_done)
    if eval_env is not None:
        eval_env.close()
    env.close()

if __name__ == "__main__":
    main()

"""Trainer completo da arena seek_coin.

Este é o trainer “histórico” da arena original. Ele cobre currículo de
oponentes, multiagente (IPPO/MAPPO/HAPPO), vetorização, sobreposição de
coleta/update, TensorBoard e orçamento por tempo de parede. As arenas
`risk` e `economy` usam o trainer genérico `arenas/common/trainer.py`; o
seek_coin mantém o seu porque o laço tem bem mais casos.

Mapa do arquivo (busque pelos nomes):

- utilidades: `resolve_resume_path`, `budget_snapshots`,
  `time_budget_reached`, `epsilon_for_episode`, `schedule_epsilon`;
- argparse: `_DefaultHelpFormatter`, `parse_args`, `tensorboard_run_name`,
  `validate_train_args`;
- treino single-agent: `_train_vectorized`;
- multiagente: `_train_multi_agent`;
- treino on-policy vetorizado: `_train_vectorized_on_policy`;
- entrada: `main`.

As peças reutilizáveis ficam no pacote `arenas.seek_coin.training/`:
`common.py` (helpers vetorizados e orçamento de tempo), `telemetry.py`
(CSV, stats, checkpoint final, imagem da observação), `opponents.py`
(currículo `auto` e load tolerante), `evaluation.py` (avaliação gulosa e
vídeo) e `overlap.py` (coleta CPU sobreposta ao update GPU).

Use o CLI da raiz (`python train.py --arena seek_coin ...`) ou rode este
arquivo diretamente (`python arenas/seek_coin/train.py ...`).
"""

import sys
from pathlib import Path

# Permite rodar o arquivo diretamente (python arenas/seek_coin/train.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse, csv, os, random, time
import numpy as np
import torch
from arenas.seek_coin.arena.environment import ArenaEnv
from arenas.seek_coin.arena.multi_environment import ArenaMultiEnv
from arenas.seek_coin.arena.scenario import Scenario
from arenas.seek_coin.arena.actions import PUSH_INTENSITY
from arenas.seek_coin.arena.config import ArenaConfig
from arenas.seek_coin.arena.skins import list_skins
from arenas.seek_coin.arena.levels import (
    get_level_channel_names, uses_local_view)
from arenas.seek_coin.arena.vector import make_vector_env
from arenas.seek_coin.training.common import (
    _append_rollout_segment,
    _final_observation,
    _rollout_target,
    _worker_final_info,
    time_budget_reached,
)
from arenas.seek_coin.training.evaluation import (
    greedy_evaluation,
    multi_greedy_evaluation,
    record_multi_episode,
)
from arenas.seek_coin.training.opponents import (
    AutoOpponent,
    _try_load_checkpoint,  # reexportado para os testes
)
from arenas.seek_coin.training.overlap import (
    _act_batch_on_cpu,  # reexportado para os testes
    _train_on_policy_overlap,
)
from arenas.seek_coin.training.telemetry import (
    _episode_stats,
    _finish_training,
    _record_episode,
    _training_csv,
    obs_to_image,
)
from arenas.common.yaml_config import add_config_args, apply_yaml_defaults
from arenas.algs.alg_registry import (
    available_algorithms,
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


def _coerce_agent_param(value: str):
    """Converte `KEY=VALUE` em bool/int/float/str para o agente."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_agent_params(items):
    """Transforma `--agent-param k=v` em um dicionário tipado."""
    params = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"--agent-param inválido: {item!r} (use KEY=VALUE)")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--agent-param sem chave: {item!r}")
        params[key] = _coerce_agent_param(value.strip())
    return params


def budget_snapshots(total_episodes):
    """Episódios (absolutos) em que vale salvar um snapshot de orçamento.

    Os orçamentos são pontos fixos de episódios treinados (50/100/150/300),
    usados no protocolo de A/B para comparar fases no mesmo orçamento.
    """
    return {b for b in (50, 100, 150, 300) if b <= total_episodes}


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
        "episode_seconds": "90 (ArenaConfig)",
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
    add_config_args(ap)
    ap.add_argument("--episodes", type=int, default=300,
                    help="episódios de treino (adicionais ao retomar)")
    ap.add_argument("--level", type=int, default=0,
                    help="nível do mapa (0 = histórico; 1 = plataformas móveis; "
                         "2 = areia/gelo; 3 = botões/super moeda; "
                         "4 = nível 3 com visão local)")
    ap.add_argument("--view-size", type=int, default=None,
                    help="lado do recorte egocêntrico no nível 4, em células "
                         "(ímpar; padrão do ArenaConfig: 9)")
    ap.add_argument("--episode-seconds", type=float, default=None,
                    help="duração do episódio em segundos; omitido usa o "
                         "ArenaConfig")
    ap.add_argument("--render", action="store_true",
                    help="abre a janela do renderer durante o treino")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint de onde retomar (pesos, otimizador e contadores)")
    ap.add_argument("--save-every", type=int, default=10,
                    help="salva latest.pt a cada N episódios; 0 desliga o "
                         "save periódico (o checkpoint final ainda é gravado)")
    ap.add_argument("--seed", type=int, default=7,
                    help="seed de random/numpy/torch; cada episódio usa seed+ep")
    ap.add_argument("--opponent", choices=["rule", "random", "coins", "auto"], default="auto",
                    help="política do adversário (auto = currículo que alterna "
                         "random, coins, rule e checkpoints)")
    ap.add_argument("--agents", type=int, default=None,
                    help="número de agentes no modo multiagente (IPPO/MAPPO/HAPPO); "
                         "omitido = 4 com um algoritmo multiagente e 1 nos demais")
    ap.add_argument(
        "--scenario",
        choices=["all-vs-all", "teams", "collectors-vs-pushers"],
        default="all-vs-all",
        help="cenário multiagente: todos-contra-todos (padrão), times ou "
             "times mistos (dentro de cada time, metade coleta e metade "
             "empurra — usa a força de meia arena e recarga de 2 s)")
    ap.add_argument("--teams", type=int, default=2,
                    help="número de times com --scenario teams/"
                         "collectors-vs-pushers; no cenário assimétrico cada "
                         "time precisa ter número par de membros (padrão 2)")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="diretório de checkpoints (saves e leitura do adversário auto)")
    ap.add_argument("--stats", action="store_true",
                    help="coleta estatísticas do treino da CNN (perda, gradiente, Q) "
                         "em logs/stats.csv e no terminal")
    ap.add_argument("--budget-snapshots", action="store_true",
                    help="salva snapshots pesos-only em <checkpoint-dir>/budget/ "
                         "nos episódios 50/100/150/300 (protocolo A/B)")
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
                             if name not in
                             {"ppo_continuous", "ddpg", "td3", "sac"}],
                    default="dqn",
                    help="algoritmo de RL: pacote alg_<nome> em arenas/algs/ "
                         "(dqn = DQN enxuto; doubledqn_per_dueling = Double DQN "
                         "+ PER + dueling; ppo = PPO; eppo = PPO com crítico "
                         "evidencial; ppo_aes = PPO com entropia adaptativa; "
                         "gtr = PPO com região de confiança gaussiana; "
                         "a2c = A2C; ippo/mappo/happo = multiagente)")
    ap.add_argument("--lr", type=float, default=2.5e-4,
                    help="taxa de aprendizado do Adam (na retomada, o otimizador "
                         "restaurado do checkpoint sobrepõe este valor)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="tamanho do batch amostrado do replay buffer por learn()")
    ap.add_argument("--agent-param", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="hiperparâmetro extra do agente (ex.: "
                         "rollout_steps=2048); pode repetir. Usado por "
                         "campanhas de tuning/YAML.")
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
    apply_yaml_defaults(ap, argv, "train")
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
        if args.scenario in ("teams", "collectors-vs-pushers"):
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
    save_every = getattr(args, "save_every", 10)
    if save_every < 0:
        raise SystemExit(
            f"--save-every deve ser >= 0 (recebido {save_every}); "
            "0 desliga o save periódico")
    episode_seconds = getattr(args, "episode_seconds", None)
    if episode_seconds is not None and episode_seconds <= 0:
        raise SystemExit(
            f"--episode-seconds deve ser > 0 (recebido {episode_seconds})")
    view_size = getattr(args, "view_size", None)
    if view_size is not None and (view_size < 3 or view_size % 2 == 0):
        raise SystemExit(
            f"--view-size deve ser ímpar e >= 3 (recebido {view_size})")
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
        if scenario in ("teams", "collectors-vs-pushers"):
            if teams < 2:
                raise SystemExit(
                    f"--teams deve ser >= 2 com --scenario {scenario} "
                    f"(recebido {teams})")
            if agents % teams:
                raise SystemExit(
                    f"--agents ({agents}) deve ser divisível por --teams ({teams})")
            if (scenario == "collectors-vs-pushers"
                    and (agents // teams) % 2):
                raise SystemExit(
                    "--scenario collectors-vs-pushers exige times com número "
                    f"par de membros (agents/teams={agents // teams}); "
                    "ex.: 4 agentes em 2 times, 8 em 2 ou 4 times")
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
        with _training_csv(args.resume) as (f, writer):
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
                    mean_loss = (float(np.mean(losses_since_row))
                                 if losses_since_row else "")
                    ep_seconds = time.monotonic() - worker_ep_start[i]
                    last_total = reward
                    best = _record_episode(
                        f, writer, args, agent, cfg, ckpt_dir, budgets,
                        fired_budgets, best, total_episodes, reward,
                        int(worker_steps[i]), info.get("coins", 0),
                        info.get("deaths", 0), mean_loss, ep_seconds)
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
    _finish_training(agent, cfg, ckpt_dir, total_episodes, last_total, best)


def _conv_weights(agent):
    """Primeira conv para o histograma, inclusive em agentes com ModuleDict."""
    online = agent.online
    features = getattr(online, "features", None)
    if features is None and "actor" in getattr(online, "keys", lambda: [])():
        actor = online["actor"]
        features = getattr(actor, "features", None)
    if features is not None and len(features) > 0:
        return features[0].weight
    return None


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
        with _training_csv(args.resume) as (f, writer):
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
                        obs_to_image(
                            obs[0],
                            n_channels=6 + len(
                                get_level_channel_names(cfg.level))),
                        ep, dataformats="HWC")
                    conv = _conv_weights(agent)
                    if conv is not None:
                        tb_writer.add_histogram("weights/conv1", conv, ep)
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
                    agg = _episode_stats(losses, grad_norms, q_means, q_maxes)
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
    _finish_training(agent, cfg, ckpt_dir, total_episodes, last_total, best)


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
        with _training_csv(args.resume) as (f, writer):
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
    _finish_training(agent, cfg, ckpt_dir, total_episodes, last_total, best)


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
    if args.view_size is not None:
        cfg.view_size = args.view_size
    if args.episode_seconds is not None:
        cfg.episode_seconds = args.episode_seconds
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
        if args.scenario == "all-vs-all":
            scenario = Scenario.all_vs_all(args.agents)
        elif args.scenario == "teams":
            scenario = Scenario.teams(args.agents, args.teams)
        else:
            scenario = Scenario.collectors_vs_pushers(
                args.agents, args.teams)
        cfg.n_agents = scenario.n_agents
        cfg.team_ids = tuple(scenario.team_ids)
        cfg.agent_roles = tuple(scenario.roles)
        if scenario.has_roles:
            # Calibra o empurrão para deslocar o alvo por meia arena e aplica
            # a recarga de 2 s pedida pelo cenário. O teto de velocidade evita
            # que o substep seguinte corte o impulso forte.
            cfg.push_impulse = cfg.push_impulse_for_range(
                cfg.width / 2.0, PUSH_INTENSITY)
            cfg.push_speed_cap = (
                cfg.push_impulse * PUSH_INTENSITY * 1.1)
            cfg.push_cooldown = 2.0
        env = ArenaMultiEnv(cfg, render_mode="human" if args.render else None,
                            skin=args.skin)
    else:
        # `auto` é um currículo gerenciado pelo AutoOpponent; o env inicia com random.
        env = ArenaEnv(
            cfg, render_mode="human" if args.render else None,
            opponent_policy="random" if args.opponent == "auto" else args.opponent,
            skin=args.skin)
    agent_kwargs = dict(lr=args.lr, batch_size=args.batch_size)
    agent_kwargs.update(parse_agent_params(args.agent_param))
    if multi_agent:
        agent_kwargs.update(n_agents=scenario.n_agents,
                            team_ids=tuple(scenario.team_ids))
        if scenario.has_roles and getattr(
                alg.Agent, "accepts_roles", False):
            agent_kwargs["agent_roles"] = tuple(scenario.roles)
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
        episodes_done = payload.get("episodes") or 0
        start_ep = int(episodes_done) + 1
        best = float(payload.get("extra", {}).get("best_reward", -float("inf")))
        saved_level = int((payload.get("config") or {}).get("level", 0))
        if saved_level != int(cfg.level):
            raise SystemExit(
                f"Nível do checkpoint ({saved_level}) difere do --level "
                f"({cfg.level}); retome com o mesmo nível.")
        if uses_local_view(cfg.level):
            saved_view = int(
                (payload.get("config") or {}).get("view_size", cfg.view_size))
            if saved_view != int(cfg.view_size):
                raise SystemExit(
                    f"view_size do checkpoint ({saved_view}) difere do "
                    f"--view-size ({cfg.view_size}); retome com a mesma "
                    "janela de visão.")
        if multi_agent:
            # O cenário não pode mudar entre execuções: os buffers e a
            # interpretação dos canais de aliados/inimigos dependem dele.
            saved = payload.get("config") or {}
            saved_n = int(saved.get("n_agents", 0) or 0)
            saved_teams = tuple(saved.get("team_ids") or ())
            saved_roles = tuple(saved.get("agent_roles") or ())
            if (saved_n != scenario.n_agents
                    or saved_teams != tuple(scenario.team_ids)
                    or saved_roles != tuple(scenario.roles)):
                raise SystemExit(
                    f"Cenário do checkpoint incompatível (n_agents={saved_n}, "
                    f"team_ids={saved_teams}, roles={saved_roles}); esperado "
                    f"n_agents={scenario.n_agents}, "
                    f"team_ids={tuple(scenario.team_ids)}, "
                    f"roles={tuple(scenario.roles)}.")
        if auto is not None:
            auto.init_from_disk()
    last_ep = start_ep + args.episodes - 1
    # Orçamentos contam episódios absolutos (retomada continua a numeração),
    # então o snapshot compara fases com o mesmo total de treino.
    budgets = (budget_snapshots(last_ep)
               if getattr(args, "budget_snapshots", False) else set())
    # Orçamentos já cruzados antes desta execução (retomada) não são regravados.
    fired_budgets = {b for b in budgets if b <= start_ep - 1}
    Path("logs").mkdir(exist_ok=True)
    # parents=True: o protocolo de A/B usa caminhos aninhados
    # (ex.: checkpoints/ab/fase0-branch/seed7-100).
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if multi_agent:
        _train_multi_agent(args, cfg, scenario, env, agent, ckpt_dir,
                           start_ep, last_ep, best, budgets, start_time)
        return
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
    with _training_csv(args.resume) as (f, writer):
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
            if args.stats or tb_writer is not None:
                agg = _episode_stats(losses, grad_norms, q_means, q_maxes)
            if tb_writer is not None:
                tb_writer.add_scalar("train/reward", total, ep)
                tb_writer.add_scalar("train/coins", info.get("coins", 0), ep)
                tb_writer.add_scalar("train/deaths", info.get("deaths", 0), ep)
                for name in ("mean_loss", "min_loss", "max_loss", "grad_mean", "grad_max", "q_mean", "q_max"):
                    tb_writer.add_scalar(f"train/{name}", agg[name], ep)
                tb_writer.add_scalar("train/epsilon", agent.epsilon, ep)
                tb_writer.add_scalar("train/buffer_size", len(agent.buffer), ep)
                tb_writer.add_image(
                    "env/observation",
                    obs_to_image(
                        obs, n_channels=5 + len(get_level_channel_names(cfg.level))),
                    ep, dataformats="HWC")
                conv = _conv_weights(agent)
                if conv is not None:
                    tb_writer.add_histogram("weights/conv1", conv, ep)
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
            prev_best = best
            best = _record_episode(
                f, writer, args, agent, cfg, ckpt_dir, budgets,
                fired_budgets, best, ep, total, steps, info.get("coins", 0),
                info.get("deaths", 0),
                float(np.mean(losses)) if losses else "", ep_seconds)
            if auto is not None and best > prev_best:
                auto.update_best(agent)
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
    _finish_training(agent, cfg, ckpt_dir, last_done, last_total, best)
    if eval_env is not None:
        eval_env.close()
    env.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Duelo multiagente: um time treinado com MAPPO contra um com IPPO.

O `play.py` só aceita um checkpoint no modo multiagente (modo espectador).
Aqui dois checkpoints multiagentes controlam lados opostos do mesmo
`ArenaMultiEnv`: os agentes do time `--mappo-team` usam o checkpoint MAPPO e
os demais usam o IPPO. Os dois checkpoints precisam ter o mesmo formato de
observação (`frame_stack`/`frame_skip`) e o mesmo cenário
(`n_agents`/`team_ids`), senão o duelo não é possível.

A política é gulosa e as seeds são fixas, então o mesmo par de checkpoints
produz a mesma partida — útil para comparar snapshots ao longo do treino.

Uso:
    python scripts/duel_multiagent.py \
        --mappo-checkpoint checkpoints/mappo_teams4_3h/best.pt \
        --ippo-checkpoint checkpoints/ippo_teams4_3h/best.pt \
        --record logs/mappo_teams4_3h/videos/duel.mp4
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.config import ArenaConfig
from arena.multi_environment import ArenaMultiEnv
from arena.scenario import Scenario
from arena.skins import list_skins
from algs.alg_registry import (
    algorithm_from_checkpoint,
    apply_env_config,
    load_checkpoint,
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mappo-checkpoint", required=True,
                    help="checkpoint multiagente treinado com MAPPO")
    ap.add_argument("--ippo-checkpoint", required=True,
                    help="checkpoint multiagente treinado com IPPO")
    ap.add_argument("--mappo-team", type=int, default=0,
                    help="time (team_id) controlado pelo checkpoint MAPPO; "
                         "os demais times usam o IPPO")
    ap.add_argument("--episodes", type=int, default=1,
                    help="partidas a jogar (padrão: 1)")
    ap.add_argument("--seed", type=int, default=1000,
                    help="seed base; o episódio i usa seed + i")
    ap.add_argument("--render", action="store_true",
                    help="abre a janela durante o duelo")
    ap.add_argument("--record", default=None, metavar="ARQUIVO.mp4",
                    help="grava o duelo em vídeo (com SDL_VIDEODRIVER=dummy "
                         "grava sem janela)")
    ap.add_argument("--fps", type=int, default=None,
                    help="fps do vídeo (padrão: fps do ambiente)")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (padrão: clássico)")
    ap.add_argument("--device", default="cpu",
                    help="device das redes (padrão: cpu, para não disputar "
                         "a GPU com o treino em andamento)")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes deve ser >= 1")
    return args


def checkpoint_config(meta):
    """Campos que precisam coincidir entre os dois checkpoints."""
    config = meta.get("config") or {}
    return {
        "obs_shape": tuple(meta.get("obs_shape") or ()),
        "n_actions": meta.get("n_actions"),
        "frame_stack": config.get("frame_stack"),
        "frame_skip": config.get("frame_skip"),
        "n_agents": config.get("n_agents"),
        "team_ids": tuple(config.get("team_ids") or ()),
        "agent_roles": tuple(config.get("agent_roles") or ()),
    }


def load_metadata(path, label, attempts=5, delay=3.0):
    """Lê o checkpoint com retentativas: ele pode estar sendo salvo agora."""
    last = None
    for _ in range(attempts):
        try:
            alg, meta = algorithm_from_checkpoint(path)
            return alg, meta
        except Exception as exc:  # arquivo parcial, ausente ou inválido
            last = exc
            time.sleep(delay)
    raise SystemExit(f"Checkpoint {label} ilegível ({path}): {last}")


def build_agent(alg, path, env, scenario, device, attempts=5, delay=3.0):
    last = None
    for _ in range(attempts):
        try:
            kwargs = dict(n_agents=scenario.n_agents,
                          team_ids=tuple(scenario.team_ids), device=device)
            if scenario.has_roles and getattr(
                    alg.Agent, "accepts_roles", False):
                kwargs["agent_roles"] = tuple(scenario.roles)
            agent = alg.Agent(
                env.observation_space.shape, env.action_space.n, **kwargs)
            load_checkpoint(path, agent, resume=False)
            return agent
        except Exception as exc:
            last = exc
            time.sleep(delay)
    raise SystemExit(f"Não foi possível carregar {path}: {last}")


def record_frame(writer):
    """Captura o frame atual da janela e anexa ao vídeo."""
    import pygame
    import pygame.surfarray  # noqa: F401
    frame = np.transpose(
        pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
    writer.append_data(frame)


def validate_pair(mappo_meta, ippo_meta):
    a = checkpoint_config(mappo_meta)
    b = checkpoint_config(ippo_meta)
    # O duelo nasceu como MAPPO x IPPO, mas serve para qualquer par de
    # algoritmos multiagentes (ex.: MAPPO x HAPPO, HAPPO x IPPO).
    allowed = {"mappo", "ippo", "happo"}
    for flag, meta in (("--mappo-checkpoint", mappo_meta),
                       ("--ippo-checkpoint", ippo_meta)):
        algorithm = meta.get("algorithm")
        if algorithm not in allowed:
            raise SystemExit(
                f"{flag} precisa ser multiagente ({sorted(allowed)}); "
                f"algorithm={algorithm!r}")
    if a != b:
        diff = ", ".join(f"{k}: {a[k]!r} != {b[k]!r}"
                         for k in a if a[k] != b[k])
        raise SystemExit(
            "Checkpoints incompatíveis para o duelo "
            f"(MAPPO vs IPPO): {diff}")
    if (a["n_agents"] or 0) < 2:
        raise SystemExit("Os checkpoints precisam ser multiagentes "
                         "(n_agents >= 2)")
    return a


def main(argv=None):
    args = parse_args(argv)
    mappo_alg, mappo_meta = load_metadata(
        args.mappo_checkpoint, "MAPPO")
    ippo_alg, ippo_meta = load_metadata(args.ippo_checkpoint, "IPPO")
    expected = validate_pair(mappo_meta, ippo_meta)

    cfg = ArenaConfig()
    apply_env_config(cfg, mappo_meta)
    scenario = Scenario.from_config(cfg)
    if args.mappo_team not in set(scenario.team_ids):
        raise SystemExit(
            f"--mappo-team {args.mappo_team} não existe no cenário "
            f"{tuple(scenario.team_ids)}")
    mappo_mask = np.array(
        [scenario.team_of(i) == args.mappo_team
         for i in range(scenario.n_agents)], dtype=bool)
    if mappo_mask.all() or not mappo_mask.any():
        raise SystemExit(
            "O time do MAPPO precisa dividir o cenário com o time do IPPO")

    render_mode = "human" if (args.render or args.record) else None
    env = ArenaMultiEnv(cfg, scenario=scenario, render_mode=render_mode,
                        skin=args.skin)
    mappo_agent = build_agent(
        mappo_alg, args.mappo_checkpoint, env, scenario, args.device)
    ippo_agent = build_agent(
        ippo_alg, args.ippo_checkpoint, env, scenario, args.device)
    label_a = str(mappo_meta.get("algorithm", "A")).upper()
    label_b = str(ippo_meta.get("algorithm", "B")).upper()
    print(f"duelo: {label_a} time {args.mappo_team} "
          f"({expected['n_agents']} agentes, fs={expected['frame_stack']}, "
          f"fk={expected['frame_skip']}) vs {label_b} "
          f"times {sorted(set(scenario.team_ids) - {args.mappo_team})}")

    video_writer = None
    if args.record:
        try:
            import imageio
        except ImportError as exc:
            raise SystemExit(
                "Gravar vídeo requer: pip install imageio imageio-ffmpeg"
            ) from exc
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            args.record, fps=args.fps or cfg.fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", "23",
                           "-movflags", "+faststart"])

    def actions_for(obs):
        # Os dois agentes recebem o lote completo (n_agents, C, H, W): o
        # HAPPO indexa atores/máscaras pelo índice global do agente, então
        # passar só o subconjunto do time quebraria. Cada lado contribui só
        # com as suas posições no vetor final.
        actions_a = mappo_agent.act_batch(obs, greedy=True)
        actions_b = ippo_agent.act_batch(obs, greedy=True)
        actions = np.empty(scenario.n_agents, dtype=np.int64)
        actions[mappo_mask] = actions_a[mappo_mask]
        actions[~mappo_mask] = actions_b[~mappo_mask]
        return actions

    results = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            if video_writer is not None:
                env.render()
                record_frame(video_writer)
            totals = np.zeros(scenario.n_agents, dtype=np.float64)
            steps = 0
            while True:
                obs, rewards, terminated, truncated, info = env.step(
                    actions_for(obs))
                totals += rewards
                steps += 1
                if video_writer is not None:
                    env.render()
                    record_frame(video_writer)
                if bool(np.all(truncated)):
                    break
            by_team = {row["team"]: row for row in info.get("teams", [])}
            mappo_row = by_team[args.mappo_team]
            ippo_rows = [row for team, row in by_team.items()
                         if team != args.mappo_team]
            ippo_coins = sum(row["coins"] for row in ippo_rows)
            # `info["teams"]` traz a recompensa do último passo; o placar do
            # duelo usa o total acumulado no episódio.
            mappo_reward = float(totals[mappo_mask].mean())
            ippo_reward = float(totals[~mappo_mask].mean())
            winner = (label_a if mappo_row["coins"] > ippo_coins else
                      label_b if ippo_coins > mappo_row["coins"] else "empate")
            print(f"episódio {ep + 1} ({steps} passos): "
                  f"{label_a} time {args.mappo_team} "
                  f"{mappo_row['coins']} moedas "
                  f"(reward {mappo_reward:.2f}) | "
                  f"{label_b} {ippo_coins} moedas "
                  f"(reward {ippo_reward:.2f}) -> {winner}")
            results.append({
                "episode": ep + 1, "steps": steps,
                "mappo_coins": mappo_row["coins"],
                "ippo_coins": ippo_coins, "winner": winner,
            })
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"vídeo salvo em {args.record}")
        env.close()
    return results


if __name__ == "__main__":
    main()

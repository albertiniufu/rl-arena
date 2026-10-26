import argparse
import time

import numpy as np

from agents.baseline import POLICIES, PolicyContext, make_policy
from arena.actions import N_ACTIONS
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.skins import list_skins
from alg_registry import (
    CheckpointPolicy,
    algorithm_from_checkpoint,
    apply_env_config,
    load_checkpoint,
)


def record_frame(writer):
    """Captura o frame atual da janela e anexa ao vídeo (H, W, 3 uint8)."""
    import pygame
    import pygame.surfarray  # noqa: F401 — submódulo precisa de import explícito
    frame = np.transpose(pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
    writer.append_data(frame)


def needs_render(args):
    """Humano em qualquer lado e gravação exigem janela."""
    return bool(args.render or args.player == "human"
                or args.opponent == "human" or args.record)


def validate_play_args(args):
    """Recusa dois controles humanos: os dois leriam o mesmo teclado."""
    if args.player == "human" and args.opponent == "human":
        raise SystemExit("Use apenas um controle humano: escolha --player human "
                         "ou --opponent human, não ambos.")


def resolve_opponent(spec, cfg, player_frame=None):
    """Resolve `--opponent` para uma política aceita pelo ArenaEnv.

    `spec` pode ser um nome de `agents.baseline` (`rule`, `random`, `coins`),
    `human` (teclado) ou o caminho de um checkpoint `.pt`, que vira uma
    `CheckpointPolicy` gulosa com a config de observação gravada no arquivo.

    `player_frame` é o par `(frame_stack, frame_skip)` já fixado pelo
    checkpoint do jogador, quando houver. Dois checkpoints no mesmo ambiente
    precisam do mesmo formato de observação; a incompatibilidade falha antes
    de criar o ambiente.
    """
    if spec == "human":
        from agents.human import HumanPolicy
        return HumanPolicy()
    if spec in POLICIES:
        return make_policy(spec)

    try:
        opp_alg, meta = algorithm_from_checkpoint(spec)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Checkpoint de adversário inválido: {exc}") from exc

    opp_cfg = apply_env_config(ArenaConfig(), meta)
    opp_frame = (opp_cfg.frame_stack, opp_cfg.frame_skip)
    if player_frame is not None and opp_frame != tuple(player_frame):
        raise SystemExit(
            f"Adversário {spec} usa frame_stack={opp_frame[0]}, "
            f"frame_skip={opp_frame[1]}, mas o jogador usa "
            f"frame_stack={player_frame[0]}, frame_skip={player_frame[1]}; "
            "use checkpoints com o mesmo formato de observação.")

    apply_env_config(cfg, meta)
    obs_shape = (5 * cfg.frame_stack, cfg.height // cfg.cell, cfg.width // cfg.cell)
    agent = opp_alg.Agent(obs_shape, N_ACTIONS)
    load_checkpoint(spec, agent, resume=False)
    agent.epsilon = 0.0
    return CheckpointPolicy(agent)


def parse_args(argv=None):
    """Argumentos da CLI; separado de `main` para poder ser testado."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--opponent", default="rule",
                    help="adversário: rule/random/coins, human (teclado) ou "
                         "caminho de um checkpoint .pt")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (arena/renderer_<skin>.py; padrão: renderer.py)")
    ap.add_argument("--player", choices=["coins", "random", "human"], default=None,
                    help="política baseline para o jogador em vez do checkpoint DQN "
                         "(ex.: coins contra coins: --player coins --opponent coins; "
                         "humano no teclado: --player human)")
    ap.add_argument("--record", default=None, metavar="ARQUIVO.mp4",
                    help="grava a sessão em vídeo (ativa o render; com "
                         "SDL_VIDEODRIVER=dummy grava sem janela)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    validate_play_args(args)
    render_mode = "human" if needs_render(args) else None
    cfg = ArenaConfig()
    alg = meta = None
    player_frame = None
    if args.player is None:
        # Jogador DQN: o checkpoint identifica o algoritmo (pacote alg_*) e a
        # config gravada define o formato da observação (frame stack/skip).
        # Com política baseline/humana não há checkpoint e a config fica nos
        # defaults — ou na do checkpoint do adversário, se ele existir.
        try:
            alg, meta = algorithm_from_checkpoint(args.checkpoint)
        except (ValueError, OSError) as exc:
            raise SystemExit(f"Checkpoint inválido: {exc}") from exc
        apply_env_config(cfg, meta)
        player_frame = (cfg.frame_stack, cfg.frame_skip)
    opponent_policy = resolve_opponent(args.opponent, cfg, player_frame=player_frame)
    env = ArenaEnv(cfg, render_mode=render_mode, opponent_policy=opponent_policy, skin=args.skin)
    obs, _ = env.reset(seed=1)
    video_writer = None
    if args.record:
        try:
            import imageio
        except ImportError as exc:
            raise SystemExit("Gravar vídeo requer: pip install imageio imageio-ffmpeg") from exc
        # H.264 + yuv420p + crf 23 + faststart: compatível com WhatsApp e
        # outros players, com tamanho razoável (o padrão do imageio é crf 5,
        # quase sem perdas — arquivos enormes).
        video_writer = imageio.get_writer(
            args.record, fps=env.cfg.fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart"])
    # Sem --player, comportamento antigo: política gulosa do checkpoint DQN.
    player_policy = None
    if args.player == "human":
        from agents.human import HumanPolicy
        player_policy = HumanPolicy()
        print("Controles do jogador: setas/AD andar | espaço/↑/W pular | "
              "Q empurrar ← | E empurrar →")
    elif args.player:
        player_policy = make_policy(args.player)
    if args.opponent == "human":
        print("Controles do adversário: setas/AD andar | espaço/↑/W pular | "
              "Q empurrar ← | E empurrar →")
    agent = None
    if player_policy is None:
        agent = alg.Agent(env.observation_space.shape, env.action_space.n)
        load_checkpoint(args.checkpoint, agent, resume=False)
        agent.epsilon = 0.0
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep + 1)
            if video_writer is not None:
                env.render()  # estado inicial do episódio também entra no vídeo
                record_frame(video_writer)
            while True:
                if player_policy is not None:
                    # `self_character` permite ao humano continuar andando no
                    # ar em vez de repetir o pulo; as baselines ignoram os campos.
                    ctx = PolicyContext(
                        obs=obs, rng=env.np_random,
                        self_character=env.engine.characters[0],
                        player_character=env.engine.characters[0])
                    action = player_policy.act(ctx)
                else:
                    action = agent.act(obs, greedy=True)
                obs, _, terminated, truncated, info = env.step(action)
                if video_writer is not None:
                    record_frame(video_writer)
                if terminated or truncated: break
            print(f"episode {ep+1}: coins={info['coins']} deaths={info['deaths']}")
            if env.render_mode is None: time.sleep(.1)
    finally:
        # Fechar mesmo se a janela for fechada no meio (SystemExit).
        if video_writer is not None:
            video_writer.close()
            print(f"vídeo salvo em {args.record}")
    env.close()

if __name__ == "__main__":
    main()

import argparse
import time

import numpy as np

from agents.baseline import PolicyContext, make_policy
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from arena.skins import list_skins
from dqn.agent import DQNAgent
from dqn.checkpoint import load_checkpoint


def record_frame(writer):
    """Captura o frame atual da janela e anexa ao vídeo (H, W, 3 uint8)."""
    import pygame
    import pygame.surfarray  # noqa: F401 — submódulo precisa de import explícito
    frame = np.transpose(pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
    writer.append_data(frame)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--opponent", choices=["rule", "random", "coins"], default="rule")
    ap.add_argument("--skin", choices=list_skins(), default=None,
                    help="visual do renderer (arena/renderer_<skin>.py; padrão: renderer.py)")
    ap.add_argument("--player", choices=["coins", "random", "human"], default=None,
                    help="política baseline para o jogador em vez do checkpoint DQN "
                         "(ex.: coins contra coins: --player coins --opponent coins; "
                         "humano no teclado: --player human)")
    ap.add_argument("--record", default=None, metavar="ARQUIVO.mp4",
                    help="grava a sessão em vídeo (ativa o render; com "
                         "SDL_VIDEODRIVER=dummy grava sem janela)")
    args = ap.parse_args()
    # Humano e gravação só fazem sentido com janela; ativam o render.
    render_mode = "human" if (args.render or args.player == "human" or args.record) else None
    env = ArenaEnv(ArenaConfig(), render_mode=render_mode, opponent_policy=args.opponent, skin=args.skin)
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
        print("Controles: setas/AD andar | espaço/↑/W pular | Q empurrar ← | E empurrar →")
    elif args.player:
        player_policy = make_policy(args.player)
    agent = None
    if player_policy is None:
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)
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
                    ctx = PolicyContext(obs=obs, rng=env.np_random)
                    action = player_policy.act(ctx)
                else:
                    action = agent.act(obs, greedy=True)
                obs, _, terminated, truncated, info = env.step(action)
                if video_writer is not None:
                    record_frame(video_writer)
                if terminated or truncated: break
            print(f"episode {ep+1}: coins={info['coins']} deaths={info['deaths']} eliminations={info['eliminations']}")
            if env.render_mode is None: time.sleep(.1)
    finally:
        # Fechar mesmo se a janela for fechada no meio (SystemExit).
        if video_writer is not None:
            video_writer.close()
            print(f"vídeo salvo em {args.record}")
    env.close()

if __name__ == "__main__":
    main()

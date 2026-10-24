import argparse
import time
from agents.baseline import PolicyContext, make_policy
from arena.environment import ArenaEnv
from arena.config import ArenaConfig
from dqn.agent import DQNAgent
from dqn.checkpoint import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--opponent", choices=["rule", "random", "coins"], default="rule")
    ap.add_argument("--player", choices=["coins", "random", "human"], default=None,
                    help="política baseline para o jogador em vez do checkpoint DQN "
                         "(ex.: coins contra coins: --player coins --opponent coins; "
                         "humano no teclado: --player human)")
    args = ap.parse_args()
    # Humano só faz sentido com janela; ativa o render automaticamente.
    render_mode = "human" if (args.render or args.player == "human") else None
    env = ArenaEnv(ArenaConfig(), render_mode=render_mode, opponent_policy=args.opponent)
    obs, _ = env.reset(seed=1)
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
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep + 1)
        while True:
            if player_policy is not None:
                ctx = PolicyContext(obs=obs, rng=env.np_random)
                action = player_policy.act(ctx)
            else:
                action = agent.act(obs, greedy=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated: break
        print(f"episode {ep+1}: coins={info['coins']} deaths={info['deaths']} eliminations={info['eliminations']}")
        if not args.render: time.sleep(.1)
    env.close()

if __name__ == "__main__":
    main()

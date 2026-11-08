"""Modo jogável em texto do jogo de conquista.

Uso (a partir da raiz do repositório):

    python -m games.risk.play              # humano (J1) contra bot
    python -m games.risk.play --players 4  # humano contra 3 bots
    python -m games.risk.play --demo       # bots contra bots
"""
from __future__ import annotations

import argparse

from games.risk.agents import RiskBot
from games.risk.board import (
    CONTINENTS,
    CONTINENT_BONUS,
    TERRITORIES,
    TERRITORY_INDEX,
)
from games.risk.game import RiskGame

HELP = """comandos:
  p <territorio>            reforçar (colocar 1 exército)
  a <origem> <destino>      atacar território vizinho
  f <origem> <destino>      fortificar (mover 1 exército)
  fim                       encerrar a fase atual
  ajuda                     mostrar esta ajuda
  sair                      encerrar o jogo"""


def resolve_territory(token: str) -> str:
    key = token.strip().lower()
    exact = [name for name in TERRITORIES if name.lower() == key]
    if exact:
        return exact[0]
    matches = [name for name in TERRITORIES if name.lower().startswith(key)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"território desconhecido: {token}")
    raise ValueError("território ambíguo: " + ", ".join(matches))


def print_board(game: RiskGame) -> None:
    print()
    for continent, names in CONTINENTS.items():
        owners = [game.owner[TERRITORY_INDEX[name]] for name in names]
        bonus = CONTINENT_BONUS[continent]
        full = owners and len(set(owners)) == 1
        print(f"{continent} (bônus {bonus}{' — dominado' if full else ''})")
        for name in names:
            index = TERRITORY_INDEX[name]
            print(f"  {name:14s} J{game.owner[index] + 1} "
                  f"x{game.armies[index]}")
    print()


def print_status(game: RiskGame) -> None:
    print(f"== Turno {game.turn} | Jogador {game.current_player + 1} | "
          f"fase: {game.phase} | reforços: {game.reinforcements_left} ==")


def encode(game: RiskGame, src: str, dst: str) -> int:
    return (game.n_territories + 1
            + TERRITORY_INDEX[src] * game.n_territories
            + TERRITORY_INDEX[dst])


def human_turn(game: RiskGame) -> None:
    while game.current_player == 0 and game.winner is None:
        print_status(game)
        try:
            parts = input("> ").strip().split()
        except EOFError:
            raise SystemExit(0)
        if not parts:
            continue
        command = parts[0].lower()
        try:
            if command in ("ajuda", "help", "?"):
                print(HELP)
                continue
            if command in ("sair", "quit", "exit"):
                raise SystemExit(0)
            if command in ("p", "reforcar", "reforçar"):
                territory = resolve_territory(parts[1])
                game.apply_action(1 + TERRITORY_INDEX[territory])
                print(f"reforçou {territory}.")
            elif command in ("a", "atacar"):
                src = resolve_territory(parts[1])
                dst = resolve_territory(parts[2])
                battle = game.apply_action(encode(game, src, dst))
                print_battle(game, battle)
            elif command in ("f", "fortificar"):
                src = resolve_territory(parts[1])
                dst = resolve_territory(parts[2])
                game.apply_action(encode(game, src, dst))
                print(f"fortificou {src} -> {dst}.")
            elif command in ("fim", "end", "passar"):
                game.apply_action(0)
            else:
                print(HELP)
        except (ValueError, IndexError) as error:
            print(f"erro: {error}")
        if game.winner is not None:
            print(f"\nJogador {game.winner + 1} conquistou o mundo!")


def print_battle(game: RiskGame, battle: dict | None) -> None:
    if not battle:
        return
    src = TERRITORIES[battle["src"]]
    dst = TERRITORIES[battle["dst"]]
    print(f"ataque {src} -> {dst}: atacante {battle['attacker_dice']} "
          f"x defensor {battle['defender_dice']} | "
          f"perdas {battle['attacker_losses']}/{battle['defender_losses']}"
          + (" | território conquistado!" if battle["conquered"] else ""))
    if game.winner is not None:
        print(f"\nJogador {game.winner + 1} conquistou o mundo!")


def bot_turn(game: RiskGame, bot: RiskBot, verbose: bool = True) -> None:
    while game.current_player != 0 and game.winner is None:
        player = game.current_player
        action = bot.act(game)
        if action is None:
            break
        kind, src, dst = game.decode(action)
        if game.phase == "attack" and action != 0:
            battle = game.apply_action(action)
            if verbose:
                print(f"J{player + 1} " + _battle_line(battle))
        else:
            game.apply_action(action)
            if verbose and action != 0:
                if kind == "place":
                    print(f"J{player + 1} reforçou "
                          f"{TERRITORIES[src]}.")
                elif kind == "move":
                    print(f"J{player + 1} fortificou "
                          f"{TERRITORIES[src]} -> {TERRITORIES[dst]}.")
    if game.winner is not None:
        print(f"\nJogador {game.winner + 1} conquistou o mundo!")


def _battle_line(battle: dict) -> str:
    return (f"atacou {TERRITORIES[battle['src']]} -> "
            f"{TERRITORIES[battle['dst']]} "
            f"{battle['attacker_dice']}x{battle['defender_dice']}"
            + (" (conquistou)" if battle["conquered"] else ""))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=2,
                        help="número de jogadores (humano = J1)")
    parser.add_argument("--armies", type=int, default=20,
                        help="exércitos iniciais por jogador")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--demo", action="store_true",
                        help="bots contra bots, sem entrada humana")
    args = parser.parse_args(argv)

    game = RiskGame(n_players=args.players,
                    initial_armies=args.armies, seed=args.seed)
    bots = [RiskBot(seed=args.seed + player)
            for player in range(args.players)]
    print_board(game)

    if args.demo:
        while game.winner is None and game.turn <= args.max_turns:
            print_status(game)
            player = game.current_player
            bot = bots[player]
            while game.current_player == player and game.winner is None:
                action = bot.act(game)
                if action is None:
                    break
                if game.phase == "attack" and action != 0:
                    battle = game.apply_action(action)
                    print(f"J{player + 1} " + _battle_line(battle))
                else:
                    kind, src, dst = game.decode(action)
                    game.apply_action(action)
                    if action != 0 and kind == "place":
                        print(f"J{player + 1} reforçou "
                              f"{TERRITORIES[src]}.")
                    elif action != 0 and kind == "move":
                        print(f"J{player + 1} fortificou "
                              f"{TERRITORIES[src]} -> {TERRITORIES[dst]}.")
        if game.winner is not None:
            print(f"\nJogador {game.winner + 1} conquistou o mundo!")
        else:
            print(f"\nLimite de {args.max_turns} turnos atingido.")
        return

    while game.winner is None:
        human_turn(game)
        if game.winner is not None:
            break
        bot_turn(game, bots[game.current_player])


if __name__ == "__main__":
    main()

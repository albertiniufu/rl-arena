"""Grade semântica da arena, opcionalmente egocêntrica por agente.

A versão legada (`ArenaEnv`) usa 5 canais e sempre trata o índice 0 como o
jogador: canal 1 = jogador, canal 2 = todos os demais. O ambiente multiagente
usa 6 canais e renderiza a grade **do ponto de vista de cada agente**:

    0 plataformas
    1 o próprio agente
    2 inimigos (outros times; em todos-contra-todos, todos os demais)
    3 moedas
    4 mortos/respawn (todos os personagens mortos)
    5 aliados (membros do mesmo time, excluindo o próprio agente)
"""
import numpy as np


def render_grid(engine, cfg, viewer=0, channels=5, team_ids=None):
    """Monta a grade `(channels, H, W)` vista pelo agente `viewer`.

    Com `team_ids=None` reproduz exatamente a grade legada do `ArenaEnv`
    (canal 1 = índice `viewer`, canal 2 = os demais). Com `team_ids`, separa
    aliados (canal 5, se existir) de inimigos (canal 2).
    """
    cell, gh, gw = cfg.cell, cfg.height // cfg.cell, cfg.width // cfg.cell
    g = np.zeros((channels, gh, gw), dtype=np.float32)
    for x, y, w, h in engine.platforms:
        c0, c1 = max(0, int(x // cell)), min(gw, int((x + w) // cell))
        r0, r1 = max(0, int(y // cell)), min(
            gh, max(int((y + h) // cell), int(y // cell) + 1))
        g[0, r0:r1, c0:c1] = 1
    for i, c in enumerate(engine.characters):
        if c.alive:
            if team_ids is None:
                channel = 1 if i == viewer else 2
            elif i == viewer:
                channel = 1
            elif team_ids[i] == team_ids[viewer]:
                channel = 5 if channels > 5 else 2
            else:
                channel = 2
            y = c.y - cfg.player_size / 2
        else:
            channel = 4
            y = cfg.height - 5
        col, row = int(c.x // cell), int(y // cell)
        col = 0 if col < 0 else (gw - 1 if col >= gw else col)
        row = 0 if row < 0 else (gh - 1 if row >= gh else row)
        g[channel, row, col] = 1.0
    for coin in engine.coins:
        col, row = int(coin.x // cell), int(coin.y // cell)
        col = 0 if col < 0 else (gw - 1 if col >= gw else col)
        row = 0 if row < 0 else (gh - 1 if row >= gh else row)
        g[3, row, col] = 1.0
    return g

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

No nível 4 (`levels.uses_local_view`) a grade é **recortada** em uma janela
`view_size x view_size` centrada na célula do agente; tudo o que fica fora da
janela simplesmente não entra na observação. Fora dos limites do mapa a
janela é preenchida com zeros (fundo). O recorte é aplicado depois de montar
a grade completa, então os canais e a semântica das entidades não mudam.
O nível 5 usa o mesmo recorte e acrescenta o canal `hole` (buracos do chão
procedural).
"""
import numpy as np

from .levels import get_level_channel_names, uses_local_view

# Valor das células fora dos limites do mapa no recorte local.
LOCAL_VIEW_PAD = 0.0


def observation_grid_shape(cfg):
    """Devolve `(linhas, colunas)` da grade **observada** pelo agente.

    Nos níveis com visão local é `(view_size, view_size)`; nos demais, o mapa
    inteiro em células. `view_size` precisa ser ímpar para o agente ficar
    exatamente no centro; valores inválidos falham aqui, antes de montar a
    rede ou carregar um checkpoint.
    """
    if uses_local_view(cfg.level):
        size = int(cfg.view_size)
        if size < 3 or size % 2 == 0:
            raise ValueError(
                f"view_size deve ser ímpar e >= 3 no nível {cfg.level} "
                f"(recebido {size})")
        return size, size
    return cfg.height // cfg.cell, cfg.width // cfg.cell


def _cell(cfg, x, y):
    """Célula `(coluna, linha)` de um ponto em pixels (truncamento)."""
    return int(x // cfg.cell), int(y // cfg.cell)


def render_grid(engine, cfg, viewer=0, channels=5, team_ids=None):
    """Monta a grade `(channels, H, W)` vista pelo agente `viewer`.

    Com `team_ids=None` reproduz exatamente a grade legada do `ArenaEnv`
    (canal 1 = índice `viewer`, canal 2 = os demais). Com `team_ids`, separa
    aliados (canal 5, se existir) de inimigos (canal 2).

    No nível 4 `H = W = view_size` e a janela é centrada na célula do agente
    `viewer`; cada agente vê o próprio recorte (visão egocêntrica).
    """
    cell = cfg.cell
    map_h, map_w = cfg.height // cfg.cell, cfg.width // cfg.cell
    gh, gw = observation_grid_shape(cfg)
    local = uses_local_view(cfg.level)
    g = np.zeros((channels, gh, gw), dtype=np.float32)
    if local:
        # A câmera é ancorada na mesma célula usada para desenhar o agente
        # (centro do corpo, não os pés). `view_size` é ímpar, então a célula
        # do agente cai exatamente no centro da janela.
        anchor = engine.characters[viewer]
        col, row = _cell(cfg, anchor.x, anchor.y - cfg.player_size / 2)
        col0, row0 = col - gw // 2, row - gh // 2
    else:
        col0 = row0 = 0

    def fill_rect(channel, x, y, w, h):
        c0 = max(0, int(x // cell)) - col0
        c1 = min(map_w, max(int((x + w) // cell), int(x // cell) + 1)) - col0
        r0 = max(0, int(y // cell)) - row0
        r1 = min(map_h, max(int((y + h) // cell), int(y // cell) + 1)) - row0
        c0, c1 = max(0, c0), min(gw, c1)
        r0, r1 = max(0, r0), min(gh, r1)
        if c0 < c1 and r0 < r1:
            g[channel, r0:r1, c0:c1] = 1.0

    def put_cell(channel, col, row):
        """Desenha uma célula; no recorte local, ignora o que está fora."""
        if local:
            col, row = col - col0, row - row0
            if 0 <= col < gw and 0 <= row < gh:
                g[channel, row, col] = 1.0
        else:
            # Grade legada: entidades fora do mapa são presas na borda.
            col = 0 if col < 0 else (gw - 1 if col >= gw else col)
            row = 0 if row < 0 else (gh - 1 if row >= gh else row)
            g[channel, row, col] = 1.0

    for x, y, w, h in engine.platforms:
        fill_rect(0, x, y, w, h)
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
            col, row = _cell(cfg, c.x, c.y - cfg.player_size / 2)
        else:
            channel = 4
            col, row = _cell(cfg, c.x, cfg.height - 5)
        put_cell(channel, col, row)
    for coin in engine.coins:
        col, row = _cell(cfg, coin.x, coin.y)
        put_cell(3, col, row)
    # Canais extras do nível (hazards, botões e super moeda) logo após os base.
    base_channels = 6 if team_ids is not None else 5

    for offset, name in enumerate(get_level_channel_names(cfg.level)):
        channel = base_channels + offset
        if channel >= channels:
            continue
        if name in ("sand", "ice"):
            for hazard in getattr(engine, "hazards", ()):
                if hazard.kind == name:
                    fill_rect(channel, hazard.x, hazard.y,
                              hazard.w, hazard.h)
        elif name in ("activate_button", "cancel_button"):
            kind = "activate" if name == "activate_button" else "cancel"
            for button in getattr(engine, "buttons", ()):
                if button.kind == kind:
                    fill_rect(channel, button.x, button.y,
                              button.w, button.h)
        elif name == "super_coin":
            coin = getattr(engine, "super_coin", None)
            if coin is not None:
                fill_rect(channel, coin.x - 8, coin.y - 8, 16, 16)
        elif name == "hole":
            # Nível 5: buracos do chão procedural (regiões sem colisão).
            for hole_x, hole_y, hole_w, hole_h in getattr(
                    engine, "holes", ()):
                fill_rect(channel, hole_x, hole_y, hole_w, hole_h)
    return g

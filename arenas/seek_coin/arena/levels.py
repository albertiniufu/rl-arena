"""Definições dos níveis do seek_coin.

O nível 0 é o mapa histórico (mapa fixo). O nível 1 acrescenta duas
plataformas móveis: a mais alta (topo esquerdo), que sobe e desce, e uma
central, que se desloca horizontalmente. Há também duas moedas fixas no alto,
alcançáveis apenas quando a plataforma mais alta está no ponto máximo
(y=145); na posição baixa (y=185) o pulo não chega. O nível 2 usa o mesmo
mapa do nível 1 e acrescenta hazards temporários: areia (reduz a velocidade)
e gelo (escorrega ~1/5 da largura da tela numa direção aleatória).

O nível 3 usa o mapa do nível 1 com dois botões de contato e uma super moeda:
pisar no botão de ativação (chão da esquerda) liga a super moeda no chão da
direita por `super_coin_seconds`; o botão de cancelamento fica na plataforma
central e desliga a super moeda. Qualquer personagem pode ativar, cancelar ou
coletar; a coleta paga `super_coin_reward` e adiciona `super_coin_coins`.

O nível 4 usa o **mesmo mundo do nível 3**, mas com **observação local**: cada
agente recebe apenas um recorte `(view_size, view_size)` da grade semântica,
centrado em si (ver `observations.render_grid`). O mapa inteiro continua
existindo no engine; a câmera é que acompanha o agente. Fora dos limites do
mapa o recorte é preenchido com zeros (fundo). A visão local é ativada por
`uses_local_view(level)` e não muda o espaço de ações.

O nível 5 é um **mundo procedural infinito**: o mapa não é fixo, e sim gerado
em chunks determinísticos por seed à medida que o agente avança; o mundo rola
continuamente (as entidades andam para a esquerda quando o agente passa de
~55% da largura) e o que fica para trás é descartado. Usa a visão local do
nível 4 para manter o shape fixo, canais extras de areia/gelo/buraco e
plataformas móveis ocasionais. É single-agent por enquanto (geração infinita
multiagente ficou fora do escopo inicial).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MovingPlatform:
    """Plataforma móvel; a posição é calculada a partir de `sim_time`."""

    x: float
    y: float
    w: int
    h: int
    axis: str = "x"  # "x" ou "y"
    amplitude: float = 0.0
    period: float = 6.0
    phase: float = 0.0

    def at(self, t: float):
        offset = self.amplitude * (
            0.5 - 0.5 * math.cos(2.0 * math.pi * t / self.period + self.phase))
        if self.axis == "x":
            return (self.x + offset, self.y, self.w, self.h)
        return (self.x, self.y + offset, self.w, self.h)


@dataclass(frozen=True)
class ButtonSpec:
    """Botão de contato do nível 3 (sem ação extra: pisar já ativa)."""

    kind: str  # "activate" ou "cancel"
    x: float
    y: float
    w: float
    h: float


STATIC_PLATFORMS = {
    0: (
        (0, 360, 220, 20), (280, 360, 360, 20),
        (70, 285, 160, 14), (350, 285, 150, 14),
        (210, 215, 180, 14), (35, 145, 130, 14),
        (445, 155, 150, 14),
    ),
    1: (
        (0, 360, 220, 20), (280, 360, 360, 20),
        (70, 285, 160, 14), (350, 285, 150, 14),
        (210, 215, 180, 14),
        (445, 155, 150, 14),
    ),
}

# Moedas fixas (x, y) no alto do nível 1. Com a plataforma no topo (y=145),
# a subida até y=60 é ~85 px (dentro do pulo ~105 px); na posição baixa
# (y=185) a subida é ~125 px (fora do alcance).
FIXED_COIN_POSITIONS = {
    0: (),
    1: ((75.0, 60.0), (135.0, 60.0)),
}

# Moedas presas a uma plataforma móvel: (índice em MOVING_PLATFORMS, dx, dy)
# em relação ao centro/topo da plataforma. A moeda acompanha o movimento.
# No nível 1, o índice 1 é a plataforma mais alta.
MOVING_COIN_SPECS = {
    0: (),
    1: (
        (1, -40.0, -30.0),
        (1, 0.0, -30.0),
        (1, 40.0, -30.0),
    ),
}

MOVING_PLATFORMS = {
    0: (),
    # Entre as plataformas de y=215 e y=285; passa pelo corredor central.
    1: (
        # Plataforma central, desloca-se na horizontal.
        MovingPlatform(
            x=280, y=250, w=100, h=14,
            axis="x", amplitude=140.0, period=6.0,
        ),
        # Plataforma mais alta (topo esquerdo), sobe e desce.
        MovingPlatform(
            x=35, y=145, w=130, h=14,
            axis="y", amplitude=40.0, period=5.0,
        ),
    ),
}

# Nível 2: mesmo mapa/moedas do nível 1, com hazards temporários.
STATIC_PLATFORMS[2] = STATIC_PLATFORMS[1]
FIXED_COIN_POSITIONS[2] = FIXED_COIN_POSITIONS[1]
MOVING_COIN_SPECS[2] = MOVING_COIN_SPECS[1]
MOVING_PLATFORMS[2] = MOVING_PLATFORMS[1]

HAZARD_KINDS = {
    0: (),
    1: (),
    2: ("sand", "ice"),
    3: (),
    4: (),
    5: ("sand", "ice"),
}

# Nível 3: mesmo mapa do nível 1, com botões e uma super moeda.
STATIC_PLATFORMS[3] = STATIC_PLATFORMS[1]
FIXED_COIN_POSITIONS[3] = FIXED_COIN_POSITIONS[1]
MOVING_COIN_SPECS[3] = MOVING_COIN_SPECS[1]
MOVING_PLATFORMS[3] = MOVING_PLATFORMS[1]

# Ativação no chão da esquerda; cancelamento na plataforma central.
BUTTONS = {
    0: (),
    1: (),
    2: (),
    3: (
        ButtonSpec("activate", 25.0, 348.0, 55.0, 12.0),
        ButtonSpec("cancel", 270.0, 203.0, 60.0, 12.0),
    ),
}

# A super moeda nasce no chão da direita; o agente precisa atravessar o mapa.
SUPER_COIN_POSITIONS = {
    0: None,
    1: None,
    2: None,
    3: (610.0, 342.0),
}

# Canais extras de observação, na ordem em que entram na grade.
LEVEL_CHANNEL_NAMES = {
    0: (),
    1: (),
    2: ("sand", "ice"),
    3: ("activate_button", "cancel_button", "super_coin"),
    5: ("sand", "ice", "hole"),
}

# Nível 4: mesmo mundo do nível 3 (botões/super moeda) com visão local.
STATIC_PLATFORMS[4] = STATIC_PLATFORMS[3]
FIXED_COIN_POSITIONS[4] = FIXED_COIN_POSITIONS[3]
MOVING_COIN_SPECS[4] = MOVING_COIN_SPECS[3]
MOVING_PLATFORMS[4] = MOVING_PLATFORMS[3]
BUTTONS[4] = BUTTONS[3]
SUPER_COIN_POSITIONS[4] = SUPER_COIN_POSITIONS[3]
LEVEL_CHANNEL_NAMES[4] = LEVEL_CHANNEL_NAMES[3]

# Nível 5: mundo procedural; o mapa fixo é vazio e as entidades nascem em
# `endless.py` (a visão local mantém o shape do nível 4).
STATIC_PLATFORMS[5] = ()
FIXED_COIN_POSITIONS[5] = ()
MOVING_COIN_SPECS[5] = ()
MOVING_PLATFORMS[5] = ()
BUTTONS[5] = ()
SUPER_COIN_POSITIONS[5] = None

# Níveis cuja observação é um recorte egocêntrico (visão parcial). Os demais
# continuam entregando o mapa inteiro — mudar isso quebraria checkpoints e a
# comparação entre níveis.
LOCAL_VIEW_LEVELS = frozenset({4, 5})

# Níveis com mundo gerado em chunks (sem mapa fixo).
ENDLESS_LEVELS = frozenset({5})


def uses_local_view(level: int) -> bool:
    """O nível entrega apenas o recorte ao redor do agente?"""
    return int(level) in LOCAL_VIEW_LEVELS


def uses_endless_world(level: int) -> bool:
    """O nível gera o mundo em chunks (mundo procedural infinito)?"""
    return int(level) in ENDLESS_LEVELS


def get_level(level: int):
    """Devolve `(plataformas_estáticas, plataformas_móveis)` do nível."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return STATIC_PLATFORMS[level], MOVING_PLATFORMS[level]


def get_fixed_coins(level: int):
    """Moedas fixas do nível (posições no mapa, não presas a plataformas)."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return FIXED_COIN_POSITIONS[level]


def get_moving_coins(level: int):
    """Moedas presas a plataformas móveis: `(moving_index, dx, dy)`."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return MOVING_COIN_SPECS[level]


def get_hazard_kinds(level: int):
    """Tipos de hazard que aparecem no nível (vazio nos níveis 0/1/3/4)."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return HAZARD_KINDS[level]


def get_buttons(level: int):
    """Botões de contato do nível (vazio fora do nível 3)."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return BUTTONS[level]


def get_super_coin_position(level: int):
    """Posição onde a super moeda nasce, ou `None` se o nível não tem uma."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return SUPER_COIN_POSITIONS[level]


def get_level_channel_names(level: int):
    """Nomes dos canais extras de observação do nível."""
    level = int(level)
    if level not in STATIC_PLATFORMS:
        raise ValueError(
            f"nível desconhecido: {level} "
            f"(disponíveis: {sorted(STATIC_PLATFORMS)})")
    return LEVEL_CHANNEL_NAMES[level]

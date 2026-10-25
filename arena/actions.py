"""Espaço de ações reduzido da arena (6 ações).

O agente tem uma única intensidade de pulo (a máxima) e uma única
intensidade de empurrão (a máxima). O empurrão mantém as duas direções e o
pulo é sempre vertical; o deslocamento horizontal continua disponível com
`LEFT`/`RIGHT`, inclusive no ar.

Os códigos são compartilhados por engine, environment, políticas e testes —
nenhum módulo deve usar números soltos para se referir a uma ação.
"""

STOP = 0
LEFT = 1
RIGHT = 2
JUMP = 3
PUSH_LEFT = 4
PUSH_RIGHT = 5

N_ACTIONS = 6

# Intensidade máxima do espaço antigo (ações 8 e 11 do espaço de 12).
PUSH_INTENSITY = 3

# Ações sem empurrão, usadas pelo adversário aleatório histórico.
NON_PUSH_ACTIONS = (STOP, LEFT, RIGHT, JUMP)

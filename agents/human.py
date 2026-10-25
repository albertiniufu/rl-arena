"""Política controlada por humano via teclado.

Usada pelo `play.py --player human`: o jogador (personagem 0) é controlado
pelo teclado enquanto o adversário usa a política de `--opponent`.
Requer renderização, que o play.py ativa automaticamente.
"""
import pygame

from arena.actions import JUMP, LEFT, PUSH_LEFT, PUSH_RIGHT, RIGHT, STOP


class HumanPolicy:
    """Mapeia o teclado para o espaço de ações de 6 entradas.

    Setas/WASD: andar | Espaço/↑/W: pular | Q: empurrar ← | E: empurrar →
    O pulo é único (vertical). Segurar uma direção no ar continua movendo o
    personagem; por isso, com o pulo segurado no ar, a política devolve a
    direção em vez de repetir o pulo.
    """
    def act(self, ctx):
        # Sem sistema de vídeo (janela) não há teclado para ler: fica parado.
        if not pygame.display.get_init():
            return STOP
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        jump = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]
        if keys[pygame.K_q]:
            return PUSH_LEFT
        if keys[pygame.K_e]:
            return PUSH_RIGHT
        character = getattr(ctx, "self_character", None)
        on_ground = character is None or character.on_ground
        if jump and on_ground:
            return JUMP
        if left and not right:
            return LEFT
        if right and not left:
            return RIGHT
        return STOP

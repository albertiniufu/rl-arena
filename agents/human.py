"""Política controlada por humano via teclado.

Usada pelo `play.py --player human`: o jogador (personagem 0) é controlado
pelo teclado enquanto o adversário usa a política de `--opponent`.
Requer renderização, que o play.py ativa automaticamente.
"""
import pygame


class HumanPolicy:
    """Mapeia o teclado para o espaço de ações de 12 entradas.

    Setas/WASD: andar | Espaço/↑/W: pular (diagonal se estiver andando)
    Q: empurrar para a esquerda (força máxima) | E: empurrar para a direita
    """
    def act(self, ctx):
        # Sem sistema de vídeo (janela) não há teclado para ler: fica parado.
        if not pygame.display.get_init():
            return 0
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        jump = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]
        if keys[pygame.K_q]:
            return 8
        if keys[pygame.K_e]:
            return 11
        if jump:
            if left and not right:
                return 4
            if right and not left:
                return 5
            return 3
        if left and not right:
            return 1
        if right and not left:
            return 2
        return 0

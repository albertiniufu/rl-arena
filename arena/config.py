from dataclasses import dataclass

@dataclass
class ArenaConfig:
    width: int = 640
    height: int = 400
    cell: int = 20
    fps: int = 30
    gravity: float = 900.0 # na terra, lua ou marte?
    move_accel: float = 700.0
    max_speed: float = 150.0
    # O pulo precisa vencer o maior degrau entre plataformas adjacentes
    # (75px: chão 360 → plataforma 285). Altura discreta ≈ v²/2g − v·dt/2;
    # com 450, o pico é ~105px — margem confortável para todos os degraus.
    jump_speed: float = 450.0
    push_impulse: float = 130.0
    player_size: int = 25
    episode_seconds: float = 45.0
    respawn_seconds: float = 5.0
    coin_lifetime: float = 60.0
    coin_reward: float = 1.0
    elimination_reward: float = 3.0 # exterminador do futuro?
    death_penalty: float = -2.0
    opponent_count: int = 2 # adicionar mais bots?
    coin_count: int = 10
    frame_stack: int = 1 # quantos frames compõem a observação
    frame_skip: int = 1 # quantos substeps de física cada step do agente executa
    map_version: str = "arena-v1"

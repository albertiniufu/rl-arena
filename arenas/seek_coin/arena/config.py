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
    death_penalty: float = -2.0
    stop_penalty: float = -0.1 # escolher STOP (ficar parado) a cada step do agente
    push_miss_penalty: float = -0.1 # empurrar sem oponente em contato no início do step
    push_coin_transfer: int = 5 # moedas que passam do empurrado para quem empurra (0 desliga)
    push_coin_reward: float = 1.0 # peso de recompensa por moeda transferida (a mecânica não muda)
    pbrs_coef: float = 0.0 # PBRS sobre a diferença de moedas (0 desliga)
    pbrs_gamma: float = 0.99 # desconto do PBRS; deve casar com o gamma do agente
    opponent_count: int = 1 # adicionar mais bots?
    n_agents: int = 0 # 0 = legado (1 + opponent_count); >= 2 = ambiente multiagente
    team_ids: tuple = () # id do time por agente (multiagente); () = legado
    coin_count: int = 10
    frame_stack: int = 1 # quantos frames compõem a observação
    frame_skip: int = 1 # quantos substeps de física cada step do agente executa
    map_version: str = "arena-v1"
    level: int = 0  # 0 = mapa histórico; 1 = plataforma móvel

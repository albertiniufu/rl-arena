from dataclasses import dataclass

COLLECTOR_ROLE = "collector"
PUSHER_ROLE = "pusher"
ROLES = (COLLECTOR_ROLE, PUSHER_ROLE)

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
    # Teto de velocidade horizontal de quem é empurrado. 0 mantém o limite
    # legado (max_speed * 2); o cenário coletores × empurradores eleva o teto
    # para o impulso forte não ser cortado no substep seguinte.
    push_speed_cap: float = 0.0
    # Tempo mínimo entre dois empurrões bem-sucedidos do mesmo personagem.
    # 0 preserva o comportamento legado (sem recarga).
    push_cooldown: float = 0.0
    player_size: int = 25
    episode_seconds: float = 90.0
    respawn_seconds: float = 5.0
    coin_lifetime: float = 60.0
    coin_reward: float = 1.0
    # Nível 3: botão liga uma super moeda por uma janela curta; ela some se
    # o adversário cancelar, se ninguém coletar a tempo ou se alguém coletar.
    super_coin_seconds: float = 12.0
    super_coin_cooldown: float = 5.0
    super_coin_reward: float = 25.0
    super_coin_coins: int = 10
    # Liga botões + super moeda (níveis 3/4). None = padrão do nível:
    # ligado no nível 3 e desligado no nível 4 (visão local sem o objetivo
    # distante). `--super-coin`/`--no-super-coin` sobrescreve.
    super_coin_enabled: bool | None = None
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
    level: int = 0  # 0 = histórico; 1 = plataformas móveis; 2 = areia/gelo; 3 = botões/super moeda
    agent_roles: tuple = ()  # papéis por agente (HAPPO); () = simétrico
    # Nível 4: visão local. A observação é um recorte egocêntrico de
    # `view_size x view_size` células centrado no agente (ímpar; 9 = ~180 px).
    # Ignorado nos níveis 0..3, que continuam entregando o mapa inteiro.
    view_size: int = 9

    # Hazards temporários do nível 2 (ignorados nos níveis 0/1)
    hazard_count: int = 2  # por tipo (areia e gelo)
    hazard_width_min: float = 50.0
    hazard_width_max: float = 110.0
    hazard_height: float = 6.0
    hazard_min_seconds: float = 3.0
    hazard_max_seconds: float = 6.0
    sand_speed_factor: float = 0.3
    sand_duration: float = 5.0
    ice_slide_fraction: float = 0.2  # 1/5 da largura da tela
    ice_slide_seconds: float = 0.6
    ice_cooldown: float = 1.0

    def push_impulse_for_range(self, range_px, intensity):
        """Impulso que desloca um alvo ~`range_px` após um empurrão.

        O engine aplica o atrito **antes** de mover no substep do empurrão:
        `v0` vira `v0 * 0.88` e só então o alvo anda. A distância total é
        `v0 * dt * 0.88 / (1 - 0.88)`; invertendo, chega-se ao impulso por
        unidade de intensidade. Serve para calibrar "empurrar meia arena" sem
        números mágicos no chamador.
        """
        dt = 1.0 / float(self.fps)
        decay = 0.88 ** (dt * float(self.fps))
        speed = float(range_px) * (1.0 - decay) / (dt * decay)
        return speed / float(int(intensity))

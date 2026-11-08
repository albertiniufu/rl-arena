"""Cenários multiagente: todos-contra-todos e times.

Um `Scenario` é a fonte única de verdade sobre quantos agentes existem e a que
time cada um pertence. Ele é imutável, não depende de Gym/Torch e é gravado no
`ArenaConfig` do checkpoint (`n_agents` + `team_ids`), de modo que
`evaluate.py`/`play.py` reconstroem o mesmo cenário sem argumentos extras.

Em todos-contra-todos cada agente é o seu próprio time (`team_ids` distintos).
Em times os ids são contíguos e iguais dentro do time (`(0, 0, 1, 1)` para
2 times de 2). Os times também definem quem pode ser empurrado: aliados nunca
são alvo.
"""
from dataclasses import dataclass

from .config import COLLECTOR_ROLE, PUSHER_ROLE, ROLES


@dataclass(frozen=True)
class Scenario:
    n_agents: int
    team_ids: tuple
    # Papel opcional por agente. Vazio = cenário simétrico (todos fazem tudo).
    # O cenário coletores × empurradores preenche metade dos times com
    # `collector` e a outra metade com `pusher`.
    roles: tuple = ()

    def __post_init__(self):
        if self.n_agents < 2:
            raise ValueError(f"n_agents deve ser >= 2 (recebido {self.n_agents})")
        if len(self.team_ids) != self.n_agents:
            raise ValueError(
                f"team_ids deve ter {self.n_agents} entradas "
                f"(recebido {len(self.team_ids)})")
        if any(not isinstance(t, int) or t < 0 for t in self.team_ids):
            raise ValueError(f"team_ids inválidos: {self.team_ids!r}")
        if self.roles:
            if len(self.roles) != self.n_agents:
                raise ValueError(
                    f"roles deve ter {self.n_agents} entradas "
                    f"(recebido {len(self.roles)})")
            invalid = [r for r in self.roles if r not in ROLES]
            if invalid:
                raise ValueError(f"papéis inválidos: {invalid!r}")

    @classmethod
    def all_vs_all(cls, n_agents):
        """Cada agente é o seu próprio time."""
        n = int(n_agents)
        return cls(n, tuple(range(n)))

    @classmethod
    def teams(cls, n_agents, n_teams):
        """Times contíguos e de mesmo tamanho (ex.: 4 agentes, 2 times -> 2x2)."""
        n, k = int(n_agents), int(n_teams)
        if k < 2:
            raise ValueError(f"n_teams deve ser >= 2 (recebido {k})")
        if n % k:
            raise ValueError(
                f"n_agents ({n}) deve ser divisível por n_teams ({k})")
        size = n // k
        return cls(n, tuple(i // size for i in range(n)))

    @classmethod
    def collectors_vs_pushers(cls, n_agents, n_teams):
        """Times assimétricos: metade só coleta, metade só empurra.

        Os times são os mesmos de `teams` (contíguos e iguais); a diferença é
        o papel atribuído a cada um. `n_teams` precisa ser par para que a
        metade "coletores" e a metade "empurradores" tenham o mesmo tamanho.
        """
        n, k = int(n_agents), int(n_teams)
        if k % 2:
            raise ValueError(
                f"n_teams deve ser par em coletores × empurradores "
                f"(recebido {k})")
        base = cls.teams(n, k)
        half = k // 2
        roles = tuple(
            COLLECTOR_ROLE if team < half else PUSHER_ROLE
            for team in base.team_ids)
        return cls(n, base.team_ids, roles)

    @classmethod
    def from_config(cls, cfg):
        """Reconstrói o cenário gravado em `ArenaConfig`."""
        n = int(getattr(cfg, "n_agents", 0) or 0)
        ids = tuple(getattr(cfg, "team_ids", ()) or ())
        roles = tuple(getattr(cfg, "agent_roles", ()) or ())
        if n <= 1:
            raise ValueError(
                "ArenaConfig sem cenário multiagente (n_agents deve ser >= 2)")
        if not ids:
            ids = tuple(range(n))
        return cls(n, ids, roles)

    @property
    def n_teams(self):
        return len(set(self.team_ids))

    def team_of(self, index):
        return self.team_ids[index]

    def teammates(self, index):
        """Aliados vivos do agente (índices), excluindo ele próprio."""
        team = self.team_ids[index]
        return [i for i in range(self.n_agents)
                if i != index and self.team_ids[i] == team]

    def enemies(self, index):
        """Agentes de outros times (índices)."""
        team = self.team_ids[index]
        return [i for i in range(self.n_agents) if self.team_ids[i] != team]

    @property
    def has_roles(self):
        return bool(self.roles)

    def role_of(self, index):
        """Papel do agente ou `None` no cenário simétrico."""
        return self.roles[index] if self.roles else None

    def can_push(self, index):
        """Coletores não podem empurrar; sem papéis todos podem."""
        return self.role_of(index) != COLLECTOR_ROLE

    def can_collect(self, index):
        """Empurradores não coletam moedas do mapa."""
        return self.role_of(index) != PUSHER_ROLE

    def collectors(self):
        return [i for i in range(self.n_agents) if self.role_of(i) == COLLECTOR_ROLE]

    def pushers(self):
        return [i for i in range(self.n_agents) if self.role_of(i) == PUSHER_ROLE]

    def spawn_positions(self, cfg):
        """Posições iniciais no chão, com cada time na sua faixa horizontal.

        Divide a largura útil em `n_teams` faixas e distribui os membros de cada
        time dentro da faixa. Em todos-contra-todos há uma faixa por agente, o
        que também evita sobreposição inicial.
        """
        margin = float(cfg.player_size)
        usable = max(1.0, cfg.width - 2.0 * margin)
        groups = {}
        for i, team in enumerate(self.team_ids):
            groups.setdefault(team, []).append(i)
        ordered = sorted(groups)
        band = usable / len(ordered)
        y = float(cfg.height - 80.0)  # mesma altura do spawn legado (320 em 400)
        positions = [None] * self.n_agents
        for k, team in enumerate(ordered):
            members = groups[team]
            for j, index in enumerate(members):
                x = margin + k * band + (j + 1) * band / (len(members) + 1)
                positions[index] = (float(x), y)
        return positions

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


@dataclass(frozen=True)
class Scenario:
    n_agents: int
    team_ids: tuple

    def __post_init__(self):
        if self.n_agents < 2:
            raise ValueError(f"n_agents deve ser >= 2 (recebido {self.n_agents})")
        if len(self.team_ids) != self.n_agents:
            raise ValueError(
                f"team_ids deve ter {self.n_agents} entradas "
                f"(recebido {len(self.team_ids)})")
        if any(not isinstance(t, int) or t < 0 for t in self.team_ids):
            raise ValueError(f"team_ids inválidos: {self.team_ids!r}")

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
    def from_config(cls, cfg):
        """Reconstrói o cenário gravado em `ArenaConfig`."""
        n = int(getattr(cfg, "n_agents", 0) or 0)
        ids = tuple(getattr(cfg, "team_ids", ()) or ())
        if n <= 1:
            raise ValueError(
                "ArenaConfig sem cenário multiagente (n_agents deve ser >= 2)")
        if not ids:
            ids = tuple(range(n))
        return cls(n, ids)

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

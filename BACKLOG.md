# Backlog

## Níveis do cenário

- [ ] **Sistema de níveis** — transformar o mapa fixo em níveis selecionáveis. O nível 0 é o mapa atual (`arena-v1`, plataformas fixas em `arena/engine.py`); o nível 1 introduz uma **plataforma flutuante móvel**.
  - Extrair a lista `self.platforms` (hoje fixa em `ArenaEngine.__init__`) para definições de nível, ex.: `arena/levels.py` com `LEVELS = {0: ..., 1: ...}`.
  - Adicionar `level: int = 0` em `ArenaConfig` (ou evoluir `map_version`, que hoje só existe como metadado) e expor `--level` em `train.py`, `evaluate.py` e `play.py`.
  - Gravar o nível no checkpoint e recusar retomada com nível diferente, como já é feito com `n_agents`/`team_ids`.
  - **Nível 1 — plataforma flutuante móvel**: plataforma que se desloca (horizontal e/ou vertical) por uma trajetória; atualizar a posição a cada substep de física, antes da colisão.
  - Física: personagem em cima da plataforma deve ser carregado junto (herdar o deslocamento horizontal) e continuar colidindo/aterrissando normalmente; moedas não devem nascer em posição inválida sobre a plataforma móvel.
  - Observação/render: o canal 0 de `render_grid` e todas as skins (`arena/renderer*.py`) já leem `engine.platforms`, mas hoje como tuplas estáticas — garantir que leiam a posição atualizada por substep.
  - Testes: mapa do nível 0 idêntico ao atual (regressão); nível 1 com plataforma móvel (movimento, colisão, caronagem e reprodução com seed).
  - Documentar os níveis e o `--level` no `CLAUDE.md` e no `README.md`.

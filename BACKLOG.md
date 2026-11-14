# Backlog

## Níveis do cenário

> Implementado: `arenas/seek_coin/arena/levels.py` (nível 0 = mapa histórico; nível 1 = plataforma móvel), `--level` nos entry points e nível gravado/validado no checkpoint; o nível 1 tem duas plataformas móveis (a mais alta, vertical, e a central, horizontal).

- [x] **Sistema de níveis** — transformar o mapa fixo em níveis selecionáveis. O nível 0 é o mapa atual (`arena-v1`, plataformas fixas em `arenas/seek_coin/arena/engine.py`); o nível 1 introduz uma **plataforma flutuante móvel**.
  - Extrair a lista `self.platforms` (hoje fixa em `ArenaEngine.__init__`) para definições de nível, ex.: `arenas/seek_coin/arena/levels.py` com `LEVELS = {0: ..., 1: ...}`.
  - Adicionar `level: int = 0` em `ArenaConfig` (ou evoluir `map_version`, que hoje só existe como metadado) e expor `--level` em `arenas/seek_coin/train.py`, `evaluate.py` e `play.py`.
  - Gravar o nível no checkpoint e recusar retomada com nível diferente, como já é feito com `n_agents`/`team_ids`.
  - **Nível 1 — plataforma flutuante móvel**: plataforma que se desloca (horizontal e/ou vertical) por uma trajetória; atualizar a posição a cada substep de física, antes da colisão.
  - Física: personagem em cima da plataforma deve ser carregado junto (herdar o deslocamento horizontal) e continuar colidindo/aterrissando normalmente; moedas não devem nascer em posição inválida sobre a plataforma móvel.
  - Observação/render: o canal 0 de `render_grid` e todas as skins (`arena/renderer*.py`) já leem `engine.platforms`, mas hoje como tuplas estáticas — garantir que leiam a posição atualizada por substep.
  - Testes: mapa do nível 0 idêntico ao atual (regressão); nível 1 com plataforma móvel (movimento, colisão, caronagem e reprodução com seed).
  - Documentar os níveis e o `--level` no `CLAUDE.md` e no `README.md`.

## The Option Keyboard

- [ ] **Implementar ambiente e algoritmo do paper "The Option Keyboard: Combining Skills in Reinforcement Learning"** (Barreto, Borsa, Quan, Schaul, Silver, Hessel, Mankowitz, Zidek, Munos — NeurIPS 2019).
  - Ideia central: aprender *successor features* `ψ(s, a)` e um conjunto de opções para vetores de pesos de cumulantes `w`; a política combinada usa GPI sobre `Q_w(s, a) = ψ(s, a) · w`; o "teclado de opções" combina convexamente as opções aprendidas para generalizar a tarefas novas sem treinar do zero.
  - Ambiente: usar `arenas/seek_coin` com cumulantes configuráveis (moedas, empurrões, sobrevivência) e `w` amostrado por episódio; se o acoplamento com o Arena ficar artificial, criar `arenas/option_keyboard/` com um gridworld dedicado e recompensa `r = φ(s, a) · w`.
  - Algoritmo: `arenas/algs/alg_option_keyboard/` seguindo o contrato de `arenas/algs/alg_dqn/agent.py` — aprender `ψ` e as opções-base (ex.: vértices/cantos do simplex de `w`), calcular o Q combinado por GPI no momento de agir e escolher/combinear opções para um `w` alvo.
  - Avaliação: tarefas com `w` retidos (fora do conjunto de treino), comparando com treinar cada `w` do zero; medir a generalização do teclado.
  - Checkpoint: gravar `algorithm`, `arch_version`, conjunto de opções, vetores `w` de treino e `obs_shape`; retomada com o mesmo conjunto.
  - Testes: reprodutibilidade por seed, GPI ≥ cada opção individual no conjunto de treino, formato de observação/ação e retomada.
  - CLI: `--alg option_keyboard`, opções para pesos/cumulantes de treino e `--eval-weights` para as tarefas retidas.


## Nível 2 — areia e gelo

> Implementado: `--level 2`, hazards de areia/gelo com spawn aleatório, canais de observação extras e renderer default. Testes e smoke PPO/IPPO ok; treino noturno em `logs/overnight_l2`.

- [x] **Nível 2 com hazards temporários** — áreas de areia e gelo aparecem em posições aleatórias sobre as plataformas por alguns segundos e desaparecem/renascem.
  - Areia: personagem que passa por cima tem a aceleração/velocidade máxima reduzida (ex.: 30%).
  - Gelo: ao pisar, escorrega na horizontal numa direção aleatória por ~1/5 da largura da tela (128 px), com cooldown para não re-disparar.
  - Engine: `Hazard` (tipo, retângulo, timer, plataforma), spawn/respawn com RNG, acompanhar plataformas móveis e serializar em `get_state`/`set_state`.
  - Observação: canais extras só no nível 2 (areia/gelo) para não mudar o shape dos níveis 0/1; renderer default desenha os hazards.
  - Testes: spawn/timers, efeito da areia, distância do escorregão, canais de observação, shapes e checkpoint com `--level 2`.
  - Documentar no README/CLAUDE.

# Documentação do RL Arena

Este diretório é a porta de entrada para **pessoas**. Os READMEs dentro de
cada arena explicam o domínio; aqui explicamos como o monorepo se organiza e
por onde começar a ler.

## Trilha de leitura sugerida

1. **[README da raiz](../README.md)** — o que é o projeto e como rodar.
2. **[Arquitetura](arquitetura.md)** — as camadas, o caminho de uma chamada
   `train.py` e o formato do checkpoint.
3. **[Algoritmos](algoritmos.md)** — catálogo, contrato dos `alg_*` e como
   escolher um algoritmo.
4. **README da arena que você quer estudar**:
   - [`seek_coin`](../arenas/seek_coin/README.md) — plataforma 2D, o ambiente
     mais rico (multiagente, vetorização, níveis, skins).
   - [`risk`](../arenas/risk/README.md) — regras de tabuleiro e combate.
   - [`economy`](../arenas/economy/README.md) — simulação macro e banco.
5. **Aula HTML do algoritmo** (se quiser a matemática):
   [`arenas/seek_coin/docs/index.html`](../arenas/seek_coin/docs/index.html) —
   o [README da raiz](../README.md) também tem um índice visual com as 12
   aulas.
6. **[Como contribuir](como-contribuir.md)** — só quando for mexer no código.
7. **[Experimentos](experimentos/README.md)** — como registrar a evolução dos
   setups e comparar algoritmos de forma justa. Os princípios estão em
   **[Desenho empírico](experimentos/empirical-design.md)**.
8. **[Fila de Experimentos](fila-de-experimentos.md)** — como submeter, acompanhar e
   cancelar jobs de treino/BO pela interface web.

## Do que é feito cada tipo de arquivo

| Tipo | Onde fica | Para que serve |
| --- | --- | --- |
| Ambiente (Gymnasium) | `arenas/<arena>/env*.py` | `reset`/`step`, observação, recompensa. |
| Regras/simulação pura | `arenas/<arena>/*.py` | Motor sem dependência de RL. |
| Entry points da arena | `arenas/<arena>/{train,evaluate,play}.py` | Argumentos e composição do agente. |
| CLI unificado | `train.py`, `evaluate.py`, `play.py` + `arenas/cli.py` | Despacho por `--arena`. |
| Infraestrutura comum | `arenas/common/` | Loop de treino, avaliação, checkpoint. |
| Algoritmos | `arenas/algs/alg_<nome>/` | Modelo, buffer e agente. |
| Catálogo de arenas | `arenas/registry.py` | Metadados e entry points de cada arena. |
| Testes | `arenas/<arena>/tests/`, `arenas/common/tests/` | Contrato e regressões. |
| Artefatos locais | `checkpoints/`, `logs/`, `experiments/`, `videos/` | Gerados; não vão para o git. |

## Convenções rápidas

- Documentação, comentários e mensagens de erro em **português**.
- Observação vetorial `(N,)` usa MLP; imagem `(C,H,W)` usa CNN.
- Ações discretas: `dqn`, `doubledqn_per_dueling`, `mka`, `ppo`, `eppo`,
  `ppo_aes`, `gtr`, `a2c`, `ippo`, `mappo`, `happo`.
- Ações contínuas: `ppo_continuous`.
- Seed sempre reprodutível; RNG injetável nos ambientes.
- Checkpoint = pesos + metadados + config da arena (sem buffer nem RNG).

## Glossário mínimo

| Termo | Significado no repositório |
| --- | --- |
| **arena** | Um ambiente de RL (seek_coin, risk, economy). |
| **algoritmo** | Pacote `arenas/algs/alg_*` selecionado por `--alg`. |
| **entry point** | `main()` da arena chamado pelo CLI da raiz. |
| **checkpoint** | Arquivo `.pt` com pesos e a configuração necessária para reabrir o ambiente. |
| **rollout** | Sequência de transições coletadas antes de um update (on-policy). |
| **replay** | Buffer de transições antigas reamostradas (off-policy). |
| **GH** | Grupo homogêneo de risco, unidade de decisão do banco (economy). |
| **CTDE** | *Centralized training, decentralized execution* (MAPPO). |

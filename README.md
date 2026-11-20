# RL Arena

Monorepo didático de Reinforcement Learning: **três ambientes**, **dezesseis
algoritmos** e uma infraestrutura compartilhada de treino, avaliação e
checkpoints. A ideia é servir tanto para estudar RL na prática quanto para
comparar algoritmos no mesmo protocolo.

> **Novo por aqui?** Siga a trilha de leitura em [`docs/README.md`](docs/README.md).
> Ela diz em que ordem ler cada parte sem se perder.

## As arenas

| Arena | O que é | Ações | Algoritmo padrão |
| --- | --- | --- | --- |
| [`seek_coin`](arenas/seek_coin/README.md) | Plataforma 2D: coletar moedas e empurrar o adversário. Níveis com plataformas móveis, areia e gelo. | `Discrete(6)` | `dqn` |
| [`risk`](arenas/risk/README.md) | Conquista de territórios inspirada em War/Risk: reforço, ataque por dados e fortificação. | `Discrete(n²+n+1)` | `dqn` |
| [`economy`](arenas/economy/README.md) | Banco-agente em um simulador macroeconômico: aprovação, limite e spread por grupo de risco. | `Box(3·n_GH)` | `ppo_continuous` |

Cada arena é um pacote autocontido em `arenas/<nome>/`, com o seu próprio
`README.md`, testes e entry points.

## Comece em 5 minutos

```bash
git clone <url-do-repo> rl-arena
cd rl-arena

python -m venv .venv            # .venv, venv ou o nome que preferir
source .venv/bin/activate
pip install -r requirements.txt

# Arena padrão = seek_coin
python train.py --arena seek_coin --alg dqn --episodes 300
python train.py --config configs/seek_coin/ppo-level3.yaml   # preset YAML
python evaluate.py --arena seek_coin --checkpoint checkpoints/best.pt
python play.py --arena seek_coin --checkpoint checkpoints/best.pt --render

pytest -q                       # suíte completa
```

Há um `Makefile` com atalhos: `make help`, `make install`, `make test`,
`make smoke` (2 episódios em cada arena) e `make docs` (serve as aulas HTML).

Os três scripts da raiz são atalhos: escolhem a arena com `--arena` e
repassam o resto dos argumentos para o entry point daquela arena. Para ver as
opções de uma arena específica:

```bash
python train.py --arena risk --help
python train.py --arena economy --help
```

## Mapa do repositório

```text
rl-arena/
├── train.py / evaluate.py / play.py   # CLI unificado (--arena <nome>)
├── conftest.py                        # pytest: raiz no sys.path
├── docs/                              # guias para humanos
├── configs/                           # presets YAML e espaços de busca
├── scripts/                           # atalhos (relatório, BO, validação)
├── arenas/
│   ├── registry.py                    # catálogo de arenas (ArenaSpec)
│   ├── cli.py                         # despacho de --arena
│   ├── common/                        # trainer, evaluator, checkpoint, fábrica
│   ├── algs/                          # algoritmos compartilhados (alg_*)
│   ├── seek_coin/                     # arena: plataforma 2D
│   ├── risk/                          # arena: territórios
│   └── economy/                       # arena: economia/banco
├── checkpoints/                       # artefatos de treino (gitignored)
├── logs/                              # CSVs e TensorBoard (gitignored)
├── experiments/                       # execuções longas (gitignored)
└── videos/                            # gravações locais (gitignored)
```

Regra de ouro: **código específico de uma arena fica dentro dela**; o que é
reutilizável vai para `arenas/common/` ou `arenas/algs/`. Os detalhes de cada
camada estão em [`docs/arquitetura.md`](docs/arquitetura.md).

## Os algoritmos

Todos vivem em [`arenas/algs/`](arenas/algs/README.md) e são selecionados por
`--alg <nome>`. O nome é sempre o sufixo do diretório `alg_<nome>`.

| `--alg` | Família | Ações | Quando usar |
| --- | --- | --- | --- |
| `dqn` | off-policy, valor | discreta | Primeira parada; simples e rápido de depurar. |
| `doubledqn_per_dueling` | off-policy, valor | discreta | Quando o DQN superestima ou o replay uniforme é pouco. |
| `mka` | off-policy, valor | discreta | Experimento de arquitetura (CNN custom) sobre o DQN. |
| `ppo` | on-policy, ator-crítico | discreta | Referência on-policy; estável e paralelizável. |
| `eppo` | on-policy, ator-crítico | discreta | Exploração guiada por incerteza (crítico evidencial). |
| `ppo_aes` | on-policy, ator-crítico | discreta | Entropia adaptativa em vez de coeficiente fixo. |
| `gtr` | on-policy, ator-crítico | discreta | Região de confiança gaussiana com âncoras. |
| `a2c` | on-policy, ator-crítico | discreta | Versão mais simples do PPO, com retornos n-step. |
| `ppo_continuous` | on-policy, ator-crítico | contínua | Arenas com `Box` (economy). |
| `ddpg` | off-policy, determinístico | contínua | Referência simples para `Box`. |
| `td3` | off-policy, determinístico | contínua | Twin critics + target smoothing. |
| `sac` | off-policy, entropia | contínua | Eficiência amostral em `Box`. |
| `sac_discrete` | off-policy, entropia | discreta | SAC para ações discretas no seek_coin. |
| `ippo` | multiagente, on-policy | discreta | Vários agentes com crítico local. |
| `mappo` | multiagente, on-policy | discreta | Vários agentes com crítico centralizado (CTDE). |
| `happo` | multiagente, on-policy | discreta | Agentes heterogêneos com atualização sequencial. |

## Aulas HTML (índice visual)

Cada algoritmo tem uma aula ilustrada com a matemática, a arquitetura, o
código e os hiperparâmetros. Abra o [índice completo](arenas/seek_coin/docs/index.html)
no navegador ou rode `make docs` e acesse
`http://localhost:8000/arenas/seek_coin/docs/index.html`.

| # | Aula | Família | O que cobre |
| ---: | --- | --- | --- |
| 1 | [DQN](arenas/seek_coin/docs/dqn-aula.html) | off-policy | Q-learning profundo, replay uniforme, alvo congelado e ε-greedy. A base de tudo. |
| 2 | [Double DQN + PER + dueling](arenas/seek_coin/docs/doubledqn-aula.html) | off-policy | Alvo menos otimista, replay priorizado e rede que separa valor de vantagem. |
| 3 | [MKA](arenas/seek_coin/docs/mka-aula.html) | off-policy | O mesmo DQN com CNN custom (GELU): experimento de arquitetura. |
| 4 | [PPO](arenas/seek_coin/docs/ppo-aula.html) | on-policy | Ator-crítico com surrogate recortado, GAE e política categórica. |
| 5 | [A2C](arenas/seek_coin/docs/a2c-aula.html) | on-policy | O degrau mais simples do actor-critic: retornos n-step e um update por rollout. |
| 6 | [EPPO](arenas/seek_coin/docs/eppo-aula.html) | on-policy | Crítico evidencial (NIG), GAE com variância e vantagem UCB. |
| 7 | [PPO-AES](arenas/seek_coin/docs/ppo_aes-aula.html) | on-policy | Entropia adaptativa agendada pelo erro de TD. |
| 8 | [GTR](arenas/seek_coin/docs/gtr-aula.html) | on-policy | Região de confiança gaussiana com mistura de âncoras. |
| 9 | [PPO contínuo](arenas/seek_coin/docs/ppo_continuous-aula.html) | on-policy | Política gaussiana para ações `Box` (a base da arena economy). |
| 10 | [IPPO](arenas/seek_coin/docs/ippo-aula.html) | multiagente | Vários agentes, política compartilhada e crítico local. |
| 11 | [MAPPO](arenas/seek_coin/docs/mappo-aula.html) | multiagente | Ator local + crítico centralizado (CTDE). |
| 12 | [HAPPO](arenas/seek_coin/docs/happo-aula.html) | multiagente | Atores heterogêneos com update sequencial e correção de razão. |

A visão geral e o contrato que todo `alg_*` segue estão em
[`docs/algoritmos.md`](docs/algoritmos.md).

## Comandos essenciais

```bash
# Seek Coin (a arena original)
python train.py --arena seek_coin --alg dqn --episodes 300
python train.py --arena seek_coin --alg ppo --envs 8 --opponent rule
python train.py --arena seek_coin --alg ippo --agents 4 --scenario teams --teams 2

# Risk (ações discretas)
python train.py --arena risk --alg dqn --episodes 300
python play.py --arena risk --checkpoint checkpoints/risk/best.pt

# Economy (ações contínuas)
python train.py --arena economy --alg ppo_continuous --episodes 200
python play.py --arena economy --checkpoint checkpoints/economy/best.pt

# Submeter um experimento à Fila de Experimentos (inicia o servidor se preciso)
python scripts/experiment.py --name ppo-30min -- \
  python train.py --config configs/seek_coin/ppo-level3.yaml \
  --set training.max_hours=0.5

# train.py direto também entra na fila e espera a vez; use --no-queue para
# rodar imediatamente sem a fila.
python train.py --no-queue --episodes 2
```

Testes por escopo:

```bash
pytest -q                     # tudo
pytest -q arenas/risk         # uma arena
pytest -q arenas/common       # infraestrutura e CLI
pytest -q arenas/seek_coin    # arena + testes de integração dos algoritmos
```

## Como o treino funciona (visão de 30 segundos)

```text
train.py --arena risk --alg dqn
        │
        ▼
arenas/cli.py  ──►  arenas/registry.py  ──►  arenas/risk/train.py
                                                │
                                                ├── RiskEnv          (arena)
                                                ├── build_agent      (common)
                                                └── train_agent      (common)
                                                        │
                                                        └── save_checkpoint (algs/alg_registry)
```

O checkpoint guarda o algoritmo, a arquitetura, o formato da observação e a
configuração da arena. Por isso `evaluate.py` e `play.py` conseguem
reconstruir o ambiente **sem** você repetir `--alg` ou os hiperparâmetros.
Detalhes em [`docs/arquitetura.md`](docs/arquitetura.md).

## Adicionar uma arena ou um algoritmo

- **Nova arena:** criar `arenas/<nome>/` com um ambiente Gymnasium, entry
  points e testes; registrar em `arenas/registry.py`.
- **Novo algoritmo:** criar `arenas/algs/alg_<nome>/agent.py` com a classe
  `Agent` no contrato comum. O `--alg <nome>` passa a funcionar sozinho.

O passo a passo, as convenções e o checklist estão em
[`docs/como-contribuir.md`](docs/como-contribuir.md).

## Documentação

- [`docs/README.md`](docs/README.md) — índice e trilha de leitura
- [`docs/arquitetura.md`](docs/arquitetura.md) — camadas, fluxo de treino e checkpoints
- [`docs/algoritmos.md`](docs/algoritmos.md) — catálogo e contrato dos `alg_*`
- [`docs/fila-de-experimentos.md`](docs/fila-de-experimentos.md) — fila de experimentos e interface web
- [`configs/README.md`](configs/README.md) — presets YAML e espaços de busca
- [`docs/como-contribuir.md`](docs/como-contribuir.md) — convenções, testes e extensão
- [`docs/experimentos/README.md`](docs/experimentos/README.md) — registro de experimentos, setups e comparação de algoritmos
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — atalho para o guia de contribuição
- [`BACKLOG.md`](BACKLOG.md) — ideias e tarefas futuras
- [`CLAUDE.md`](CLAUDE.md) — instruções para agentes de código (não é guia de uso)

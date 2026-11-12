# Arenas

Monorepo de arenas de RL e simuladores. Cada arena vive em `arenas/<nome>/`
e compartilha os algoritmos (`arenas/algs`) e a infraestrutura de treino,
avaliação e checkpoints (`arenas/common`). Os entry points da raiz escolhem a
arena com `--arena`.

| Arena | Diretório | Descrição |
| --- | --- | --- |
| Seek Coin | [`arenas/seek_coin/`](arenas/seek_coin/README.md) | Plataforma 2D: coletar moedas e empurrar o adversário. DQN, Double DQN/PER/dueling, PPO, A2C, IPPO, MAPPO e HAPPO. Nível 1 com plataforma móvel. |
| Risk | [`arenas/risk/`](arenas/risk/README.md) | Conquista de territórios inspirada em War/Risk: reforço, ataque por dados e fortificação. |
| Economy | [`arenas/economy/`](arenas/economy/README.md) | Banco-agente: aprovação, limite e spread por grupo homogêneo de risco (GH), com recompensa por RAROC líquido, sobre uma economia com população, empresas, governo e choques macro. |

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Treinar, avaliar e reproduzir

```bash
# Arena padrão = seek_coin
python train.py --arena seek_coin --alg dqn --episodes 300
python evaluate.py --arena seek_coin --checkpoint checkpoints/seek_coin/best.pt
python play.py --arena seek_coin --checkpoint checkpoints/seek_coin/best.pt --render

# Risk (ações discretas)
python train.py --arena risk --alg dqn --episodes 300
python evaluate.py --arena risk --checkpoint checkpoints/risk/best.pt
python play.py --arena risk --checkpoint checkpoints/risk/best.pt

# Economy (ações contínuas)
python train.py --arena economy --alg ppo_continuous --episodes 200
python evaluate.py --arena economy --checkpoint checkpoints/economy/best.pt
python play.py --arena economy --checkpoint checkpoints/economy/best.pt

python train.py --help        # lista as arenas
```

Sem `--arena`, a arena padrão é `seek_coin`. Cada arena tem o seu próprio
`--help` (`python train.py --arena risk --help`).

## Testes

```bash
pytest -q                     # suíte completa
pytest -q arenas/risk         # uma arena
pytest -q arenas/common       # infraestrutura e CLI
```

## Estrutura

- `arenas/algs/` — algoritmos compartilhados (`alg_dqn`,
  `alg_doubledqn_per_dueling`, `alg_ppo`, `alg_a2c`, `alg_ppo_continuous`,
  `alg_ippo`, `alg_mappo`) e `alg_registry` (descoberta, checkpoint,
  `CheckpointPolicy`). Os modelos aceitam observação em imagem `(C,H,W)` ou
  vetor `(N,)`.
- `arenas/common/` — fábrica de agentes, loop de treino single-agent,
  avaliação e carregamento de checkpoints.
- `arenas/registry.py` — metadados e entry points de cada arena.
- `arenas/cli.py` + `train.py`/`evaluate.py`/`play.py` na raiz — despacho
  unificado por `--arena`.
- `arenas/seek_coin/` — jogo original (engine, ambiente, renderers, tests,
  trainer completo com multiagente/vetorização/TensorBoard).
- `arenas/risk/` — territórios, fases, combate por dados, bot, ambiente com
  fallback para ações inválidas e modo texto.
- `arenas/economy/` — economia, banco, `BankEnv`, `EconomyEnv` (governo),
  demos e testes.

## Adicionar uma arena

1. Crie `arenas/<nome>/` com `env.py` (contrato Gymnasium), `config.py` e
   testes em `arenas/<nome>/tests/`.
2. Implemente `train.py`, `evaluate.py` e `play.py` usando
   `arenas.common.trainer`/`evaluator`/`checkpoint` (como risk e economy).
3. Registre a arena em `arenas/registry.py` (`ARENAS`).
4. Rode `pytest -q`.

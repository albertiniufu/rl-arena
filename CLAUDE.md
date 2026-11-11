# CLAUDE.md

Monorepo de arenas de RL. Cada arena tem o seu próprio `CLAUDE.md` com os
detalhes de arquitetura e comandos; o Claude Code carrega o arquivo mais
próximo ao diretório de trabalho.

## Estrutura

- `arenas/algs/` — algoritmos compartilhados: `alg_dqn`,
  `alg_doubledqn_per_dueling`, `alg_ppo`, `alg_a2c`, `alg_ppo_continuous`,
  `alg_ippo`, `alg_mappo` e `alg_registry` (descoberta, checkpoints,
  `CheckpointPolicy`). Os modelos aceitam imagem `(C,H,W)` ou vetor `(N,)`.
- `arenas/common/` — `agent_factory`, `trainer` (single-agent), `evaluator` e
  `checkpoint` (reconstrução a partir do checkpoint).
- `arenas/registry.py` — `ArenaSpec` de cada arena (entry points, algoritmo
  padrão, tipo de ação).
- `arenas/cli.py` + `train.py`/`evaluate.py`/`play.py` na raiz — CLI
  unificado com `--arena` (padrão `seek_coin`).
- `arenas/seek_coin/` — RL Arena original (plataforma 2D, `arena/`, `agents/`,
  trainer completo com multiagente/vetorização/TensorBoard). Leia
  [`arenas/seek_coin/CLAUDE.md`](arenas/seek_coin/CLAUDE.md).
- `arenas/risk/` — conquista de territórios (War/Risk) + `RiskEnv`.
- `arenas/economy/` — economia + `BankEnv` (banco-agente) e `EconomyEnv`
  (governo).

## Comandos

```bash
pytest -q                                  # suíte completa
pytest -q arenas/risk                      # uma arena
pytest -q arenas/common                    # infraestrutura e CLI

python train.py --arena seek_coin --alg dqn --episodes 300
python train.py --arena risk --alg dqn --episodes 300
python train.py --arena economy --alg ppo_continuous --episodes 200
python evaluate.py --arena risk --checkpoint checkpoints/risk/best.pt
python play.py --arena economy --checkpoint checkpoints/economy/best.pt
```

As arenas têm `--help` próprio: `python train.py --arena risk --help`.
O seek_coin também pode ser executado de dentro da pasta:
`cd arenas/seek_coin && python train.py --episodes 300`.

## Convenções

- Documentação e comentários em português.
- `arenas/algs` e `arenas/common` são compartilhados; código específico de
  uma arena fica dentro dela.
- Ambientes novos: contrato Gymnasium (`reset` → `(obs, info)`, `step` →
  `(obs, reward, terminated, truncated, info)`), RNG injetável e seed
  reprodutível.
- Observações vetoriais `(N,)` usam MLP; imagens `(C,H,W)` usam CNN.
- Ações contínuas exigem `ppo_continuous`; DQN/PPO/A2C são para `Discrete`.
- Checkpoints guardam `config` com os parâmetros da arena; `evaluate`/`play`
  reconstroem o ambiente a partir dele.

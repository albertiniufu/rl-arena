# CLAUDE.md

Monorepo de arenas de RL. Cada arena tem o seu próprio `CLAUDE.md` com os
detalhes de arquitetura e comandos; o Claude Code carrega o arquivo mais
próximo ao diretório de trabalho.

## Estrutura

- `arenas/algs/` — algoritmos compartilhados: `alg_dqn`,
  `alg_doubledqn_per_dueling`, `alg_mka`, `alg_ppo`, `alg_eppo`, `alg_ppo_aes`,
  `alg_gtr`, `alg_a2c`, `alg_ppo_continuous`, `alg_ddpg`, `alg_td3`,
  `alg_sac`, `alg_sac_discrete`, `alg_ippo`, `alg_mappo`, `alg_happo` e
  `alg_registry` (descoberta, checkpoints,
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
- `docs/` — guias para **pessoas** (trilha de leitura, arquitetura,
  algoritmos, como contribuir). `README.md` é a porta de entrada; atualize
  `docs/` junto com mudanças estruturais.
- `configs/` — presets YAML de execução e espaços de busca da BO.
- `arenas/common/queue_gate.py` — `train.py` direto registra um job local e
  espera permissão da Fila de Experimentos; `--no-queue` desliga.
  `train.py`/`evaluate.py`/`play.py` aceitam `--config` e `--set`.
- Fila de Experimentos: `job_queue.py` (fila/scheduler), `job_queue_ui.py`
  (abas Fila, Algoritmos, Análise de resultados e Proposta),
  `algorithm_catalog.py` (algoritmos por arena + docs),
  `experiment_analysis.py` (relatório estatístico via `report.py`) e
  `experiment_proposal.py` (geração/validação MD+YAML, compartilhada com a
  skill `experimenter-helper`).

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
- Ações contínuas usam `ppo_continuous`, `ddpg`, `td3` ou `sac`; DQN/PPO/A2C e `sac_discrete` são para `Discrete`.
- Checkpoints guardam `config` com os parâmetros da arena; `evaluate`/`play`
  reconstroem o ambiente a partir dele.

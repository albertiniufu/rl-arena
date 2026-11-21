# Algoritmos compartilhados

Cada subdiretório `alg_<nome>/` é um algoritmo autocontido. O sufixo do
diretório é o nome usado na CLI: `alg_dqn` → `--alg dqn`.

## Pacotes disponíveis

| Pacote | `--alg` | Tipo |
| --- | --- | --- |
| `alg_dqn` | `dqn` | off-policy (valor) |
| `alg_doubledqn_per_dueling` | `doubledqn_per_dueling` | off-policy (valor) |
| `alg_mka` | `mka` | off-policy (valor, arquitetura custom) |
| `alg_ppo` | `ppo` | on-policy (ator-crítico) |
| `alg_eppo` | `eppo` | on-policy (crítico evidencial) |
| `alg_ppo_aes` | `ppo_aes` | on-policy (entropia adaptativa) |
| `alg_gtr` | `gtr` | on-policy (região de confiança) |
| `alg_a2c` | `a2c` | on-policy (retornos n-step) |
| `alg_ppo_continuous` | `ppo_continuous` | on-policy (ações contínuas) |
| `alg_ddpg` | `ddpg` | off-policy (ações contínuas) |
| `alg_td3` | `td3` | off-policy (ações contínuas) |
| `alg_sac` | `sac` | off-policy (ações contínuas, entropia) |
| `alg_sac_discrete` | `sac_discrete` | off-policy (ações discretas, entropia) |
| `alg_ippo` | `ippo` | multiagente (crítico local) |
| `alg_mappo` | `mappo` | multiagente (crítico centralizado) |
| `alg_happo` | `happo` | multiagente (heterogêneo) |

## Anatomia de um pacote

```text
alg_dqn/
├── __init__.py        # exporta Agent
├── model.py           # rede neural
├── replay_buffer.py   # memória off-policy (ou rollout_buffer.py)
└── agent.py           # class Agent: act/learn/checkpoint
```

O contrato (atributos e métodos) está em
[`docs/algoritmos.md`](../../docs/algoritmos.md). O
[`alg_registry.py`](alg_registry.py) faz a descoberta automática, o
`save_checkpoint`/`load_checkpoint` pesos-only e o adaptador
`CheckpointPolicy`.

## Regra prática

- Código que **não** é específico de um algoritmo (descoberta, checkpoint,
  adaptador de política) fica em `alg_registry.py`.
- Código específico (modelo, buffer, loss, update) fica no pacote `alg_*`.
- Não edite uma lista de algoritmos: crie o diretório com `agent.py` e ele
  aparece sozinho no `--alg`.

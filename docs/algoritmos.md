# Algoritmos

Os algoritmos ficam em `arenas/algs/`. Cada subdiretório `alg_<nome>` é
autocontido e vira uma opção de `--alg <nome>` automaticamente — não existe
lista central para editar.

## Catálogo

| `--alg` | Pacote | Política | Buffer | Ideia central |
| --- | --- | --- | --- | --- |
| `dqn` | `alg_dqn` | off-policy | replay uniforme | Q-learning com rede-alvo e alvo fixo `C = 1000`. |
| `doubledqn_per_dueling` | `alg_doubledqn_per_dueling` | off-policy | replay priorizado | Double DQN + PER rank-based + rede dueling. |
| `mka` | `alg_mka` | off-policy | replay uniforme | Variante do DQN com CNN custom (GELU), para comparação de arquitetura. |
| `ppo` | `alg_ppo` | on-policy | rollout | Surrogate recortado, GAE(λ) e ator categórico. |
| `eppo` | `alg_eppo` | on-policy | rollout | Crítico evidencial (NIG) e vantagem UCB. |
| `ppo_aes` | `alg_ppo_aes` | on-policy | rollout | Coeficiente de entropia agendado pelo erro de TD. |
| `gtr` | `alg_gtr` | on-policy | rollout | Região de confiança gaussiana + âncoras. |
| `a2c` | `alg_a2c` | on-policy | rollout | Retornos n-step e um update por rollout. |
| `ppo_continuous` | `alg_ppo_continuous` | on-policy | rollout | Política gaussiana para `Box`. |
| `ddpg` | `alg_ddpg` | off-policy | replay | Ator determinístico + crítico Q para `Box`. |
| `td3` | `alg_td3` | off-policy | replay | DDPG com dois críticos, target smoothing e ator atrasado. |
| `sac` | `alg_sac` | off-policy | replay | Ator gaussiano com entropia e alpha automático para `Box`. |
| `sac_discrete` | `alg_sac_discrete` | off-policy | replay | SAC categórico para ações discretas. |
| `ippo` | `alg_ippo` | on-policy | rollout por agente | Vários agentes, política compartilhada, crítico local. |
| `mappo` | `alg_mappo` | on-policy | rollout conjunto | Crítico centralizado (CTDE), ator local. |
| `happo` | `alg_happo` | on-policy | rollout | Agentes heterogêneos, update sequencial com correção de razão. |

Off-policy usa **replay** e exploração ε-greedy (ou ruído, em DDPG/TD3); on-policy coleta um
**rollout** e descarta depois do update. SAC usa entropia em vez de ε. A implementação de cada um é
explicada em uma aula HTML:
[`arenas/seek_coin/docs/index.html`](../arenas/seek_coin/docs/index.html).

## Como escolher

```text
Ação é contínua (Box)?
├── sim → quer simplicidade/on-policy? ppo_continuous
│         quer entropia/robustez? sac
│         quer twin critics determinístico? td3
│         quer a referência determinística? ddpg
└── não
    ├── Vários agentes no mesmo ambiente?
    │   ├── sim, agentes homogêneos e crítico local      → ippo
    │   ├── sim, crítico centralizado (CTDE)             → mappo
    │   └── sim, papéis/arquiteturas diferentes          → happo
    └── não
        ├── Quer simplicidade e reprodutibilidade        → dqn
        ├── DQN superestima / replay uniforme é pouco    → doubledqn_per_dueling
        ├── Quer a referência on-policy                  → ppo
        ├── Quer entropia off-policy                     → sac_discrete
        └── Quer variantes de pesquisa                   → eppo, ppo_aes, gtr
```

Regras práticas:

- **Comece pelo `dqn`** para entender o loop; ele é o mais fácil de depurar.
- **Compare no mesmo protocolo**: mesma seed, mesmo `frame_stack`, mesmo
  orçamento. O seek_coin tem scripts para isso (veja o README da arena).
- **Mude uma coisa por vez.** `eppo`, `ppo_aes` e `gtr` são variações do
  PPO; compare-as contra o `ppo`, não contra o `dqn`.

## O contrato de um `alg_*`

O pacote precisa de um `agent.py` que exponha `Agent`. A classe deve ter:

### Atributos

| Atributo | Obrigatório | Papel |
| --- | --- | --- |
| `algorithm` | sim | Nome gravado no checkpoint (deve casar com `--alg`). |
| `arch_version` | sim | Versão da arquitetura; checkpoint incompatível é recusado. |
| `obs_shape` / `n_actions` | sim | Validação do checkpoint e da rede. |
| `online` / `target` | sim | Redes. On-policy usa `target = online`. |
| `optimizer` | sim | Salvo/restaurado no `--resume`. |
| `buffer` | sim | `add(...)`; on-policy também `mark_episode_end()`. |
| `steps` / `epsilon` | sim | Contadores e exploração. |
| `on_policy` | não | `True` desliga replay/ε e faz `learn()` esperar o rollout. |
| `multi_agent` | não | `True` habilita `ArenaMultiEnv` no seek_coin. |
| `continuous` | não | `True` para ações `Box`. |

### Métodos

| Método | Papel |
| --- | --- |
| `act(state, greedy=False)` | Escolhe uma ação. `greedy=True` na avaliação. |
| `act_batch(states)` | Ações para um lote (vetorização). |
| `learn()` | Um passo de aprendizado; retorna a loss ou `None` se ainda não dá. |
| `q_stats(state)` | Estatísticas para TensorBoard (Q ou valor do crítico). |
| `head_weights()` | Pesos da última camada para histogramas. |
| `end_episode(info)` | Opcional. Marca a fronteira do episódio no rollout. |
| `finish_training()` | Opcional. Consome rollout parcial no fim do treino. |
| `add_transition(i, ...)` | Só multiagente. Contribuição do agente `i` no passo. |

O `train_agent` genérico (risk/economy) e o trainer do seek_coin usam esse
contrato. Se você implementar `act` com o parâmetro `action_mask`, o trainer
passa a máscara automaticamente (usado no Risk).

## Adicionar um algoritmo

1. Copie um pacote vizinho como ponto de partida:
   - off-policy discreto: `alg_dqn`;
   - on-policy discreto: `alg_ppo`;
   - on-policy contínuo: `alg_ppo_continuous`;
   - off-policy contínuo: `alg_ddpg`, `alg_td3`, `alg_sac`;
   - off-policy discreto com entropia: `alg_sac_discrete`;
   - multiagente: `alg_ippo` (crítico local) ou `alg_mappo` (centralizado).
2. Renomeie para `alg_<nome>` e ajuste `algorithm` e `arch_version`.
3. Rode `python train.py --arena seek_coin --alg <nome> --episodes 10`.
4. `pytest -q` — o checkpoint e o contrato são cobertos pelos testes da arena.

O `--alg <nome>` passa a aparecer sozinho no `--help`; `evaluate.py` e
`play.py` reconhecem o algoritmo pelo checkpoint.

## Onde cada algoritmo é testado

Os testes de integração dos algoritmos vivem na arena que os exercita
(`arenas/seek_coin/tests/`): DQN/Double DQN em `test_agent.py`, PPO/A2C em
`test_on_policy.py`, EPPO/PPO-AES/GTR em `test_eppo.py`, `test_ppo_aes.py`
e `test_gtr.py`, multiagente em `test_multi_agent.py`/`test_mappo.py`/
`test_happo.py`. DDPG/TD3/SAC contínuos e SAC discreto têm testes em
`arenas/common/tests/test_continuous_algorithms.py` e
`test_sac_discrete.py`; a ASHA em `test_asha_campaign.py`.

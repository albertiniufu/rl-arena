# Registro de experimentos

Índice cronológico. Cada linha aponta para a entrada com protocolo e análise.

> **Pausa para reboot (2026-11-20 22:45).** A máquina foi reiniciada com o
> vazamento do Samba (`rpcd_lsad`) e a Fila **pausada**. Onde cada
> experimento parou e os comandos de retomada estão em
> [`retomada-20261120.md`](retomada-20261120.md).

| ID | Data | Status | Setup | Arena | Algoritmos | Orçamento | Resultado | Conclusão |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [EXP-20261115-bo-level3-6h](EXP-20261115-bo-level3-6h.md) | 2026-11-15 | concluído | S6 | seek_coin nível 3 | `ppo`, `eppo` | ~6 h BO cada | EPPO final: reward 288,77 vs PPO 263,71 | EPPO tunado foi significativamente melhor (p=0,0001; d=0,64) |
| [EXP-20261120-ppo-vs-eppo-level3-10seeds](EXP-20261120-ppo-vs-eppo-level3-10seeds.md) | 2026-11-20 | pausado (OOM-kill 22:26; retomar com `retry` em `job-20261120-183301-9e4e70`) | S6 | seek_coin nível 3 | `ppo`, `eppo` | 2 h/run × 7 seeds | seeds 7 e 42 e 123/PPO concluídas | confirmação multi-seed do ranking da BO |
| [EXP-20261120-ppo-vs-eppo-level4-7seeds](EXP-20261120-ppo-vs-eppo-level4-7seeds.md) | 2026-11-20 | planejado (na fila como `job-20261120-223455-65d70a`, depende do job do nível 3) | S7 | seek_coin nível 4 (visão local 9×9) | `ppo`, `eppo` | 2 h/run × 7 seeds | aguardando | o ranking PPO × EPPO se mantém com observação parcial? |
| [EXP-20261115-ppo-vs-eppo-30min](EXP-20261115-ppo-vs-eppo-30min.md) | 2026-11-15 | concluído | S1 | seek_coin | `ppo`, `eppo` | 30 min cada | PPO latest: reward 8,48 vs EPPO 1,12 | PPO melhor no orçamento curto; EPPO sem tuning e com 1 seed |
| [EXP-20261108-happo-vs-mappo](EXP-20261108-happo-vs-mappo.md) | 2026-11-08 | concluído | S4 | seek_coin | `happo`, `mappo` | ~8 h 16 | HAPPO best 6,05; MAPPO latest 4,71 | HAPPO tem o melhor pico; MAPPO latest é mais equilibrado; repetir com 3 seeds |

## Ranking por condição

O ranking **só vale dentro da mesma condição**: arena + setup + orçamento +
protocolo de avaliação. Para comparar algoritmos, copie a tabela abaixo na
entrada do experimento e preencha com média ± desvio.

### S1 — seek_coin, comparação controlada

| Algoritmo | Passos até o alvo | Recompensa final | Moedas | Estabilidade |
| --- | ---: | ---: | ---: | ---: |
| a preencher |  |  |  |  |

### S3/S4 — seek_coin, multiagente

| Algoritmo | Recompensa do time | Moedas | Mortes | Estabilidade |
| --- | ---: | ---: | ---: | ---: |
| a preencher |  |  |  |  |

> Não misture `best.pt` com `latest.pt` na mesma comparação. Escolha um
> critério e use o mesmo para todos os algoritmos.

## Como atualizar

1. Adicione a linha quando o experimento for registrado (status `planejado`).
2. Atualize para `rodando` ao iniciar.
3. Ao terminar, preencha resultado e conclusão e mude para `concluído`.
4. Se abortar, registre o motivo — isso também é resultado.

# EXP-20261108-happo-vs-mappo

| Campo | Valor |
| --- | --- |
| **ID** | EXP-20261108-happo-vs-mappo |
| **Status** | concluído |
| **Data** | 2026-11-08 a 2026-11-09 |
| **Setup** | S4 — papéis heterogêneos ([`setups.md`](setups.md#histórico)) |
| **Arena** | seek_coin |
| **Cenário** | `collectors-vs-pushers` (4 agentes, 2 times, 1 coletor + 1 empurrador por time) |
| **Algoritmos** | `happo`, `mappo` |
| **Orçamento** | ~8 h 16 de parede (`--max-hours 8.274`), HAPPO 5.544 eps / 2,42 M passos; MAPPO 11.546 eps / 5,05 M passos |
| **Seeds** | 7 |
| **Artefatos** | `experiments/happo_vs_mappo_3h_20261108_1945/mixed/` (local, ignorado pelo git) |

## Pergunta

Em um cenário com papéis heterogêneos (coletores e empurradores no mesmo
time), o HAPPO aprende uma política gulosa melhor que o MAPPO com o mesmo
orçamento de parede?

## Hipótese

Sim. O HAPPO tem atores por agente e atualização sequencial, o que deveria
ajudar quando os agentes têm papéis e funções diferentes. O MAPPO usa um ator
compartilhado e um crítico centralizado; pode aprender mais rápido, mas talvez
não separe bem os papéis.

## Setup e mudanças

- Setup S4: `Scenario.collectors_vs_pushers`, máscaras de ação por papel,
  empurrão de meia arena e `push_cooldown=2 s`.
- `push_coin_transfer=15`.
- Seed 7; `frame_stack=4`, `frame_skip=3`; batch 64; lr 2.5e-4.
- PPO: rollout 1024, 4 épocas, clip 0.2, GAE(λ=0.95).
- Avaliação final: 50 episódios por checkpoint, seeds 1000..1049, política
  gulosa.

## Protocolo

Comando original do run `mixed` (worktree da época; hoje o mesmo caminho está
no `train.py` da `main`):

```bash
python train.py --alg happo --scenario collectors-vs-pushers \
  --agents 4 --teams 2 --seed 7 \
  --frame-stack 4 --frame-skip 3 --batch-size 64 --lr 0.00025 \
  --push-coin-transfer 15 \
  --tensorboard --run-name happo_roles_mixed --stats \
  --max-hours 8.274 --episodes 1000000 \
  --eval-every 100 --eval-episodes 5 --eval-seed 1000 \
  --save-every-minutes 30
```

O MAPPO usa o mesmo comando com `--alg mappo --run-name mappo_roles_mixed`.

## Resultados

Avaliação final com 50 episódios (seeds 1000..1049), política gulosa:

| Run | Checkpoint | Ep. treino | Reward | Moedas | Mortes | Moedas t0 | Moedas t1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **HAPPO** | **best** | 5115 | **6,05** | **8,12** | 0,32 | 11,72 | 4,51 |
| HAPPO | latest | 5544 | 3,56 | 7,75 | 0,22 | 9,05 | 6,44 |
| MAPPO | best | 2423 | 3,57 | 4,32 | 0,62 | 4,24 | 4,40 |
| MAPPO | latest | 11546 | 4,71 | 7,32 | **0,21** | 8,15 | 6,48 |

Artefatos locais:

- `mixed/final_metrics.jsonl` — métricas finais;
- `mixed/final_eval/` — CSVs das avaliações de 50 episódios;
- `mixed/*/checkpoints/{best,latest}.pt`;
- TensorBoard: runs `happo_roles_mixed` e `mappo_roles_mixed`.

## Análise

- **HAPPO best** teve a melhor política gulosa agregada (6,05 de reward), mas
  depende mais do time 0 (11,72 moedas contra 4,51 do time 1).
- **MAPPO best** é enganoso: maior reward de treino (18,40), mas apenas 3,57
  no greedy e o pior índice de mortes (0,62). Selecionar checkpoint por reward
  de treino não funcionou aqui.
- **MAPPO latest** foi o checkpoint mais equilibrado: 4,71 de reward, 0,21 de
  mortes e os dois times próximos (8,15 / 6,48 moedas).
- A correção dos papéis funcionou: em todos os checkpoints avaliados os dois
  times coletam e empurram.
- Menos mortes não implica mais reward: HAPPO latest e MAPPO latest têm as
  menores mortes, mas HAPPO best tem o maior reward.

## Conclusão

- **Eficiência amostral:** o MAPPO coletou ~2,1× mais amostras no mesmo tempo
  de parede (5,05 M contra 2,42 M passos), mas isso não se traduziu em melhor
  política gulosa.
- **Melhor política independente:** HAPPO best.
- **Melhor equilíbrio:** MAPPO latest.
- **Hipótese:** parcialmente confirmada. O HAPPO produziu o melhor pico, mas a
  vantagem não é robusta com uma seed e um checkpoint.
- **Decisão:** repetir com pelo menos 3 seeds e escolher o checkpoint por
  avaliação gulosa periódica, não por reward de treino.

## Próximos passos

- [ ] Repetir S4 com seeds 7, 42 e 123.
- [ ] Avaliar `best` e `latest` com o mesmo protocolo e reportar média ± desvio.
- [ ] Investigar por que o MAPPO best tem reward de treino alto e greedy ruim
      (possível overfitting ao comportamento dos colegas de treino).
- [ ] Medir custo por passo dos dois algoritmos para separar eficiência
      amostral de eficiência de parede.

## Limitações

- Uma seed.
- O run `mixed` teve ~8 h 16, não as 3 h do protocolo original.
- A avaliação gulosa tem 50 episódios; diferenças pequenas podem ser ruído.
- O duelo best × best tem 1 episódio por lado e é apenas ilustrativo.

# EXP-20261115-ppo-vs-eppo-30min

| Campo | Valor |
| --- | --- |
| **ID** | EXP-20261115-ppo-vs-eppo-30min |
| **Status** | concluído |
| **Data** | 2026-11-15 |
| **Setup** | S1 — comparação controlada ([`setups.md`](setups.md#histórico)) |
| **Arena** | seek_coin, nível 0 |
| **Cenário** | single-agent contra `rule` |
| **Algoritmos** | `ppo`, `eppo` |
| **Orçamento** | 30 min de parede cada, executados em paralelo na mesma GPU |
| **Seeds** | treino: 7; avaliação: 50 episódios, seeds 1000..1049 |
| **Artefatos** | `experiments/EXP-20261115-ppo-vs-eppo-30min/` (local) |

## Pergunta

Com o mesmo orçamento curto de parede (30 min), o PPO e o EPPO
produzem políticas gulosas comparáveis no seek_coin?

## Hipótese

O EPPO deveria ser mais exploratório por usar incerteza no crítico, mas com
pouco tempo de treino o custo do update pode dominar. Esperávamos PPO e EPPO
próximos, com possível vantagem do PPO em efficiency de parede.

## Setup e protocolo

- `frame_stack=4`, `frame_skip=3`, batch 64, lr 2.5e-4, seed 7.
- Adversário fixo `rule`.
- Episódios de 90 s; `--max-hours 0.5` em cada algoritmo.
- Os dois treinos rodaram **em paralelo** na mesma RTX 3050 (6 GB), então
  dividiram GPU/CPU; o orçamento de parede foi idêntico, mas o hardware não
  foi exclusivo.
- Avaliação gulosa com 50 episódios (seeds 1000..1049), checkpoint `latest`
  e também `best`.

Comando base (rodado no diretório de cada algoritmo, para separar `logs/`):

```bash
python /caminho/rl-arena/train.py --arena seek_coin --alg ppo \
  --episodes 1000000 --max-hours 0.5 --opponent rule --seed 7 \
  --frame-stack 4 --frame-skip 3 --batch-size 64 --lr 2.5e-4 \
  --save-every 10 --tensorboard --checkpoint-dir checkpoints \
  --run-name ppo_30min
```

O EPPO usa o mesmo comando com `--alg eppo --run-name eppo_30min`.

## Resultados

### Treino

| Algoritmo | Episódios | Checkpoint final |
| --- | ---: | --- |
| PPO | 1185 | `latest.pt` |
| EPPO | 882 | `latest.pt` |

### Avaliação — checkpoint `latest`

50 episódios pareados por seed:

| Algoritmo | Reward média | Mediana | IC 95% | Moedas média |
| --- | ---: | ---: | --- | ---: |
| PPO | 8,484 | 15,650 | [0,224; 16,284] | 22,840 |
| EPPO | 1,116 | 0,000 | [0,652; 1,636] | 1,100 |

| Métrica | Diferença (eppo − ppo) | p corrigido | Efeito | Significativo |
| --- | ---: | ---: | --- | --- |
| Reward | −7,368 | 0,0178 | Cohen's d = −0,254 | sim |
| Moedas | −21,740 | <0,001 | Cohen's d = −1,430 | sim |

### Avaliação — checkpoint `best`

| Algoritmo | Reward média | Moedas média |
| --- | ---: | ---: |
| PPO | −3,378 | 5,340 |
| EPPO | 0,320 | 0,320 |

- Reward: diferença 3,698 (eppo − ppo), p corrigido 0,5085, **não significativo**.
- Moedas: diferença −5,020, p corrigido <0,001, favorece PPO.

## Análise

- **PPO latest foi claramente melhor** em reward e moedas no orçamento de
  30 min. A diferença de moedas tem efeito grande (Cliff/d = −1,43).
- **PPO latest tem alta variância**: média 8,48, mediana 15,65, IC largo. Há
  episódios bons e episódios ruins; a política ainda não estabilizou.
- **EPPO ficou quase sempre em zero** (mediana 0), com 882 episódios. Com os
  mesmos hiperparâmetros do PPO, o EPPO não teve tempo/estabilidade para
  aprender; o resultado não permite concluir que o algoritmo é pior em geral.
- **`best.pt` é enganoso**: o PPO best tem reward médio negativo e moedas
  baixas. O critério de “melhor reward de treino” não selecionou a melhor
  política gulosa. Isso reforça a necessidade de avaliação periódica e de
  escolher checkpoint por avaliação, não por reward de treino.
- A comparação é **uma seed de treino**; a significância estatística dos 50
  episódios de avaliação não substitui repetição com seeds de treino.

## Conclusão

No protocolo de 30 min, **PPO foi mais efetivo que EPPO** para esta
configuração. A hipótese de que ficariam próximos não se confirmou.

Decisão: **não** declarar EPPO pior de forma geral; repetir com tuning
específico do EPPO, 3+ seeds de treino e, se possível, treinos sequenciais
(sem compartilhar GPU) para separar efeito de hardware.

## Próximos passos

- [ ] Repetir com seeds 7, 42 e 123.
- [ ] Rodar PPO e EPPO sequencialmente, com a GPU exclusiva.
- [ ] Dar ao EPPO um orçamento/rollout maior ou tuning próprio (`rollout_steps`, épocas, lr).
- [ ] Avaliar `best` por uma métrica de validação separada, não pelo reward de treino.
- [ ] Registrar a próxima campanha como experimento novo, sem sobrescrever este.

## Limitações

- Uma seed de treino.
- Treinos concorrentes na mesma GPU.
- EPPO com hiperparâmetros do PPO, não tunados.
- 50 episódios de avaliação por checkpoint; a variância do PPO é alta.

## Relatórios gerados

Na pasta local do experimento:

- `report-reward.html` / `report-reward.md` — `latest`
- `report-coins.html` / `report-coins.md` — `latest`
- `report-deaths.html` / `report-deaths.md` — `latest`
- `report-best-reward.html` / `report-best-reward.md` — `best`
- `report-best-coins.md` — `best`
- `results.csv` e `results-best.csv` — dados usados nos relatórios

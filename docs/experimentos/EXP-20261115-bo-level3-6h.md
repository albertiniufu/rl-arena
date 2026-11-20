# EXP-20261115-bo-level3-6h

| Campo | Valor |
| --- | --- |
| **ID** | EXP-20261115-bo-level3-6h |
| **Status** | concluído |
| **Data** | 2026-11-15/16 |
| **Setup** | S6 — nível 3 ([`setups.md`](setups.md#histórico)) |
| **Arena** | seek_coin, nível 3 |
| **Algoritmos** | `ppo`, `eppo` |
| **Orçamento** | ~6 h de BO por algoritmo; 32 trials de 10 min + validação final de 30 min |
| **Seeds** | treino/BO: 7; avaliação final: 50 episódios, seeds 1000..1049 |
| **Artefatos** | `experiments/BO-level3-6h-20261115/` (local) |

## Pergunta

Quais hiperparâmetros fazem PPO e EPPO funcionarem bem no nível 3 do
seek_coin, e qual dos dois é mais efetivo quando ambos recebem uma busca
bayesiana comparável?

## Protocolo

- Nível 3, adversário `rule`, 90 s por episódio.
- `frame_stack=4`, `frame_skip=3`, seed 7 nos trials.
- BO com GP (kernel RBF) + Expected Improvement, 32 trials de 10 min por
  algoritmo, seguidos de uma validação de 30 min do melhor trial.
- Objetivo dos trials: reward médio guloso em 20 episódios, com pequeno
  bônus por moedas (`score = reward + 0.05 * coins`).
- Validação final: 50 episódios, seeds 1000..1049, comparação pareada.
- PPO e EPPO rodaram **sequencialmente**, cada um com sua própria janela de
  ~6 h.

## Melhores hiperparâmetros

### PPO — trial 17

```json
{
  "lr": 0.0006854892425731042,
  "batch_size": 256,
  "rollout_steps": 512,
  "epochs": 4,
  "clip": 0.16816582937835758,
  "entropy_coef": 0.019246082533472705,
  "gae_lambda": 0.9043933720019918,
  "max_grad_norm": 3.315859829149825,
  "value_coef": 0.3229967938087496
}
```

### EPPO — trial 10

```json
{
  "lr": 0.0005960087748299635,
  "batch_size": 128,
  "rollout_steps": 2048,
  "epochs": 4,
  "clip": 0.2800262855915594,
  "entropy_coef": 0.008547273117776729,
  "gae_lambda": 0.9156731636110801,
  "max_grad_norm": 5.12988893677983,
  "ucb_radius": 0.03585985265683301,
  "evidential_coef": 0.025932201655941102,
  "gae_variant": "ind"
}
```

## Resultados

### Validação final — 30 min, 50 episódios

| Algoritmo | Reward média | Mediana | IC 95% | Moedas | Mortes |
| --- | ---: | ---: | --- | ---: | ---: |
| EPPO | **288,766** | 286,300 | [281,202; 296,478] | **145,180** | **0,000** |
| PPO | 263,708 | 267,200 | [255,008; 272,106] | 122,920 | 0,140 |

### Comparações pareadas

| Métrica | Diferença (eppo − ppo) | p corrigido | Efeito | Significativo |
| --- | ---: | ---: | --- | --- |
| Reward | +25,058 | 0,0001 | Cohen's d = 0,642 | sim |
| Moedas | +22,260 | <0,001 | Cohen's d = 0,707 | sim |
| Mortes | −0,140 | 0,0625 | Cohen's d = −0,310 | não |

### Melhor trial de baixa fidelidade

| Algoritmo | Trial | Reward (20 eps) | Moedas |
| --- | ---: | ---: | ---: |
| PPO | 17 | 208,865 | 81,35 |
| EPPO | 10 | 287,830 | 169,25 |

## Análise

- **EPPO tunado foi melhor que PPO tunado** neste protocolo: +25 de reward
  e +22 moedas, com efeito médio (d ≈ 0,64–0,71).
- O EPPO chegou a **0,0 mortes** em média na avaliação final.
- O PPO melhorou muito em relação ao default, mas ainda ficou abaixo do EPPO
  no nível 3 com este orçamento.
- A busca foi feita com uma seed de treino; a validação de 50 episódios é
  robusta para a política final, mas não substitui repetir o treino com 3+
  seeds.
- O ranking de baixa fidelidade já apontava EPPO à frente, e a validação
  final confirmou.

## Conclusão

Com ~6 h de BO por algoritmo, **EPPO foi mais efetivo que PPO no nível 3**
para esta configuração. A diferença em reward e moedas é estatisticamente
significativa na avaliação pareada; a diferença em mortes não é.

Próximo passo: repetir os dois melhores presets com seeds 7, 42 e 123 para
confirmar o ranking e estimar a variância entre treinos.

## Relatórios

Na pasta local da campanha:

- `report-final-reward.html` / `.md`
- `report-final-coins.html` / `.md`
- `report-final-deaths.html` / `.md`
- `ppo/best.json`, `eppo/best.json`
- `ppo/trials.jsonl`, `eppo/trials.jsonl`
- `campaign.json`

## Vídeo do duelo

- `experiments/BO-level3-6h-20261115/duel-eppo-vs-ppo-90s.mp4` — EPPO
  (jogador) contra PPO no nível 3, 90 s em tempo real (frame_skip=3
  reconstruído). O EPPO terminou com 49 moedas e 0 mortes nesse episódio.
- `experiments/BO-level3-6h-20261115/duel-eppo-vs-ppo.mp4` — versão 3× mais
  rápida gerada diretamente pelo `play.py`.

## Limitações

- Uma seed de treino por trial.
- BO de 32 trials por algoritmo, não uma varredura exaustiva.
- O objetivo de baixa fidelidade inclui um pequeno peso para moedas.
- Validação final de 30 min, não do orçamento completo de 6 h.

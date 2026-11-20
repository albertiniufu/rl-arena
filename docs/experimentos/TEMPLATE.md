# EXP-AAAAMMDD-slug

> Copie este arquivo para `docs/experimentos/EXP-AAAAMMDD-slug.md` e preencha.
> Remova as instruções entre `>`.

| Campo | Valor |
| --- | --- |
| **ID** | EXP-AAAAMMDD-slug |
| **Status** | planejado / rodando / concluído / abortado |
| **Data** | AAAA-MM-DD |
| **Setup** | Sx — link para [`setups.md`](setups.md) |
| **Arena** | seek_coin / risk / economy |
| **Cenário** | ex.: `collectors-vs-pushers`, `teams`, `all-vs-all` |
| **Algoritmos** | ex.: `ppo`, `mappo`, `happo` |
| **Orçamento** | ex.: 100 mil `env.step`, 3 h, 300 episódios |
| **Seeds** | treino: ex.: 7, 42, 123; avaliação: ex.: 1000–1049 |
| **Métrica primária** | ex.: `reward_mean`, `coins_mean`, `steps_to_target` (direção: maximizar/minimizar) |
| **Unidade independente** | seed de treino (não o episódio de avaliação) |
| **Tuning** | none / BO / ASHA / grid; espaço, orçamento e seeds de validação |
| **Checkpoint de avaliação** | `latest`, `best_validation` ou passo fixo |
| **Critério de decisão** | o que suporta, o que é inconclusivo e o que abandona |
| **Artefatos** | `experiments/EXP-AAAAMMDD-slug/` |

## Checklist de desenho empírico (antes de rodar)

Copie o checklist para a entrada e marque antes de executar. Detalhes em
[`empirical-design.md`](empirical-design.md).

- [ ] A pergunta cabe em uma frase.
- [ ] A hipótese prevê uma direção/efeito.
- [ ] Existe um resultado que refutaria a hipótese.
- [ ] A métrica primária e a direção estão definidas.
- [ ] O baseline é forte e terá tuning comparável.
- [ ] Há 3+ seeds de treino.
- [ ] O orçamento é o mesmo (passos de ambiente ou compute declarado).
- [ ] Tuning, validação e teste estão separados.
- [ ] O checkpoint será escolhido por validação, não pelo teste.
- [ ] Os critérios de decisão estão escritos.
- [ ] Runs falhas e resultados negativos serão reportados.

## Pergunta

O que este experimento quer responder? Uma frase.

## Hipótese

O que você espera que aconteça e por quê?

## Falsificação

Qual resultado faria você abandonar a hipótese? Seja específico: direção,
métrica, margem e incerteza.

## Claim permitido

O que será possível afirmar se o resultado vier? O que **não** será possível
afirmar com este desenho?

## Desenho e orçamento

- **Braços:** candidato, baseline e ablações.
- **Mesmo setup para todos:** arena, nível, física, `frame_stack/skip`,
  recompensa e duração.
- **Orçamento por run:** passos de ambiente e/ou horas; declare o compute.
- **Seeds de treino:** 3+ (ideal 5+).
- **Tuning:** espaço, orçamento, seeds de validação e regra de seleção.
- **Teste:** seeds separadas das usadas no tuning; uma conclusão por seed de
  treino.
- **Confounds:** GPU compartilhada, ordem de execução, cache, horário etc.

## Setup e mudanças

- Setup base: Sx.
- Mudanças em relação ao setup anterior:
- Arquivos/commits relevantes:

## Protocolo

Comandos exatos (copie e cole, com os hiperparâmetros):

```bash
python train.py --arena seek_coin --alg ... --episodes ... --seed ...
```

- Observação / `frame_stack` / `frame_skip`:
- Orçamento e critério de parada:
- Tuning: espaço, orçamento, seeds de validação e regra de seleção.
- Avaliação: quantos episódios, quais seeds, qual checkpoint e política.
- Unidade independente da análise: seed de treino.
- TensorBoard / logs:

## Resultados

Salve os resultados brutos em `results.csv` (uma linha por seed/execução):

```csv
algorithm,seed,reward_mean,coins_mean
...
```

Gere o relatório estatístico:

```bash
python -m arenas.common.report \
  --input experiments/EXP-AAAAMMDD-slug/results.csv \
  --metric reward_mean --paired --baseline <alg> \
  --out experiments/EXP-AAAAMMDD-slug/report.html
```

O HTML é interativo (gráficos, filtros e tabelas). Para o registro em texto,
troque a extensão para `.md`; para automação, use `.json`.

Resuma aqui as principais linhas do relatório. Use média ± desvio e informe o
número de seeds.

| Algoritmo | Passos até o alvo | Recompensa final | Moedas | Estabilidade (desvio) |
| --- | ---: | ---: | ---: | ---: |
|  |  |  |  |  |

Se houver curvas, aponte os artefatos:

- `results.csv`:
- `report.md`:
- TensorBoard:
- Vídeos:

## Análise

O que os números mostram? O que surpreendeu? O que pode ser ruído?

## Conclusão

- O algoritmo X foi mais efetivo em **passos de ambiente**? Sim/Não/Parcial.
- O algoritmo Y foi melhor em **desempenho final**? Sim/Não/Parcial.
- A hipótese foi confirmada? Sim/Não/Parcial.
- Decisão: manter, repetir com mais seeds, mudar o setup ou abortar.

## Próximos passos

- [ ] ...
- [ ] ...

## Limitações

- Número de seeds, variância, diferenças de setup, custo de avaliação etc.
- Pseudo-replicação: episódios de avaliação da mesma run não são runs
  independentes.
- O que o desenho não permite concluir.

## Checklist final

- [ ] Todas as seeds e runs falhas estão no CSV.
- [ ] A análise primária é por seed de treino.
- [ ] Há diferença com IC, tamanho de efeito e p corrigido.
- [ ] A conclusão não é mais forte que a evidência.
- [ ] O Markdown, o YAML e o `registro.md` concordam.
- [ ] Artefatos, comandos e custo computacional estão apontados.

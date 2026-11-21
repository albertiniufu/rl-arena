# Proposta de experimento — EXP-20261120-ppo-vs-eppo-level4-7seeds

> Gerado pela aba **Proposta de experimentos** da Fila de
> Experimentos. O YAML ao lado é a fonte de verdade para
> automação; os dois arquivos precisam concordar.

| Campo | Valor |
| --- | --- |
| **ID** | EXP-20261120-ppo-vs-eppo-level4-7seeds |
| **Status** | planned |
| **Data** | 2026-11-20T21:20:00-03:00 |
| **Autor** | equipe rl-arena |
| **Arena** | seek_coin |
| **Setup** | S7 |
| **Algoritmos** | ppo, eppo |
| **Baseline** | ppo |
| **Métrica primária** | reward (maximize) |
| **Unidade independente** | train_seed |

## 1. Pergunta

No nível 4 (visão local 9×9, sem super moeda) e com os presets herdados da BO do nível 3, o EPPO continua superando o PPO em recompensa gulosa quando o treino é repetido em 7 seeds?

## 2. Hipótese

O EPPO mantém vantagem sobre o PPO em recompensa mesmo com observação parcial, porque o crítico evidencial reduz a variância da estimativa de valor — que fica mais ruidosa quando o agente vê apenas uma janela local.

## 3. Falsificação

A hipótese é refutada se o EPPO não for superior ao PPO em recompensa, se o IC 95% da diferença pareada incluir zero, se o PPO tiver recompensa significativamente maior ou se a vantagem observada no nível 3 desaparecer por completo no nível 4.

## 4. Evidência existente e lacuna

- Experimentos relacionados: EXP-20261115-bo-level3-6h, EXP-20261120-ppo-vs-eppo-level3-10seeds
- Setups existentes: S6, S7
- Lacuna: A comparação multi-seed PPO × EPPO existe só no nível 3 (S6). O nível 4 muda o shape da observação (recorte egocêntrico 9×9) e desliga botões/super moeda por padrão; não se sabe se o ranking entre os algoritmos se mantém quando a observação passa a ser parcial.
- Somente análise (sem treino novo): não

**Claim permitido:** Se o resultado vier na direção esperada, será possível afirmar que, em S7 (nível 4, view_size 9, sem super moeda), com estes presets fixos herdados da BO do nível 3 e 2 h por run, o EPPO tem recompensa gulosa média superior ao PPO. Não será possível afirmar que o nível 4 é melhor ou pior que o nível 3: o setup muda observação E mecânica (super moeda desligada), e os presets não foram re-tunados para o nível 4.

## 5. Setup e algoritmos

- Arena: seek_coin
- Nível / cenário: 4 / single-agent contra adversário rule, observação local 9×9 sem botões/super moeda
- Setup: S7
- Configs: configs/seek_coin/ppo-level4.yaml, configs/seek_coin/eppo-level4.yaml
- `frame_stack` / `frame_skip`: 4 / 3
- Duração do episódio: 90 s
- Recompensa: recompensa do ambiente no nível 4 (moedas e sobrevivência; sem super moeda)
- Candidatos: ppo, eppo
- Baseline: ppo
- Ablações: —

## 6. Métricas

- **Primária:** reward (direção: maximize)
- Definição: reward guloso médio em 50 episódios de avaliação, por seed de treino
- Alvo: —
- Secundárias: coins

## 7. Seeds e orçamento

- Seeds de treino: 7, 42, 123, 314, 999, 2024, 2025
- Seeds de avaliação: 2000
- Episódios de avaliação: 50
- Passos de ambiente por run: ~4M (estimativa; o orçamento controlado é 2 h de parede e o valor real será registrado no results.csv). A obs 9×9 é menor que a do nível 3 no canal de entrada; o custo por passo não deve ser maior.
- Horas por run: 2
- Número de runs: 14
- Hardware: 1× RTX 3050 6 GB; um treino por vez (PPO, depois EPPO, seed a seed)

## 8. Tuning

- Método: none
- Espaço: —
- Orçamento: — h; trials: —
- Seeds de validação: —
- Regra de seleção: presets fixos herdados da BO do nível 3 (configs/seek_coin/{ppo,eppo}-level4.yaml); sem tuning novo e sem re-tunar para o nível 4 (declarado como limitação)

## 9. Avaliação

- Checkpoint: latest
- Política gulosa: sim
- Truncamento: sem bootstrap; recompensa acumulada do episódio
- Mesmo protocolo para todos os braços: sim

## 10. Análise estatística planejada

- Unidade independente: train_seed
- Pareado: sim
- Teste: wilcoxon
- Confiança: 0.95
- Tamanho de efeito: cohens_d
- Correção: holm
- Primária: uma linha por seed de treino; Wilcoxon pareado e Cohen's d
- Secundária: por episódio, apenas como diagnóstico e com ressalva de pseudo-replicação

## 11. Custo estimado

| Item | Estimativa |
| --- | ---: |
| Runs de treino | 14 |
| Horas de treino | 28 h |
| Horas de tuning | — |
| Total | 28 h |

## 12. Riscos e confounds

- Presets não tunados para o nível 4: os hiperparâmetros vêm de trials do nível 3 com super moeda ligada. Um resultado negativo pode refletir falta de tuning, não o algoritmo.
- Observação parcial: o nível 4 recorta a grade em 9×9 células e não há memória recorrente (frame stack fixo). Parte da dificuldade pode dominar a diferença entre PPO e EPPO.
- Comparação entre níveis é inválida: o nível 4 desliga botões/super moeda por padrão. Este experimento compara PPO × EPPO dentro de S7, não S6 × S7.
- Orçamento de parede: PPO e EPPO podem dar números diferentes de passos de ambiente em 2 h.
- Memória da máquina: o host está com pouca memória disponível (swap cheio); um OOM no meio da campanha invalidaria seeds. O runner pula runs com marcador DONE, então é possível retomar.
- Poder limitado: com 7 pares o p bilateral mínimo do Wilcoxon é 0,015625 e, após Holm em 2 métricas, 0,03125 — só atinge 0,05 se o EPPO vencer nas 7 seeds.
- Checkpoint latest pode não ser a melhor política de cada run.

## 13. Critério de decisão

- **Suporta:** EPPO > PPO em reward ou em coins, com IC 95% da diferença pareada inteiramente acima de zero, p Holm < 0,05 e Cohen's d >= 0,5.
- **Inconclusivo:** IC da diferença inclui zero ou p Holm não significativo nas duas métricas; não declarar vencedor.
- **Abandona:** PPO é significativamente melhor em reward ou o EPPO falha de forma sistemática; registrar o resultado negativo e revisar presets para o nível 4 antes de repetir.

## 14. Comandos previstos

```bash
/home/albertini@win.ufu.br/venvs/main/bin/python scripts/rigorous_validation.py --hours 2 --seeds 7,42,123,314,999,2024,2025 --level 4 --view-size 9 --no-super-coin --family-metrics reward,coins --alpha 0.05 --eval-episodes 50 --eval-seed 2000 --ppo-config configs/seek_coin/ppo-level4.yaml --eppo-config configs/seek_coin/eppo-level4.yaml --out /home/albertini@win.ufu.br/rl-arena/experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds
```

Relatório estatístico:

```bash
python -m arenas.common.report --input experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/results-by-train-seed.csv --metric reward --paired --out experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/report.html
```

## 15. Execução na Fila de Experimentos

```yaml
command:
- /home/albertini@win.ufu.br/venvs/main/bin/python
- scripts/rigorous_validation.py
- --hours
- '2'
- --seeds
- 7,42,123,314,999,2024,2025
- --level
- '4'
- --view-size
- '9'
- --no-super-coin
- --family-metrics
- reward,coins
- --alpha
- '0.05'
- --eval-episodes
- '50'
- --eval-seed
- '2000'
- --ppo-config
- configs/seek_coin/ppo-level4.yaml
- --eppo-config
- configs/seek_coin/eppo-level4.yaml
- --out
- /home/albertini@win.ufu.br/rl-arena/experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds
cwd: /home/albertini@win.ufu.br/rl-arena/.claude/worktrees/level-4-visao-local
max_runtime: 144000
priority: 0
depends_on:
- job-20261120-183301-9e4e70
```

## 16. Artefatos

- Pasta do experimento: experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds
- CSV de resultados: experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/results-by-train-seed.csv
- Relatórios: experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/report-by-seed-reward.md, experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/report-by-seed-coins.md, experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/report-by-seed-deaths.md, experiments/EXP-20261120-ppo-vs-eppo-level4-7seeds/family-tests.md
- Documentos: docs/experimentos/EXP-20261120-ppo-vs-eppo-level4-7seeds.md, docs/experimentos/EXP-20261120-ppo-vs-eppo-level4-7seeds.yaml
- Registro: docs/experimentos/registro.md

## 17. Checklist final

- [ ] Pergunta, hipótese e falsificação.
- [ ] Baseline forte e tuning comparável.
- [ ] Mesmo orçamento em passos de ambiente.
- [ ] Tuning separado da avaliação.
- [ ] 3+ seeds de treino.
- [ ] Métrica primária definida antes.
- [ ] IC, teste, efeito e correção.
- [ ] Custo declarado.
- [ ] Registro em `docs/experimentos/`.

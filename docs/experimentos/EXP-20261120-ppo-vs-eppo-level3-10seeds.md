# Proposta de experimento — EXP-20261120-ppo-vs-eppo-level3-10seeds

> Proposta em Markdown. O YAML equivalente é
> [`EXP-20261120-ppo-vs-eppo-level3-10seeds.yaml`](EXP-20261120-ppo-vs-eppo-level3-10seeds.yaml).
>
> Status: rodando
> Autor: equipe rl-arena
> Data: 2026-11-20
> Arena / setup: seek_coin, nível 3, S6
> Algoritmos: `ppo` (baseline), `eppo` (candidato)

## 1. Pergunta

Com os presets tunados da BO e 2 h de treino por run, o **EPPO** supera o
**PPO** no nível 3 do `seek_coin` em recompensa gulosa quando o treino é
repetido em **6 seeds**?

## 2. Hipótese

O EPPO mantém uma vantagem em recompensa sobre o PPO. A campanha BO anterior
favoreceu o EPPO, e o crítico evidencial deve reduzir a variância da
estimativa de valor, tornando a política final mais estável entre seeds.

## 3. Falsificação

A hipótese é refutada se:

- o EPPO não for superior ao PPO em recompensa;
- o IC 95% da diferença pareada incluir zero;
- o PPO tiver recompensa significativamente maior;
- a vantagem desaparecer quando o treino é repetido em 6 seeds.

## 4. Evidência existente e lacuna

- [`EXP-20261115-bo-level3-6h`](EXP-20261115-bo-level3-6h.md): EPPO tunado
  teve reward final 288,77 contra 263,71 do PPO, com p=0,0001 e d=0,64.
  Limitação: **1 seed de treino** por algoritmo.
- [`EXP-20261115-ppo-vs-eppo-30min`](EXP-20261115-ppo-vs-eppo-30min.md):
  comparação curta no **nível 0** com hiperparâmetros iguais; PPO foi melhor
  em 30 min, mas o EPPO não estava tunado.
- **Lacuna:** falta uma confirmação no nível 3, com presets fixos e 6 seeds
  de treino, separando avaliação de teste e medindo a variabilidade entre
  runs.

Este experimento não é uma auditoria dos experimentos antigos; é uma
confirmação direta da hipótese que a BO deixou em aberto.

## 5. Setup e algoritmos

- Arena: `seek_coin`.
- Nível / cenário: nível 3, single-agent contra adversário `rule`.
- Setup: **S6** (botão, cancelamento na plataforma e super moeda).
- Algoritmos candidatos: `ppo`, `eppo`.
- Baseline: `ppo`.
- Ablações: nenhuma.
- Configs fixos:
  - [`configs/seek_coin/ppo-level3.yaml`](../../configs/seek_coin/ppo-level3.yaml)
  - [`configs/seek_coin/eppo-level3.yaml`](../../configs/seek_coin/eppo-level3.yaml)
- Episódios de 90 s; `frame_stack=4`, `frame_skip=3`.

Os presets **não serão re-tunados**. A pergunta é sobre o ranking dos
melhores presets já encontrados, não sobre qual algoritmo tem maior teto de
tuning.

## 6. Métricas

- **Família de testes (2 métricas):**
  - `reward` (primária) — reward guloso médio em 50 episódios de avaliação,
    calculado primeiro por seed de treino;
  - `coins` (co-primária) — moedas coletadas por episódio de avaliação.
- **Diagnóstico (sem teste, sem threshold):** `deaths`, `duration_steps`,
  `env_steps_real`.
- Definição operacional: média dos 50 episódios de avaliação de uma run;
  a análise primária usa uma linha por `(algoritmo, seed de treino)`.
- Alvo de eficiência amostral: não se aplica; o orçamento é fixo em 2 h de
  parede por run.

## 7. Seeds e orçamento

- Seeds de treino: **7, 42, 123, 314, 999, 2024, 2025** (7 seeds).
- Seed base de avaliação: **2000** → 50 episódios com seeds 2000..2049.
- Episódios de avaliação: **50 por run**.
- Passos de ambiente por run: estimativa ~4M; o valor real será registrado.
- Horas por run: **2 h de parede**.
- Número de runs: **14** (2 algoritmos × 7 seeds).
- Hardware: 1× RTX 3050 6 GB; **um treino por vez** — PPO e depois EPPO,
  seed a seed (`scripts/rigorous_validation.py`).

> **Desvio de protocolo (2026-11-20 17:58).** As seeds 7 e 42 rodaram com
> PPO e EPPO **simultâneos** na mesma máquina (12:13–16:15). A seed 123 foi
> interrompida porque os dois treinos consumiam recursos demais; desde então
> o runner executa **um treino por vez** e a seed 123 precisa ser refeita do
> zero (não havia marcador `DONE`). As seeds 7/42 sofreram contenção de
> recursos e as demais não — isso deve ser discutido na análise, e o número
> de passos de ambiente por run pode diferir entre os dois regimes.
>
> **Decisão de orçamento (2026-11-20 18:35).** O plano foi reduzido de 10
> para **7 seeds** (`7, 42, 123, 314, 999, 2024, 2025`) para caber no custo em
> execução sequencial, e a família de testes foi restrita a **`reward` e
> `coins`** (`deaths` passou a diagnóstico). As contas mandam no número de
> seeds: com 5 pares o Wilcoxon bilateral tem p mínimo 0,0625 e com 6 pares
> tem 0,03125, mas o Holm sobre 2 métricas multiplica por 2 — só com **7
> pares** o menor p corrigido é `2 × 2/128 = 0,03125 < 0,05`.

> O orçamento controlado é de **parede**. Como PPO e EPPO têm custos de
> update diferentes, o número de passos de ambiente pode variar. O valor real
> será reportado junto com o resultado.

## 8. Tuning

- Espaço de busca: nenhum.
- Orçamento de BO/ASHA: nenhum.
- Conjunto de validação: não se aplica.
- Melhor config: presets fixos da campanha BO anterior.
- Reprodutibilidade: os dois YAMLs de config são versionados; os
  hiperparâmetros também ficam em `campaign.json`.

## 9. Avaliação

- Checkpoint: `latest.pt` ao final das 2 h.
- Política gulosa: sim.
- Truncamento: recompensa acumulada do episódio; sem bootstrap de valor na
  avaliação.
- Mesmo protocolo para todos os braços: sim — mesmas 50 seeds de avaliação,
  mesmo adversário, mesmo nível e mesma duração.

## 10. Análise estatística planejada

- Unidade independente: **seed de treino**.
- Teste: Wilcoxon signed-rank pareado (PPO e EPPO treinados nas mesmas seeds).
- **Família de hipóteses:** `reward` e `coins` (2 testes), corrigida por
  **Holm–Bonferroni**. `deaths`, `duration_steps` e `env_steps_real` ficam
  fora da família, como diagnóstico.
- IC: bootstrap 95% para a diferença `EPPO − PPO`.
- Tamanho de efeito: Cohen's d pareado; se infinito, reportar Cliff's delta.
- Artefato da família: `family-tests.md` e `family-tests.json` com p bruto e
  p Holm por métrica (gerados no fim de `scripts/rigorous_validation.py`).
- Análise primária: `results-by-train-seed.csv`, uma linha por seed.
- Análise secundária: `results.csv`, por episódio, apenas como diagnóstico e
  com ressalva de pseudo-replicação.
- Resultados negativos/falhas: todas as runs e seeds serão mantidas no CSV;
  runs abortadas serão registradas com o motivo.

## 11. Custo estimado

| Item | Estimativa |
| --- | ---: |
| Runs de treino | 14 |
| Horas de treino | 28 h de parede (um run por vez) |
| Horas de tuning | 0 |
| Horas de avaliação | ~14 min (14 runs × ~1 min) |
| **Total** | **~28–29 h de parede** |

## 12. Riscos e confounds

- **Orçamento de parede:** os algoritmos podem dar números diferentes de
  passos de ambiente em 2 h.
- **GPU compartilhada:** as seeds 7 e 42 rodaram com PPO+EPPO em paralelo e
  as demais rodam uma de cada vez; a contenção de recursos foi diferente
  entre os dois regimes (ver desvio de protocolo na seção 7).
- **Presets de 10 min:** os hiperparâmetros foram escolhidos em trials curtos;
  2 h pode mudar o ranking.
- **Checkpoint `latest`:** não é escolhido por validação; isso é intencional
  para comparar o orçamento fixo, mas pode não refletir o melhor momento.
- **Conclusão inconclusiva:** com 7 seeds o poder ainda é limitado e a
  variância pode impedir uma conclusão forte; o p bilateral mínimo com 7
  pares é 0,015625 e, após Holm em 2 métricas, 0,03125 (ver §13).

## 13. Critério de decisão

Declarado antes de rodar:

- **Suporta:** EPPO > PPO em `reward` **ou** em `coins` com IC 95% da
  diferença pareada inteiramente acima de zero, p Holm < 0,05 e Cohen's d ≥ 0,5.
- **Inconclusivo:** IC inclui zero ou p Holm não significativo nas duas
  métricas; não declarar vencedor.
- **Abandona:** PPO é significativamente melhor em `reward` ou o EPPO falha
  de forma sistemática; registrar o resultado negativo e revisar os presets.

> **Poder do teste com 7 seeds e Holm em 2 métricas:** o Wilcoxon bilateral
> com 7 pares tem p mínimo `2/128 = 0,015625`; multiplicado por 2 (Holm, m=2)
> dá **0,03125 < 0,05**. Só é atingível quando o EPPO vence nas 7 seeds. Com
> 6 pares o mínimo corrigido seria `2 × 0,03125 = 0,0625` e o critério seria
> impossível — foi isso que fixou o plano em **7 seeds**.

## 14. Comandos previstos

Treino + avaliação + relatórios:

```bash
python scripts/rigorous_validation.py \
  --hours 2 \
  --seeds 7,42,123,314,999,2024,2025 \
  --family-metrics reward,coins --alpha 0.05 \
  --eval-episodes 50 --eval-seed 2000 \
  --ppo-config configs/seek_coin/ppo-level3.yaml \
  --eppo-config configs/seek_coin/eppo-level3.yaml \
  --out experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds
```

Relatório primário manual, se necessário:

```bash
python -m arenas.common.report \
  --input experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds/results-by-train-seed.csv \
  --metric reward --paired --baseline ppo \
  --out experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds/report-by-seed-reward.html
```

## 15. Execução na Fila de Experimentos

O YAML já contém o bloco `execution`:

```yaml
execution:
  command:
    - python
    - scripts/rigorous_validation.py
    - --hours
    - "2"
    - --seeds
    - "7,42,123,314,999,2024,2025"
    - --family-metrics
    - reward,coins
    - --alpha
    - "0.05"
    - --eval-episodes
    - "50"
    - --eval-seed
    - "2000"
    - --ppo-config
    - configs/seek_coin/ppo-level3.yaml
    - --eppo-config
    - configs/seek_coin/eppo-level3.yaml
    - --out
    - experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds
  cwd: "."
  max_runtime: 144000
  priority: 0
  depends_on: []
```

Enfileirar:

```bash
python scripts/experiment.py \
  --experiment docs/experimentos/EXP-20261120-ppo-vs-eppo-level3-10seeds.yaml
```

## 16. Artefatos

- Checkpoints:
  `experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds/seed-*/{ppo,eppo}/checkpoints/latest.pt`
- CSVs:
  - `results.csv` — uma linha por episódio;
  - `results-by-train-seed.csv` — uma linha por seed de treino.
- Relatórios:
  - `report-by-seed-{reward,coins,deaths}.md`;
  - `report-per-episode-{reward,coins,deaths}.md`.
- Família de testes (Holm em `reward`/`coins`):
  - `family-tests.md` — tabela com p bruto, p Holm, IC e Cohen's d;
  - `family-tests.json` — mesma informação em formato consumível.
- Pasta do experimento:
  `experiments/EXP-20261120-ppo-vs-eppo-level3-10seeds/`.
- Entrada no `registro.md`: adicionar quando o status mudar para `rodando`.

## 17. Checklist final

- [x] Pergunta, hipótese e falsificação.
- [x] Baseline forte: PPO tunado da BO.
- [x] Mesmo orçamento de parede para os dois algoritmos.
- [x] Sem tuning novo; presets fixos.
- [x] 7 seeds de treino (mínimo para Holm em 2 métricas atingir p < 0,05).
- [x] Métrica primária definida antes.
- [x] Checkpoint fixo (`latest`) para todos.
- [x] IC, teste pareado e tamanho de efeito planejados.
- [x] Custo declarado.
- [x] Registro e artefatos previstos.

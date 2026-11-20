# Experimentos

Este diretório organiza a evolução dos experimentos em dois eixos:

1. **Setup** — o que mudou no ambiente, no treino ou no protocolo.
2. **Algoritmos** — quais foram mais efetivos em cada condição.

A regra é simples: **resultados pesados ficam fora do git; o registro
legível fica aqui.**

## O que é versionado e o que é local

| Onde | O que guardar | Git |
| --- | --- | --- |
| `docs/experimentos/` | pergunta, hipótese, setup, comandos, seeds, orçamento, avaliação, resumo e conclusão | versionado |
| `experiments/<ID>/` | checkpoints, CSVs, TensorBoard, vídeos, datasets | ignorado |
| `checkpoints/`, `logs/`, `videos/` | artefatos de execução | ignorado |

Assim, alguém que clona o repositório entende **o que foi testado e por quê**,
mesmo sem os arquivos grandes.

## Arquivos

| Arquivo | Papel |
| --- | --- |
| [`registro.md`](registro.md) | índice cronológico e ranking por condição |
| [`setups.md`](setups.md) | evolução dos setups (S0, S1, ...) |
| [`TEMPLATE.md`](TEMPLATE.md) | modelo para um novo experimento, com checklist de desenho empírico |
| [`empirical-design.md`](empirical-design.md) | notas práticas de Empirical Design in RL e critérios de decisão |
| [`exemplos/`](exemplos/README.md) | CSV sintético e HTML de exemplo da interface |
| [`EXP-20261108-happo-vs-mappo.md`](EXP-20261108-happo-vs-mappo.md) | exemplo preenchido |

## Como registrar um experimento

Antes de tudo, leia [`empirical-design.md`](empirical-design.md) e use o
checklist do [`TEMPLATE.md`](TEMPLATE.md).

1. Escreva a **pergunta**, a **hipótese** e a **falsificação** antes de rodar.
2. Escolha um setup em [`setups.md`](setups.md). Se mudou algo relevante,
   crie um novo `Sx` e descreva a mudança.
3. Crie o registro:

   ```bash
   cp docs/experimentos/TEMPLATE.md docs/experimentos/EXP-20261120-meu-teste.md
   ```

4. Rode e salve os artefatos em `experiments/EXP-20261120-meu-teste/`.
5. Preencha os resultados com **média ± desvio** e pelo menos 3 seeds quando
   for comparar algoritmos.
6. Adicione uma linha em [`registro.md`](registro.md).
7. Termine com uma **conclusão** e um **próximo passo** (ou “abortado” e o
   motivo).

## Nomenclatura

- ID: `EXP-AAAAMMDD-slug` — ex.: `EXP-20261108-happo-vs-mappo`.
- Pasta local: `experiments/<ID>/`.
- Runs do TensorBoard: `<alg>_<setup>_<oponente>_seed<seed>`.
- Checkpoints: `experiments/<ID>/<alg>/checkpoints/{best,latest}.pt`.

## Como comparar algoritmos de forma justa

Para o raciocínio completo — hipótese, falsificação, unidade independente,
tuning × teste e critérios de decisão — leia
[`empirical-design.md`](empirical-design.md). O resumo abaixo é o mínimo
operacional.

Antes de dizer que um algoritmo é “melhor”, garanta:

- **mesmo setup** (mesma arena, mesmas regras, mesmo `frame_stack/skip`);
- **mesmo orçamento em passos de ambiente** (`env.step`), não em horas;
- **mesmas seeds** (mínimo 3) e mesma política de avaliação;
- **mesma métrica**: recompensa, moedas, RAROC etc.;
- **mesmo checkpoint de avaliação** (`best.pt`, `latest.pt` ou passo fixo).

Se o setup mudou, a comparação entre algoritmos de setups diferentes é
**exploratória**, não conclusiva. Registre isso na conclusão.

## Relatório estatístico

O script [`arenas/common/report.py`](../../arenas/common/report.py) lê os
resultados e gera um relatório com IC, teste de hipótese e tamanho de efeito.
O formato canônico é um CSV com **uma linha por seed/execução**:

```csv
algorithm,seed,reward_mean,coins_mean
dqn,7,12.3,10.1
dqn,42,11.8,9.7
ppo,7,10.1,8.2
ppo,42,9.8,7.9
```

Exemplo:

```bash
python -m arenas.common.report \
  --input experiments/EXP-20261120-meu-teste/results.csv \
  --metric reward_mean --paired --baseline dqn \
  --out experiments/EXP-20261120-meu-teste/report.md
```

O relatório inclui:

- descritivas por algoritmo (n, média, mediana, desvio, IQR e IC);
- diferença `A - B` com IC bootstrap;
- Wilcoxon pareado (com `--paired`) ou Mann-Whitney independente;
- tamanho de efeito (Cohen's d pareado ou Cliff's delta);
- correção de múltiplas comparações (Holm-Bonferroni ou Benjamini-Hochberg);
- conclusão textual e avisos (poucas seeds, seeds repetidas etc.).

Além do Markdown, `--out report.html` gera uma **interface interativa** com:

- gráfico de médias com barras de IC;
- distribuição dos valores por seed (box + pontos);
- diferenças `A - B` com IC;
- dispersão efeito × p-valor corrigido;
- filtro de algoritmos, tabela ordenável e download do JSON.

```bash
python -m arenas.common.report \
  --input experiments/EXP-20261120-meu-teste/results.csv \
  --metric reward_mean --paired --baseline dqn \
  --out experiments/EXP-20261120-meu-teste/report.html
```

Exemplo visual com dados sintéticos:
[`exemplos/report-exemplo.html`](exemplos/report-exemplo.html).

Para dados aninhados, use os campos com pontos:

```bash
python -m arenas.common.report \
  --input experiments/EXP-20261108-happo-vs-mappo/manual_eval.jsonl \
  --value-field metrics.overall.reward_mean \
  --algorithm-field tag \
  --metric reward_mean --out /tmp/report.md
```

Se o arquivo tiver uma seed por linha e os mesmos valores nos dois
algoritmos, acrescente `--seed-field <campo> --paired`.

> Com `--paired`, cada seed precisa aparecer nos dois algoritmos. Sem seed
> comum, o script falha com mensagem clara. Para comparar checkpoints
> diferentes, use uma linha por checkpoint e trate cada um como um algoritmo
> separado (ex.: `happo_best`, `happo_latest`).

Se o CSV tiver vários episódios por seed de treino, agregue antes de comparar:

```bash
python -m arenas.common.report \
  --input experiments/EXP-.../results.csv \
  --seed-field train_seed --aggregate --paired --baseline dqn \
  --metric reward_mean --out report.md
```

`--aggregate` transforma as linhas repetidas da mesma seed/unidade em uma
média. Sem isso, o relatório emite um aviso de possível pseudo-replicação. A
análise primária deve usar **seed de treino** como unidade independente;
episódios de avaliação são análise secundária.

## O que significa “mais efetivo”

Priorize nesta ordem:

1. **Eficiência amostral** — passos de ambiente até atingir o alvo.
2. **Desempenho final** — recompensa/métrica no mesmo orçamento.
3. **Estabilidade** — desvio entre seeds e variância da curva.
4. **Custo** — tempo de update, memória e uso de GPU (secundário).

## Ranking

O ranking fica em [`registro.md`](registro.md). Ele é **por condição**
(arena + setup + orçamento), nunca global. Para uma visão nova, copie a
tabela e recalcule apenas dentro da mesma condição.

## Automação

O relatório estatístico é gerado pelo script acima. O índice
[`registro.md`](registro.md) ainda é manual; se o volume crescer, ele pode
ser gerado a partir dos CSVs de cada experimento. O importante é manter o
formato dos campos estável (`algorithm`, `seed`, métrica).

A skill **`experimenter-helper`** (em
[`.agents/skills/experimenter-helper/`](../../.agents/skills/experimenter-helper/SKILL.md))
ajuda a propor novos experimentos e gera o par `EXP-....md` + `EXP-....yaml`.
O YAML estruturado é validado por
[`scripts/validate_yaml.py`](../../.agents/skills/experimenter-helper/scripts/validate_yaml.py).
A aba **Proposta de experimentos** da
[Fila](../fila-de-experimentos.md) usa o mesmo validador
([`experiment_proposal.py`](../../arenas/common/experiment_proposal.py)) para
pré-visualizar, salvar sem sobrescrever e enfileirar a proposta; a aba
**Análise de resultados** gera o relatório estatístico de um experimento com
o mesmo [`report.py`](../../arenas/common/report.py).

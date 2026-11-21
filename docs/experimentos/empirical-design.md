# Desenho empírico de experimentos

Notas de leitura e tradução prática de **Empirical Design in Reinforcement
Learning** (Patterson, Neumann, White & White, JMLR;
[arXiv:2304.01315](https://arxiv.org/abs/2304.01315)) para o `rl-arena`.

O objetivo não é provar que um algoritmo é melhor. É montar uma comparação
que **possa refutar** a hipótese, com orçamento e critérios definidos antes de
ver o resultado.

> Este documento é um guia de engenharia, não um resumo completo do paper.
> Quando houver dúvida, consulte o paper e registre a decisão no experimento.

## Por que isso importa

Em RL é fácil produzir um resultado que parece bom e não significa nada:

- uma seed só;
- baseline sem tuning;
- métrica escolhida depois;
- checkpoint escolhido no conjunto de teste;
- muitos episódios de avaliação tratados como se fossem runs independentes;
- p-valor sem tamanho de efeito;
- resultado negativo que não foi registrado.

O desenho empírico é o que separa “rodou e deu número” de “temos evidência
para esta afirmação”.

## Fluxo mínimo

```text
pergunta
  -> hipótese + falsificação
  -> métrica primária
  -> baseline forte
  -> orçamento e seeds
  -> tuning em validação
  -> avaliação em teste
  -> estatística com incerteza
  -> decisão (suporta / inconclusivo / abandona)
  -> registro
```

Nenhuma etapa começa por `python train.py`. O comando é consequência do
desenho.

## Princípios traduzidos para o repositório

| Princípio | Como aplicar aqui |
| --- | --- |
| Pergunta antes do comando | Escreva a pergunta no `TEMPLATE.md` antes de criar o YAML de execução. |
| Métrica primária definida antes | Escolha `reward`, `coins`, `raroc` ou `steps_to_target` e registre a direção (`maximize`/`minimize`). |
| Baseline forte | Compare com a melhor configuração razoável do baseline, não com o default sem tuning. |
| Tuning justo | Todos os braços recebem orçamento comparável de BO/ASHA. |
| Validação separada do teste | O checkpoint e os hiperparâmetros são escolhidos em seeds de validação; a conclusão usa seeds de teste. |
| 3+ seeds de treino | 3 é o mínimo; 5+ é melhor. Cada seed de treino é uma unidade independente. |
| Episódios de avaliação não são seeds | Vários episódios da mesma run são pseudo-replicação; a análise primária agrega por seed de treino. |
| Mesmo orçamento | Compare em passos de ambiente ou compute declarado; tempo de parede sozinho é frágil. |
| Incerteza explícita | Reporte média, mediana, desvio, IC e distribuição — não só a média. |
| Efeito + p-valor + IC | Significância sem magnitude é incompleta; reporte Cohen's d ou Cliff's delta. |
| Correção de múltiplas comparações | Use Holm-Bonferroni ou Benjamini-Hochberg quando comparar vários pares. |
| Nada de cherry-picking | Reporte todas as seeds, runs abortadas, falhas e resultados negativos. |
| Checkpoint por validação | Nunca escolha `best.pt` olhando o conjunto de teste. |
| Custo explícito | Horas, trials, GPU/CPU, memória e número de runs fazem parte do resultado. |

## Unidade independente

A pergunta “quantas amostras eu tenho?” não é respondida pelo número de
episódios de avaliação.

- **Unidade primária:** seed de treino (uma run completa).
- **Unidade secundária:** episódio de avaliação dentro de uma run.

Com 3 seeds de treino e 50 episódios por seed, você tem **3 unidades
independentes**, não 150. Os 50 episódios ajudam a estimar a média daquela
run, mas não substituem runs independentes.

Consequências práticas:

- análise primária: uma linha por `(algoritmo, seed de treino)`;
- análise secundária: episódios, com a ressalva de pseudo-replicação;
- `report.py --paired` emparelha por seed;
- `report.py --aggregate` agrega linhas repetidas da mesma seed antes de
  comparar;
- `rigorous_validation.py` grava `results-by-train-seed.csv` para a análise
  primária e `results.csv` para a secundária.

## Métrica primária

Defina **uma** métrica antes de rodar. Exemplos:

| Pergunta | Métrica primária | Direção |
| --- | --- | --- |
| Quem aprende mais rápido? | passos de ambiente até o alvo | minimizar |
| Quem tem melhor política final? | recompensa gulosa média no mesmo orçamento | maximizar |
| Quem coleta mais? | moedas por episódio | maximizar |
| Quem é mais estável? | desvio/taxa de falha entre seeds | minimizar |

Métricas secundárias são permitidas, mas não podem substituir a primária
depois de ver o resultado.

## Baseline e ablações

- **Baseline forte:** o melhor baseline razoável, com tuning comparável.
- **Ablação:** quando o claim é “o componente X ajuda”, compare `A` com
  `A − X`, mantendo todo o resto idêntico.
- **Mesmo setup:** arena, nível, física, `frame_stack`, `frame_skip`,
  recompensa e duração iguais entre os braços.
- **Mesmo protocolo de avaliação:** mesmos episódios, seeds, checkpoint e
  política gulosa.

Se o setup mudou, a comparação entre setups diferentes é exploratória, não
conclusiva.

## Tuning, validação e teste

Um desenho defensável separa três coisas:

1. **Tuning:** BO/ASHA/grid escolhe hiperparâmetros usando seeds de validação.
2. **Seleção:** o checkpoint é escolhido pela validação, não pelo teste.
3. **Teste:** as seeds de teste são usadas uma vez, para a conclusão.

No repositório:

- `scripts/bo_campaign.py` usa uma seed de avaliação para os trials e uma
  seed separada (`--final-eval-seed`) para a validação final.
- `scripts/asha_campaign.py` promove/poda trials por rounds; mantenha as
  seeds de validação fixas.
- `scripts/rigorous_validation.py` repete o treino nas seeds de teste e
  agrega por seed de treino.

Evite escolher a “melhor config” olhando o resultado final. Se isso
acontecer, o resultado é exploratório.

## Estatística mínima

Para cada comparação, reporte:

- diferença `A − B`;
- IC bootstrap da diferença;
- teste pareado (Wilcoxon) quando as seeds são as mesmas;
- teste independente (Mann-Whitney) quando não são;
- tamanho de efeito (Cohen's d pareado ou Cliff's delta);
- p-valor corrigido para múltiplas comparações;
- número de unidades independentes.

Com 3 seeds, um teste pareado tem pouco poder. Trate como piloto e não
declare vencedor definitivo. Se o IC cruza zero e o p corrigido não é
significativo, a conclusão correta é **inconclusivo**, não “tendência”.

## Critérios de decisão

Antes de rodar, escreva:

- **Suporta:** qual resultado apoiaria a hipótese.
- **Inconclusivo:** qual resultado não permite decidir.
- **Abandona:** qual resultado refutaria ou faria repetir o desenho.

Sem isso, qualquer resultado vira “quase significativo”.

## O que não se aplica ao escopo didático

O paper discute práticas de pesquisa em RL em escala de benchmark. Neste
repositório:

- não vamos reproduzir benchmarks grandes nem treinar por semanas;
- não vamos implementar teoria nova de estimadores;
- não vamos exigir 10 seeds em todo smoke test;
- não vamos tratar todo experimento didático como publicação.

O que **se aplica** é o hábito: pergunta, hipótese, métrica, baseline,
orçamento, seeds, validação/teste, incerteza e registro.

## Ferramentas do repositório

| Ferramenta | Papel |
| --- | --- |
| [`TEMPLATE.md`](TEMPLATE.md) | Registro com checklist de desenho empírico. |
| [`setups.md`](setups.md) | Protocolo e evolução dos setups. |
| [`../../.agents/skills/experimenter-helper/SKILL.md`](../../.agents/skills/experimenter-helper/SKILL.md) | Propõe o par `EXP-....md` + `EXP-....yaml` e valida o YAML. |
| [`../../arenas/common/report.py`](../../arenas/common/report.py) | IC, testes, efeito, correção e agregação por unidade. |
| [`../../scripts/bo_campaign.py`](../../scripts/bo_campaign.py) | BO com orçamento e validação final em seed separada. |
| [`../../scripts/asha_campaign.py`](../../scripts/asha_campaign.py) | Promoção/poda por rounds. |
| [`../../scripts/rigorous_validation.py`](../../scripts/rigorous_validation.py) | Treino multi-seed e análise por seed de treino. |
| [`../../docs/fila-de-experimentos.md`](../../docs/fila-de-experimentos.md) | Execução e artefatos na Fila de Experimentos. |

## Checklist rápido

Antes de rodar:

- [ ] A pergunta cabe em uma frase.
- [ ] A hipótese prevê direção/efeito.
- [ ] Existe um resultado que refutaria a hipótese.
- [ ] A métrica primária está definida.
- [ ] O baseline é forte e terá tuning comparável.
- [ ] Há 3+ seeds de treino.
- [ ] O orçamento é o mesmo (passos ou compute declarado).
- [ ] Tuning, validação e teste estão separados.
- [ ] O checkpoint será escolhido por validação.
- [ ] Os critérios de decisão estão escritos.

Depois de rodar:

- [ ] Todas as seeds e runs falhas estão no CSV.
- [ ] A análise primária é por seed de treino.
- [ ] Há IC, efeito e p corrigido.
- [ ] A conclusão não é mais forte que a evidência.
- [ ] O registro, o YAML e o `registro.md` concordam.
- [ ] Artefatos e comandos estão apontados.

## Leitura recomendada

- Patterson et al., *Empirical Design in Reinforcement Learning*, JMLR.
  [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)
- [`references/checklist.md`](../../.agents/skills/experimenter-helper/references/checklist.md)
- [`docs/experimentos/README.md`](README.md)

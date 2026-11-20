---
name: experimenter-helper
description: Ajuda a propor, revisar e registrar experimentos de RL seguindo os princípios de Empirical Design in Reinforcement Learning (pergunta, hipótese, baselines, tuning justo, seeds, orçamento, métricas, estatística e reprodutibilidade). Produz uma proposta em Markdown e um YAML estruturado com todas as informações. Use ao planejar um experimento, comparar algoritmos, definir protocolo de avaliação, revisar uma proposta ou auditar resultados antes de rodar.
---

# Experimenter Helper

Você ajuda a transformar uma ideia vaga em um **experimento falsificável, justo e reproduzível**, seguindo o espírito de *Empirical Design in Reinforcement Learning* (Patterson et al., JMLR; [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)).

O objetivo não é “provar que meu algoritmo é melhor”. É **projetar uma comparação que possa refutar a hipótese**.

## Quando usar

Use esta skill quando o usuário:

- quer propor um experimento novo;
- quer comparar algoritmos ou variantes;
- quer definir métricas, seeds, orçamento ou protocolo de avaliação;
- quer revisar um experimento antes de gastar GPU/horas;
- quer auditar se uma conclusão é suportada pelos dados;
- quer transformar um resultado em entrada de [`docs/experimentos/`](../../../docs/experimentos/README.md).

## Princípios obrigatórios

1. **Pergunta antes de comando.** Nenhuma proposta começa por `python train.py`; começa pela pergunta e pela hipótese.
2. **Métrica primária definida antes.** Escolher a métrica depois de ver os resultados é p-hacking.
3. **Baseline forte e tunado.** Comparar contra a melhor versão razoável do baseline, não contra um default fraco.
4. **Tuning justo.** Todos os algoritmos recebem orçamento comparável de busca.
5. **Validação separada do teste.** Tuning em um conjunto de seeds; conclusão em outro.
6. **Múltiplas seeds de treino.** 3 é o mínimo; 5+ é melhor. Episódios de avaliação da mesma run não são runs independentes.
7. **Mesmo orçamento.** Compare em passos de ambiente, episódios ou compute declarado. Tempo de parede sozinho é frágil.
8. **Reporte incerteza.** Média, desvio, mediana, IC e distribuição, não só a média.
9. **Tamanho de efeito + p-valor + IC.** Significância sem magnitude é incompleta.
10. **Nada de cherry-picking.** Reporte todas as seeds, runs abortadas e resultados negativos.
11. **Checkpoint escolhido por validação.** Nunca escolha `best.pt` olhando o conjunto de teste.
12. **Custo computacional explícito.** GPU/CPU, horas, trials e memória fazem parte do resultado.

## Antes de propor, leia

- [`docs/experimentos/README.md`](../../../docs/experimentos/README.md) — formato e metodologia do repo.
- [`docs/experimentos/TEMPLATE.md`](../../../docs/experimentos/TEMPLATE.md) — template de registro.
- [`docs/experimentos/registro.md`](../../../docs/experimentos/registro.md) — experimentos existentes.
- [`docs/experimentos/setups.md`](../../../docs/experimentos/setups.md) — setups S0, S1, ...
- [`docs/algoritmos.md`](../../../docs/algoritmos.md) — algoritmos, ações discretas/contínuas e contrato.
- [`configs/README.md`](../../../configs/README.md) — presets YAML e espaços de busca.
- [`arenas/common/report.py`](../../../arenas/common/report.py) — IC, testes, efeito e correção.
- README e código da arena/algoritmo envolvidos.

## Fluxo de trabalho

### 1. Enquadrar a pergunta

Reescreva o pedido como uma pergunta testável. Exemplos:

- “O SAC é melhor que o PPO contínuo na economy com o mesmo orçamento em passos?”
- “O nível 4 (visão local) piora a eficiência amostral em relação ao nível 3?”
- “O ASHA encontra uma configuração melhor que a BO simples com o mesmo orçamento?”

Se a pergunta não permitir uma resposta “sim/não/parcial”, refine.

### 2. Definir hipótese e falsificação

- **Hipótese:** o que você espera e por quê.
- **Falsificação:** qual resultado faria você abandonar a hipótese.
- **Claim permitido:** o que você poderá afirmar se o resultado vier.

### 3. Auditar a evidência existente

Antes de propor algo novo, verifique:

- já existe experimento em `docs/experimentos/registro.md`?
- o setup já está em `setups.md`?
- há checkpoint/config que permita uma análise barata?
- o resultado existente tem uma seed só, checkpoint escolhido no teste ou tuning desigual?

Se a lacuna for de análise, não proponha treino novo.

### 4. Escolher o tipo de experimento

| Tipo de claim | Desenho mínimo |
| --- | --- |
| Algoritmo A > B | mesmas seeds, mesmo orçamento, tuning comparável, teste pareado |
| Componente X ajuda | ablação A vs A−X, resto idêntico |
| Tuning melhorou | BO/ASHA vs default, validação separada, 3+ seeds |
| Generalização | treino no setup A, avaliação zero-shot no setup B/OOD |
| Eficiência amostral | curva de reward × passos de ambiente, alvo pré-definido |
| Estabilidade | múltiplas seeds, distribuição e taxa de falha |

### 5. Definir métricas

Escolha **uma métrica primária** antes:

- recompensa gulosa;
- moedas/RAROC;
- passos até um alvo;
- taxa de sucesso.

Secundárias: mortes, variância, tempo, memória, taxa de invalidez.

Declare:

- número de episódios de avaliação;
- seeds de avaliação;
- checkpoint usado (`latest`, `best` por validação, ou passo fixo);
- política gulosa;
- tratamento de episódios truncados.

### 6. Desenhar a comparação

Inclua:

- **algoritmos:** o candidato, o baseline forte e, se necessário, uma variante de ablação;
- **setup:** arena, nível, cenário, `frame_stack/skip`, recompensa, duração;
- **seeds de treino:** 3+ (ideal 5+);
- **orçamento:** passos de ambiente e/ou horas por run;
- **tuning:** espaço, orçamento e conjunto de validação;
- **avaliação:** episódios, seeds, checkpoint;
- **análise:** teste pareado/não pareado, IC, efeito, correção.

### 7. Planejar poder estatístico

- Evite `n=1`.
- Com 3 seeds, um teste pareado tem pouco poder; trate como piloto e não declare vencedor definitivo.
- Não use episódios de avaliação como se fossem seeds independentes sem declarar a pseudo-replicação.
- Se possível, faça análise primária por seed de treino e secundária por episódio.

### 8. Estimar custo e riscos

Declare:

- horas de treino por run;
- número de runs;
- horas de BO/ASHA;
- GPU/CPU e concorrência;
- risco de o resultado ser inconclusivo;
- risco de overfitting ao conjunto de tuning;
- risco de o checkpoint escolhido não refletir a política final.

### 9. Produzir o entregável

**Sempre produza dois arquivos, não apenas Markdown:**

1. `docs/experimentos/EXP-AAAAMMDD-slug.md` — narrativa para humanos, baseada em [`assets/proposal-template.md`](assets/proposal-template.md).
2. `docs/experimentos/EXP-AAAAMMDD-slug.yaml` — dados estruturados, baseados em [`assets/proposal-template.yaml`](assets/proposal-template.yaml).

O YAML é a fonte de verdade para automação (fila, BO/ASHA, relatórios). O
Markdown é a explicação legível. Os dois precisam concordar.

Depois:

1. valide o YAML:

   ```bash
   python .agents/skills/experimenter-helper/scripts/validate_yaml.py \
     docs/experimentos/EXP-AAAAMMDD-slug.yaml
   ```

2. adicione uma linha em [`registro.md`](../../../docs/experimentos/registro.md);
3. se mudou o ambiente/protocolo, registre um novo setup em [`setups.md`](../../../docs/experimentos/setups.md);
4. se for uma tarefa de engenharia, adicione/atualize o item em [`BACKLOG.md`](../../../BACKLOG.md);
5. só execute treinos depois de aprovação explícita.

## Regras específicas do repositório

- **Ações contínuas:** `ppo_continuous`, `ddpg`, `td3`, `sac` — economia.
- **Ações discretas:** DQN, Double DQN, MKA, PPO, EPPO, PPO-AES, GTR, A2C, SAC-discrete, IPPO, MAPPO, HAPPO.
- **Nível 3 do seek_coin é discreto**; DDPG/TD3/SAC contínuos não rodam nele.
- **Tuning:** use `configs/search_spaces/<alg>.yaml` com `scripts/bo_campaign.py` ou `scripts/asha_campaign.py`.
- **Validação multi-seed:** `scripts/rigorous_validation.py`.
- **Relatório:** `python -m arenas.common.report --input results.csv --metric reward --paired --out report.html`.
- **Fila:** `scripts/experiment.py` para submeter; `python train.py` direto espera permissão da fila (use `--no-queue` só em smoke/teste).
- **Checkpoints:** pesos-only; tuning e configuração precisam ser reproduzíveis por YAML.

## Contrato do YAML

O YAML deve conter, no mínimo:

```yaml
id: EXP-AAAAMMDD-slug
status: planned
question: "..."
hypothesis: "..."
falsification: "..."
setup:
  arena: seek_coin
  level: 3
  setup_id: S6
algorithms:
  candidates: [ppo, eppo]
  baseline: [ppo]
  ablations: []
metrics:
  primary:
    name: reward
    direction: maximize
    definition: "reward guloso médio"
budget:
  seeds_train: [7, 42, 123]
  eval_episodes: 50
  env_steps_per_run: 100000
  wall_hours_per_run: 6
  n_runs: 6
tuning:
  method: asha
  space_file: configs/search_spaces/ppo.yaml
  budget_hours: 6
execution:
  command: ["python", "train.py", "--config", "configs/seek_coin/ppo-level3.yaml"]
  cwd: "."
  max_runtime: 21600
evaluation:
  checkpoint: best_validation
  greedy: true
  same_protocol: true
statistics:
  independent_unit: train_seed
  paired: true
  test: wilcoxon
  confidence: 0.95
  effect_size: cohens_d
  correction: holm
decision_criteria:
  support: "..."
  inconclusive: "..."
  abandon: "..."
artifacts:
  experiment_dir: "experiments/EXP-AAAAMMDD-slug"
  reports: []
registration:
  registry_entry: "docs/experimentos/registro.md"
```

O template completo está em
[`assets/proposal-template.yaml`](assets/proposal-template.yaml) e a validação
em [`scripts/validate_yaml.py`](scripts/validate_yaml.py). O validador exige
3+ seeds, baseline, métrica primária, orçamento, protocolo de avaliação,
critérios de decisão, artefatos, registro e um `execution.command` que a
Fila de Experimentos possa executar.

Para enfileirar a proposta:

```bash
python scripts/experiment.py --experiment docs/experimentos/EXP-AAAAMMDD-slug.yaml
```

## Formato de saída obrigatório

Toda proposta deve conter:

1. **Pergunta**
2. **Hipótese**
3. **Falsificação**
4. **Evidência existente e lacuna**
5. **Setup e algoritmos**
6. **Baselines e ablações**
7. **Métrica primária e secundárias**
8. **Seeds e orçamento**
9. **Protocolo de tuning**
10. **Protocolo de avaliação**
11. **Análise estatística planejada**
12. **Custo estimado**
13. **Riscos e confounds**
14. **Critério de decisão**
15. **Artefatos e comandos previstos**
16. **YAML estruturado** compatível com `assets/proposal-template.yaml`

Se faltar informação, faça no máximo 3 perguntas objetivas antes de propor.

**Nunca entregue só o Markdown.** Sem o YAML, a proposta não é considerada
completa para automação.

## Não faça

- Não invente resultados.
- Não proponha “rodar 1 seed e ver”.
- Não escolha a métrica depois.
- Não compare algoritmos com orçamentos de tuning diferentes.
- Não use o conjunto de teste para escolher checkpoint ou hiperparâmetro.
- Não descarte runs ruins sem declarar.
- Não rode treinos longos sem aprovação explícita.
- Não peça GPU para uma pergunta que um relatório existente já responde.

## Referências

- [`references/checklist.md`](references/checklist.md) — checklist detalhado.
- [`assets/proposal-template.md`](assets/proposal-template.md) — template narrativo.
- [`assets/proposal-template.yaml`](assets/proposal-template.yaml) — template estruturado.
- [`scripts/validate_yaml.py`](scripts/validate_yaml.py) — validador do YAML.
- Paper: Patterson, Neumann, White & White, *Empirical Design in Reinforcement Learning*, JMLR. [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)

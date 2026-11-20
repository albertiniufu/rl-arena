# Proposta de experimento — EXP-AAAAMMDD-slug

> **Obrigatório:** gere também `EXP-AAAAMMDD-slug.yaml` a partir de
> `assets/proposal-template.yaml` e valide com `scripts/validate_yaml.py`.
>
> Status: planejado
> Autor:
> Data:
> Arena / setup:
> Algoritmos:

## 1. Pergunta

Uma frase testável.

## 2. Hipótese

O que se espera e por quê.

## 3. Falsificação

Qual resultado faria a hipótese ser abandonada.

## 4. Evidência existente e lacuna

- Experimentos relacionados:
- O que já sabemos:
- O que falta:
- Por que um novo treino é necessário (e não só uma análise):

## 5. Setup e algoritmos

- Arena:
- Nível / cenário:
- Setup Sx:
- Algoritmos candidatos:
- Baseline:
- Ablações:
- Configs YAML:

## 6. Métricas

- **Primária:**
- Secundárias:
- Definição operacional:
- Alvo de eficiência amostral (se houver):

## 7. Seeds e orçamento

- Seeds de treino:
- Seeds de avaliação:
- Episódios de avaliação:
- Passos de ambiente por run:
- Horas por run:
- Número de runs:
- GPU/CPU:

## 8. Tuning

- Espaço de busca:
- Orçamento (BO/ASHA/trials):
- Conjunto de validação:
- Como o melhor config será escolhido:
- Como será reproduzido:

## 9. Avaliação

- Checkpoint usado (`latest`, `best` por validação, passo fixo):
- Política gulosa:
- Tratamento de truncamento:
- Mesmo protocolo para todos os braços:

## 10. Análise estatística planejada

- Unidade independente:
- Teste pareado/não pareado:
- IC:
- Tamanho de efeito:
- Correção de múltiplas comparações:
- Análise primária:
- Análise secundária:
- Resultados negativos/falhas:

## 11. Custo estimado

| Item | Estimativa |
| --- | ---: |
| Runs de treino |  |
| Horas de treino |  |
| Horas de tuning |  |
| Horas de avaliação |  |
| Total |  |

## 12. Riscos e confounds

- Risco de variância:
- Risco de overfitting ao tuning:
- Risco de concorrência de GPU:
- Risco de checkpoint enganoso:
- Risco de conclusão inconclusiva:

## 13. Critério de decisão

Declare antes de rodar:

- “Se X, então considero a hipótese suportada.”
- “Se Y, então repito/abandono.”
- “Se Z, então o resultado é inconclusivo.”

## 14. Comandos previstos

```bash
# treino
python train.py --config ...

# tuning
python scripts/bo_campaign.py ...
python scripts/asha_campaign.py ...

# avaliação
python -m arenas.common.report --input ... --metric ... --paired --out ...
```

## 15. Execução na Fila de Experimentos

```yaml
execution:
  command: ["python", "train.py", "--config", "configs/...yaml"]
  cwd: "."
  max_runtime: 21600
  priority: 0
```

Para enfileirar:

```bash
python scripts/experiment.py --experiment docs/experimentos/EXP-AAAAMMDD-slug.yaml
```

## 16. Artefatos

- Checkpoints:
- CSVs:
- Relatórios:
- Pasta de experimento:
- Entrada no `registro.md`:

## 17. Checklist final

- [ ] Pergunta, hipótese e falsificação.
- [ ] Baseline forte.
- [ ] Mesmo orçamento.
- [ ] Tuning igualitário.
- [ ] 3+ seeds.
- [ ] Métrica primária antes.
- [ ] Checkpoint por validação.
- [ ] IC, teste e efeito.
- [ ] Custo declarado.
- [ ] Registro em `docs/experimentos/`.

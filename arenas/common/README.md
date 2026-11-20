# Infraestrutura comum

Peças reutilizáveis por qualquer arena. Nada aqui conhece as regras de uma
arena específica.

| Arquivo | Papel |
| --- | --- |
| `agent_factory.py` | `build_agent(alg, obs_shape, action_space)` valida o tipo de ação e instancia o agente. |
| `trainer.py` | `train_agent(...)`: loop de episódios, CSV, checkpoints, TensorBoard, orçamento de tempo e vídeo. |
| `evaluator.py` | `evaluate_agent(...)`: episódios gulosos, sem update, com métricas agregadas. |
| `checkpoint.py` | `agent_from_checkpoint(...)`: reconstrói o agente a partir do `.pt`. |
| `report.py` | Relatório estatístico e HTML interativo de experimentos (descritivas, IC, testes, efeito e correção). |
| `yaml_config.py` | Carrega presets YAML com precedência CLI > YAML > defaults. |
| `report_html.py` | Template da interface HTML usada por `report.py`. |
| `algorithm_catalog.py` | Lista os algoritmos aceitos por cada arena (`--alg`) com links para aulas, catálogo e README. |
| `experiment_analysis.py` | Descobre CSVs/JSONL de um experimento, infere a unidade independente e gera o relatório via `report.py`. |
| `experiment_proposal.py` | Gera e valida propostas MD+YAML; mesma validação usada pela skill `experimenter-helper` e pela Fila. |

## Quando usar

Use `arenas/common/` quando a arena for **simples o bastante** para o loop
genérico: um ambiente Gymnasium, um agente do contrato `arenas.algs` e
métricas por episódio. É o caso de `risk` e `economy`.

Para analisar resultados de experimentos, use `report.py`:

```bash
python -m arenas.common.report --input results.csv --metric reward_mean \
  --paired --baseline dqn --out report.html
```

O formato de entrada e a metodologia estão em
[`docs/experimentos/README.md`](../../docs/experimentos/README.md).

O `seek_coin` tem necessidades próprias (currículo de oponentes,
multiagente, vetorização, sobreposição coleta/update), então mantém o seu
trainer em `arenas/seek_coin/train.py`. Essa é a única exceção intencional.

## Fluxo do `train_agent`

```text
para cada episódio:
    reset(seed=seed+ep)
    enquanto não terminar:
        act → step → buffer.add → learn (a cada learn_every)
    end_episode(info)          # opcional, on-policy
    grava CSV/TensorBoard
    salva best/latest          # quando melhora ou a cada save_every
finish_training()              # opcional, on-policy
salva latest.pt
```

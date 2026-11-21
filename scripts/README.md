# Scripts de apoio

| Script | Papel |
| --- | --- |
| `report.py` | Atalho para `python -m arenas.common.report`; gera Markdown, JSON ou HTML interativo e aceita `--aggregate` por unidade. |
| `bo_campaign.py` | BO com espaços/configs YAML, métrica de seleção e validação final em seed separada. Grava `trials.jsonl`, `best.json` e `campaign.json`, lidos pela aba **BO** da Fila. |
| `asha_campaign.py` | Campanha ASHA (rungs, promoção/poda) sobre os mesmos YAMLs; mesma saída (`trials.jsonl`, `results.csv`, `best.json`, `campaign.json`), também exibida na aba **BO**. |
| `rigorous_validation.py` | Validação multi-seed com presets fixos de config ou `best.json` de BO; análise primária por seed de treino. |
| `job_queue_server.py` | Servidor local da fila de experimentos + interface web. |
| `experiment.py` | Submete um experimento novo à fila; inicia o servidor se necessário. |
| `retomar_experimentos.sh` | Depois de um reboot: checa memória, sobe a Fila, despausa e dá `retry` no job pendente. Use `--status` para só inspecionar. Ver [`docs/experimentos/retomada-20261120.md`](../docs/experimentos/retomada-20261120.md). |

Exemplo:

```bash
python scripts/report.py --input experiments/EXP-.../results.csv \
  --metric reward_mean --paired --out experiments/EXP-.../report.html
```

A implementação e a metodologia ficam em
[`arenas/common/report.py`](../arenas/common/report.py) e
[`docs/experimentos/README.md`](../docs/experimentos/README.md).

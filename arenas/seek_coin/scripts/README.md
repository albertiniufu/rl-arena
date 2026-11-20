# Scripts do seek_coin

Scripts auxiliares da arena `seek_coin`. Eles **não** são necessários para
usar o projeto: são protocolos de comparação, filas de treino longo e
ferramentas de análise usadas durante o desenvolvimento.

Todos os scripts usam o Python do ambiente ativo. Para apontar outro
interpretador, defina `PY` (ou `PYTHON`, nos que aceitam esse nome):

```bash
PY=/caminho/do/venv/bin/python bash arenas/seek_coin/scripts/train_long.sh
```

## Protocolos reprodutíveis (comece por aqui)

| Script | O que faz |
| --- | --- |
| `compare_onpolicy.sh` | Treina PPO e A2C no mesmo protocolo (seed 7, 100 eps, `rule`, `frame_stack=4`, `frame_skip=3`, batch 64, `--budget-snapshots`) e avalia snapshots `ep050`/`ep100`/`best`. |
| `train_long.sh` | Treina cada algoritmo por `HOURS` horas (padrão 3) com `--max-hours` e avalia `best`/`latest`. |
| `summarize_all.py` | Lê os CSVs de uma pasta de comparação e gera `summary-all.txt`, `training_summary_all.csv`, `evaluation_summary_all.csv` e `comparacao-all.md`. |

```bash
PYTHON=python bash arenas/seek_coin/scripts/compare_onpolicy.sh
PYTHON=python bash arenas/seek_coin/scripts/train_long.sh
python arenas/seek_coin/scripts/summarize_all.py logs/compare
```

## Filas de treino longo

Scripts de uma execução específica (madrugada, deadline, retomada). Eles
esperam o treino anterior terminar e encadeiam PPO/IPPO/MAPPO/HAPPO.
Os deadlines e horas padrão são os do experimento original; ajuste pelas
variáveis de ambiente indicadas no topo de cada arquivo.

- `train_level1_queue.sh` — nível 1, 4 h por algoritmo.
- `train_eppo_level2_queue.sh` — EPPO no nível 2.
- `train_aes_gtr_level2_queue.sh` — PPO-AES e GTR no nível 2.
- `resume_overnight_l2.sh` — retoma o bloco noturno do nível 2.
- `run_overnight_ppo_ippo.sh` — PPO → IPPO até um deadline.
- `run_long_until_1900.sh` / `run_equal_until_0800.sh` — dividem o tempo
  restante entre os algoritmos até o deadline.

## Duelos e vídeos

| Script | O que faz |
| --- | --- |
| `duel_multiagent.py` | Dois checkpoints multiagentes (ex.: MAPPO × IPPO) jogam um contra o outro e gravam MP4. |
| `duel_ppo_ippo.py` / `duel_ppo_happo.py` | Duelo entre um checkpoint single-agent e um multiagente no nível 1. |
| `record_multiagent.py` | Grava um episódio guloso de um checkpoint multiagente. |
| `watch_hourly_videos.sh` | Watcher que grava um vídeo por hora do treino noturno. |

## Performance e GPU

| Script | O que faz |
| --- | --- |
| `benchmark_gpu_env.py` | Compara CPU × GPU em `--mode physics` e `--mode full`. |
| `train_gpu_ppo.py` | Treinador PPO vetorizado em GPU com `GpuArenaVecEnv`. |

```bash
python arenas/seek_coin/scripts/benchmark_gpu_env.py --mode physics --envs 64,256,1024
python arenas/seek_coin/scripts/train_gpu_ppo.py --envs 128 --updates 100 --obs-dtype uint8 --amp
```

## Saídas

- Checkpoints: `checkpoints/<tag>/...` (ignorado pelo git)
- Logs, CSVs e TensorBoard: `logs/<tag>/...` (ignorado pelo git)
- Vídeos: `checkpoints/<tag>/videos/` ou `videos/` (ignorado pelo git)

Para o protocolo de um experimento que você quer versionar, escreva um
`README.md`/`protocol.md` dentro da pasta do experimento **fora** de
`experiments/` (que é ignorado) ou em `docs/`.

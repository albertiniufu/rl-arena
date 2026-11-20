# Scripts do risk

Scripts auxiliares da arena `risk`. Nenhum deles é necessário para treinar:
são receitas de execução longa usadas durante o desenvolvimento.

## `train_risk_night.sh`

Treino noturno de DQN no Risk com **máscara de ações**, TensorBoard, vídeos
periódicos e parada limpa em um deadline. O script:

- retoma de `$CKPT/latest.pt` se ele existir;
- roda em 4 jogadores, 20 exércitos iniciais e até 200 turnos;
- usa `--save-every-minutes 15` e `--video-every-minutes 60`;
- tenta de novo (até 20 vezes) se o processo cair, esperando 30 s;
- grava tudo em `$LOGS/run.log`.

### Uso

```bash
# da raiz do repositório, com o venv ativo
bash arenas/risk/scripts/train_risk_night.sh

# apontando outro interpretador e outro deadline
PY=.venv/bin/python \
DEADLINE="2026-12-01 07:00" \
bash arenas/risk/scripts/train_risk_night.sh
```

### Variáveis de ambiente

| Variável | Padrão | Papel |
| --- | --- | --- |
| `PY` | `python` | Interpretador do ambiente ativo. |
| `CKPT` | `checkpoints/risk_world` | Checkpoints, vídeos e `latest.pt` de retomada. |
| `LOGS` | `logs/risk_world` | `run.log` e runs do TensorBoard. |
| `DEADLINE` | data do experimento original | Encerra de forma limpa em `"YYYY-MM-DD HH:MM"`. |

### Saídas

```text
checkpoints/risk_world/
├── best.pt
├── latest.pt
└── videos/*.mp4
logs/risk_world/
├── run.log
├── training.csv
└── runs/                 # TensorBoard: tensorboard --logdir logs/risk_world/runs
```

Todos os caminhos são ignorados pelo git. Ajuste o `DEADLINE` padrão antes
de reutilizar o script: ele foi escrito para uma execução específica.

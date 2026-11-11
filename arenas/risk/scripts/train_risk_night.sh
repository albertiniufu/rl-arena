#!/usr/bin/env bash
# Treino noturno da arena risk: DQN com máscara de ações, TensorBoard,
# vídeos a cada hora e parada limpa no deadline.
#
# Uso:
#   bash arenas/risk/scripts/train_risk_night.sh
#   DEADLINE="2026-11-12 07:00" bash arenas/risk/scripts/train_risk_night.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-$HOME/venvs/main/bin/python}"
CKPT="${CKPT:-checkpoints/risk_night}"
LOGS="${LOGS:-logs/risk_night}"
DEADLINE="${DEADLINE:-2026-11-12 07:00}"
mkdir -p "$CKPT/videos" "$LOGS"
echo "=== inicio $(date '+%F %T') deadline=$DEADLINE ===" >> "$LOGS/run.log"

for attempt in $(seq 1 20); do
    if [ -f "$CKPT/latest.pt" ]; then
        RESUME_ARG="--resume $CKPT/latest.pt"
    else
        RESUME_ARG=""
    fi
    echo "=== tentativa $attempt $(date '+%F %T') ===" >> "$LOGS/run.log"
    PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR "$PY" train.py \
        --arena risk --alg dqn --episodes 1000000 \
        --players 2 --armies 20 --max-turns 200 --seed 7 \
        --lr 0.001 --batch-size 64 --learn-every 4 \
        --warmup 1000 --epsilon-min 0.1 --epsilon-ramp-episodes 5000 \
        --save-every 0 --save-every-minutes 15 \
        --checkpoint-dir "$CKPT" --log-dir "$LOGS" \
        --tensorboard --tensorboard-dir "$LOGS/runs" \
        --run-name risk_dqn_night \
        --video-every-minutes 60 --video-dir "$CKPT/videos" \
        --video-fps 4 \
        --deadline "$DEADLINE" \
        $RESUME_ARG >> "$LOGS/run.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        echo "=== encerrado com sucesso $(date '+%F %T') ===" >> "$LOGS/run.log"
        exit 0
    fi
    echo "=== saiu com codigo $code $(date '+%F %T'); retry em 30s ===" \
        >> "$LOGS/run.log"
    sleep 30
done

echo "=== tentativas esgotadas $(date '+%F %T') ===" >> "$LOGS/run.log"
exit 1

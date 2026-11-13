#!/usr/bin/env bash
# Blocos longos com tempo igual por algoritmo, terminando no deadline (19:00
# de hoje por padrão): PPO -> IPPO -> HAPPO -> MAPPO.
#
# Uso:
#   bash arenas/seek_coin/scripts/run_long_until_1900.sh
#   DEADLINE="2026-11-13 19:00:00" bash ...
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/venvs/main/bin/python}"
TAG="${TAG:-long1900}"
DEADLINE="${DEADLINE:-$(date '+%Y-%m-%d') 19:00:00}"
OUT="${OUT:-logs/$TAG}"
CKPT="${CKPT:-checkpoints/$TAG}"
mkdir -p "$OUT" "$CKPT" logs/runs

deadline_epoch=$(date -d "$DEADLINE" +%s)
now_epoch=$(date +%s)
remaining=$((deadline_epoch - now_epoch))
if [ "$remaining" -le 0 ]; then
    echo "deadline ja passou; nada a fazer" >> "$OUT/queue.log"
    exit 0
fi
per_hours=$("$PY" -c "print($remaining / 4 / 3600)")
echo "=== inicio $(date '+%F %T') deadline=$DEADLINE per=${per_hours}h ===" \
    >> "$OUT/queue.log"

run() {
    local name="$1"
    shift
    echo "=== $name inicio $(date '+%F %T') (${per_hours}h) ===" \
        >> "$OUT/queue.log"
    PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR "$PY" train.py \
        --arena seek_coin --level 1 \
        --max-hours "$per_hours" --episodes 1000000 \
        --frame-stack 4 --frame-skip 3 --seed 7 \
        --checkpoint-dir "$CKPT/$name" \
        --tensorboard --run-name "${TAG}_$name" --stats \
        "$@" >> "$OUT/$name.log" 2>&1
    local code=$?
    echo "=== $name fim code=$code $(date '+%F %T') ===" \
        >> "$OUT/queue.log"
}

run ppo --alg ppo --opponent rule
run ippo --alg ippo --agents 4 --scenario all-vs-all
run happo --alg happo --agents 4 --scenario all-vs-all
run mappo --alg mappo --agents 4 --scenario all-vs-all

echo "=== fila concluida $(date '+%F %T') ===" >> "$OUT/queue.log"

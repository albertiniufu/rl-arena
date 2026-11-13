#!/usr/bin/env bash
# Comparativo curto: 15 minutos por algoritmo (PPO -> IPPO -> HAPPO -> MAPPO).
# TensorBoard em logs/runs/compare15_<alg>; checkpoints em
# checkpoints/compare15/<alg>; logs em logs/compare15/.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/venvs/main/bin/python}"
OUT="${OUT:-logs/compare15}"
CKPT="${CKPT:-checkpoints/compare15}"
mkdir -p "$OUT" "$CKPT" logs/runs

run() {
    local name="$1"
    shift
    echo "=== $name inicio $(date '+%F %T') ===" >> "$OUT/queue.log"
    PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR "$PY" train.py \
        --arena seek_coin --level 1 \
        --max-hours 0.25 --episodes 1000000 \
        --frame-stack 4 --frame-skip 3 --seed 7 \
        --checkpoint-dir "$CKPT/$name" \
        --tensorboard --run-name "compare15_$name" --stats \
        "$@" >> "$OUT/$name.log" 2>&1
    local code=$?
    echo "=== $name fim code=$code $(date '+%F %T') ===" >> "$OUT/queue.log"
}

run ppo --alg ppo --opponent rule
run ippo --alg ippo --agents 4 --scenario all-vs-all
run happo --alg happo --agents 4 --scenario all-vs-all
run mappo --alg mappo --agents 4 --scenario all-vs-all

echo "=== fila concluida $(date '+%F %T') ===" >> "$OUT/queue.log"

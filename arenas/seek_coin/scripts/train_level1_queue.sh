#!/usr/bin/env bash
# Fila pós-risk: nível 1 (plataforma móvel) por 4 h em cada algoritmo.
# Espera o treino de risk terminar e roda PPO -> IPPO -> MAPPO -> HAPPO.
#
# Uso:
#   bash arenas/seek_coin/scripts/train_level1_queue.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-$HOME/venvs/main/bin/python}"
OUT="${OUT:-logs/level1}"
mkdir -p "$OUT"
echo "=== aguardando o risk terminar $(date '+%F %T') ===" >> "$OUT/queue.log"

# `[t]rain...` evita que o próprio pgrep case com este script.
while pgrep -f "[t]rain.py --arena risk" >/dev/null \
   || pgrep -f "[t]rain_risk_night.sh" >/dev/null; do
    sleep 60
done

echo "=== risk terminou; iniciando nivel 1 $(date '+%F %T') ===" \
    >> "$OUT/queue.log"

run() {
    local name="$1"
    shift
    echo "=== $name inicio $(date '+%F %T') ===" >> "$OUT/queue.log"
    PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR "$PY" train.py \
        --arena seek_coin --level 1 \
        --frame-stack 4 --frame-skip 3 \
        --batch-size 64 --seed 7 \
        --max-hours 4 --episodes 1000000 \
        --checkpoint-dir "checkpoints/level1/$name" \
        --tensorboard --run-name "level1_$name" --stats \
        "$@" >> "$OUT/$name.log" 2>&1
    local code=$?
    echo "=== $name fim code=$code $(date '+%F %T') ===" >> "$OUT/queue.log"
}

run ppo --alg ppo
run ippo --alg ippo --agents 4 --scenario all-vs-all
run mappo --alg mappo --agents 4 --scenario all-vs-all
run happo --alg happo --agents 4 --scenario all-vs-all

echo "=== fila concluida $(date '+%F %T') ===" >> "$OUT/queue.log"

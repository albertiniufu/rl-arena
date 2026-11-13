#!/usr/bin/env bash
# Treino noturno PPO -> IPPO até 2026-11-14 08:00.
# PPO e IPPO dividem o tempo restante em partes iguais; o watcher grava um
# vídeo por hora do algoritmo ativo em checkpoints/overnight/videos/.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/venvs/main/bin/python}"
DEADLINE="${DEADLINE:-2026-11-14 08:00:00}"
OUT="logs/overnight"
CKPT="checkpoints/overnight"
mkdir -p "$OUT" "$CKPT" logs/runs

deadline_epoch=$(date -d "$DEADLINE" +%s)
now_epoch=$(date +%s)
remaining=$((deadline_epoch - now_epoch))
if [ "$remaining" -le 0 ]; then
    echo "deadline ja passou" >> "$OUT/queue.log"
    exit 0
fi
first_hours=$("$PY" -c "print($remaining / 2 / 3600)")
echo "=== inicio $(date '+%F %T') deadline=$DEADLINE ppo=${first_hours}h ===" \
    >> "$OUT/queue.log"

# Watcher de vídeos horários.
setsid nohup bash "$ROOT/arenas/seek_coin/scripts/watch_hourly_videos.sh" \
    >> "$OUT/watcher.log" 2>&1 </dev/null &
echo $! > "$OUT/watcher.pid"

echo ppo > "$OUT/active"
echo "=== ppo inicio $(date '+%F %T') (${first_hours}h) ===" >> "$OUT/queue.log"
PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 "$PY" train.py --arena seek_coin --level 1 \
    --alg ppo --opponent rule \
    --max-hours "$first_hours" --episodes 1000000 \
    --frame-stack 4 --frame-skip 3 --seed 7 \
    --checkpoint-dir "$CKPT/ppo" \
    --tensorboard --run-name overnight_ppo --stats \
    >> "$OUT/ppo.log" 2>&1
ppo_code=$?
echo "=== ppo fim code=$ppo_code $(date '+%F %T') ===" >> "$OUT/queue.log"

remaining=$((deadline_epoch - $(date +%s)))
if [ "$remaining" -le 0 ]; then
    echo "=== sem tempo para IPPO ===" >> "$OUT/queue.log"
    echo done > "$OUT/active"
    exit 0
fi
second_hours=$("$PY" -c "print($remaining / 3600)")
echo ippo > "$OUT/active"
echo "=== ippo inicio $(date '+%F %T') (${second_hours}h) ===" >> "$OUT/queue.log"
PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 "$PY" train.py --arena seek_coin --level 1 \
    --alg ippo --agents 4 --scenario all-vs-all \
    --max-hours "$second_hours" --episodes 1000000 \
    --frame-stack 4 --frame-skip 3 --seed 7 \
    --checkpoint-dir "$CKPT/ippo" \
    --tensorboard --run-name overnight_ippo --stats \
    >> "$OUT/ippo.log" 2>&1
ippo_code=$?
echo "=== ippo fim code=$ippo_code $(date '+%F %T') ===" >> "$OUT/queue.log"
echo done > "$OUT/active"
echo "=== fila concluida $(date '+%F %T') ===" >> "$OUT/queue.log"

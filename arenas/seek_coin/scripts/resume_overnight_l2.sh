#!/usr/bin/env bash
# Retoma o bloco interrompido do nível 2: PPO a partir do latest.pt pelo
# tempo restante do bloco original (~1h25) e depois IPPO pelo bloco cheio
# (~5h25). O tempo não está mais preso às 08:00; o watcher para ao fim.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/venvs/main/bin/python}"
TAG="${TAG:-overnight_l2}"
LEVEL="${LEVEL:-2}"
PPO_HOURS="${PPO_HOURS:-1.4125}"       # ~1h24m45 restantes do bloco original
IPPO_HOURS="${IPPO_HOURS:-5.4208333}"  # bloco cheio do IPPO
OUT="logs/$TAG"
CKPT="checkpoints/$TAG"
mkdir -p "$OUT" "$CKPT" logs/runs

now_epoch=$(date +%s)
total_seconds=$("$PY" -c "print(int(($PPO_HOURS + $IPPO_HOURS) * 3600))")
deadline_epoch=$((now_epoch + total_seconds))
echo "=== retomada $(date '+%F %T') ppo=${PPO_HOURS}h ippo=${IPPO_HOURS}h ===" \
    >> "$OUT/queue.log"

TAG="$TAG" DEADLINE_EPOCH="$deadline_epoch" \
    setsid nohup bash "$ROOT/arenas/seek_coin/scripts/watch_hourly_videos.sh" \
    >> "$OUT/watcher.log" 2>&1 </dev/null &
echo $! > "$OUT/watcher.pid"

echo ppo > "$OUT/active"
echo "=== ppo retomada $(date '+%F %T') (${PPO_HOURS}h) ===" >> "$OUT/queue.log"
PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 "$PY" train.py --arena seek_coin --level "$LEVEL" \
    --alg ppo --opponent rule \
    --resume "$CKPT/ppo/latest.pt" \
    --max-hours "$PPO_HOURS" --episodes 1000000 \
    --frame-stack 4 --frame-skip 3 --seed 7 \
    --checkpoint-dir "$CKPT/ppo" \
    --tensorboard --run-name "${TAG}_ppo" --stats \
    >> "$OUT/ppo.log" 2>&1
ppo_code=$?
echo "=== ppo fim code=$ppo_code $(date '+%F %T') ===" >> "$OUT/queue.log"

echo ippo > "$OUT/active"
echo "=== ippo inicio $(date '+%F %T') (${IPPO_HOURS}h) ===" >> "$OUT/queue.log"
PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 "$PY" train.py --arena seek_coin --level "$LEVEL" \
    --alg ippo --agents 4 --scenario all-vs-all \
    --max-hours "$IPPO_HOURS" --episodes 1000000 \
    --frame-stack 4 --frame-skip 3 --seed 7 \
    --checkpoint-dir "$CKPT/ippo" \
    --tensorboard --run-name "${TAG}_ippo" --stats \
    >> "$OUT/ippo.log" 2>&1
ippo_code=$?
echo "=== ippo fim code=$ippo_code $(date '+%F %T') ===" >> "$OUT/queue.log"
echo done > "$OUT/active"
echo "=== fila concluida $(date '+%F %T') ===" >> "$OUT/queue.log"

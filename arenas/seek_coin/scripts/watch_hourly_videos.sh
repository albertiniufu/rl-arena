#!/usr/bin/env bash
# Watcher de vídeos horários do treino noturno PPO/IPPO.
# Lê logs/overnight/active (ppo|ippo|done) e grava um clipe do último
# checkpoint do algoritmo ativo a cada 1 hora, até o deadline.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/venvs/main/bin/python}"
TAG="${TAG:-overnight}"
OUT="checkpoints/$TAG/videos"
LOG="logs/$TAG/videos.log"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(date -d '2026-11-14 08:00:00' +%s)}"
mkdir -p "$OUT" "$(dirname "$LOG")"
echo "=== watcher inicio $(date '+%F %T') ===" >> "$LOG"

while true; do
    sleep 3600
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE_EPOCH" ]; then
        echo "=== watcher deadline $(date '+%F %T') ===" >> "$LOG"
        break
    fi
    active=$(cat "logs/$TAG/active" 2>/dev/null || echo ppo)
    if [ "$active" = "done" ]; then
        echo "=== watcher done $(date '+%F %T') ===" >> "$LOG"
        break
    fi
    ts=$(date '+%Y%m%d_%H%M')
    if [ "$active" = "ppo" ]; then
        ckpt="checkpoints/$TAG/ppo/latest.pt"
        out="$OUT/ppo_${ts}.mp4"
        for _ in 1 2 3; do
            if [ -f "$ckpt" ] && SDL_VIDEODRIVER=dummy \
                TORCH_CPP_LOG_LEVEL=ERROR "$PY" play.py \
                --arena seek_coin --checkpoint "$ckpt" \
                --opponent rule --episodes 1 --record "$out" \
                >> "$LOG" 2>&1; then
                break
            fi
            sleep 10
        done
    elif [ "$active" = "ippo" ]; then
        ckpt="checkpoints/$TAG/ippo/latest.pt"
        out="$OUT/ippo_${ts}.mp4"
        for _ in 1 2 3; do
            if [ -f "$ckpt" ] && SDL_VIDEODRIVER=dummy \
                TORCH_CPP_LOG_LEVEL=ERROR "$PY" \
                arenas/seek_coin/scripts/record_multiagent.py \
                --checkpoint "$ckpt" --record "$out" \
                >> "$LOG" 2>&1; then
                break
            fi
            sleep 10
        done
    else
        echo "estado desconhecido: $active" >> "$LOG"
    fi
    echo "=== $active $ts $(date '+%F %T') ===" >> "$LOG"
done

#!/usr/bin/env bash
# Fila EPPO no nível 2 (areia/gelo).
#
# Espera o bloco noturno atual de `overnight_l2` (PPO retomado + IPPO)
# terminar e então treina o EPPO pelo mesmo orçamento do bloco IPPO. O
# protocolo casa com o PPO do nível 2 (oponente `rule`, seed 7, frame
# stack 4, frame skip 3) e usa os hiperparâmetros do paper (lr 3e-4,
# minibatch 256).
#
# Uso:
#   bash arenas/seek_coin/scripts/train_eppo_level2_queue.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
TAG="${TAG:-overnight_l2}"
LEVEL="${LEVEL:-2}"
EPPO_HOURS="${EPPO_HOURS:-5.4208333}"   # bloco cheio do IPPO (~5h25)
OUT="logs/$TAG"
CKPT="checkpoints/$TAG"
mkdir -p "$OUT" "$CKPT" logs/runs

echo "=== eppo aguardando a fila atual $(date '+%F %T') ===" >> "$OUT/queue.log"
# A fila atual publica `done` em logs/<tag>/active ao terminar. Se ela não
# estiver mais rodando (por exemplo, interrompida), começa assim mesmo.
while :; do
    if [ -f "$OUT/active" ] && [ "$(cat "$OUT/active" 2>/dev/null)" = "done" ]; then
        break
    fi
    if ! pgrep -f "[t]rain.py --arena seek_coin --level $LEVEL" >/dev/null \
       && ! pgrep -f "[r]esume_overnight_l2.sh" >/dev/null; then
        break
    fi
    sleep 60
done

echo "=== eppo inicio $(date '+%F %T') (${EPPO_HOURS}h) ===" >> "$OUT/queue.log"
PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 "$PY" train.py --arena seek_coin --level "$LEVEL" \
    --alg eppo --opponent rule \
    --lr 3e-4 --batch-size 256 \
    --max-hours "$EPPO_HOURS" --episodes 1000000 \
    --frame-stack 4 --frame-skip 3 --seed 7 \
    --checkpoint-dir "$CKPT/eppo" \
    --tensorboard --run-name "${TAG}_eppo" --stats \
    >> "$OUT/eppo.log" 2>&1
code=$?
echo "=== eppo fim code=$code $(date '+%F %T') ===" >> "$OUT/queue.log"

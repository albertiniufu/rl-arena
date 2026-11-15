#!/usr/bin/env bash
# Fila pós-EPPO no nível 2: ppo_aes e depois GTR.
#
# Espera a fila do EPPO (`train_eppo_level2_queue.sh`, que por sua vez espera
# o IPPO atual) terminar e então treina os dois algoritmos, ~5h25 cada, no
# mesmo protocolo do bloco noturno (oponente `rule`, seed 7, fs=4/fk=3,
# lr 3e-4, minibatch 256).
#
# Uso:
#   bash arenas/seek_coin/scripts/train_aes_gtr_level2_queue.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PY="${PY:-$HOME/venvs/main/bin/python}"
TAG="${TAG:-overnight_l2}"
LEVEL="${LEVEL:-2}"
HOURS="${HOURS:-5.4208333}"   # bloco cheio de cada algoritmo (~5h25)
OUT="logs/$TAG"
CKPT="checkpoints/$TAG"
mkdir -p "$OUT" "$CKPT" logs/runs
echo $$ > "$OUT/aes_gtr_queue.pid"

echo "=== aes/gtr aguardando a fila do eppo $(date '+%F %T') ===" >> "$OUT/queue.log"
while pgrep -f "[t]rain_eppo_level2_queue.sh" >/dev/null \
   || pgrep -f "[t]rain.py --arena seek_coin --level $LEVEL" >/dev/null; do
    sleep 60
done

run() {
    local name="$1"
    echo "$name" > "$OUT/active"
    echo "=== $name inicio $(date '+%F %T') (${HOURS}h) ===" >> "$OUT/queue.log"
    PYTHONPATH="$ROOT" TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 "$PY" train.py --arena seek_coin --level "$LEVEL" \
        --alg "$name" --opponent rule \
        --lr 3e-4 --batch-size 256 \
        --max-hours "$HOURS" --episodes 1000000 \
        --frame-stack 4 --frame-skip 3 --seed 7 \
        --checkpoint-dir "$CKPT/$name" \
        --tensorboard --run-name "${TAG}_$name" --stats \
        >> "$OUT/$name.log" 2>&1
    local code=$?
    echo "=== $name fim code=$code $(date '+%F %T') ===" >> "$OUT/queue.log"
}

run ppo_aes
run gtr
echo done > "$OUT/active"
echo "=== fila aes/gtr concluida $(date '+%F %T') ===" >> "$OUT/queue.log"

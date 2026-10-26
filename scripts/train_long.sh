#!/usr/bin/env bash
# Treino longo por tempo de parede: cada algoritmo roda HOURS horas
# (padrão 3), no mesmo protocolo da comparação controlada, e é avaliado
# nos checkpoints best.pt e latest.pt ao final.
#
# Uso:
#   PYTHON=/caminho/venv/bin/python bash scripts/train_long.sh
#   HOURS=3 SEED=7 OPPONENT=rule bash scripts/train_long.sh
#
# Saídas:
#   checkpoints/long3h/<tag>/{best,latest}.pt
#   logs/long3h/train-<tag>.log
#   logs/long3h/training-<tag>.csv
#   logs/long3h/eval-<tag>-{best,latest}.log
#   logs/long3h/evaluation-<tag>-{best,latest}.csv
#   logs/runs/<alg>_configs_..._<HOURS>h  (TensorBoard)
#
# O script se recusa a misturar execuções: se OUT ou CKPT já existirem,
# encerra antes de treinar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
SEED="${SEED:-7}"
OPPONENT="${OPPONENT:-rule}"
HOURS="${HOURS:-3}"
OUT="${OUT:-logs/long3h}"
CKPT="${CKPT:-checkpoints/long3h}"

export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONUNBUFFERED=1

COMMON=(
  --seed "$SEED"
  --opponent "$OPPONENT"
  --frame-stack 4
  --frame-skip 3
  --batch-size 64
  --tensorboard
  --max-hours "$HOURS"
  --episodes 1000000
)

if [ -e "$OUT" ]; then
  echo "ERRO: $OUT já existe; remova ou use OUT=... para não misturar execuções" >&2
  exit 1
fi
if [ -e "$CKPT" ]; then
  echo "ERRO: $CKPT já existe; remova ou use CKPT=... para não misturar execuções" >&2
  exit 1
fi
mkdir -p "$OUT" "$CKPT" logs/runs

avaliar() {
  local alg_tag=$1 alg_label=$2 ckpt_file=$3 nome=$4
  echo "=== [$(date +%H:%M:%S)] AVALIAÇÃO $alg_label: $nome ==="
  "$PY" evaluate.py \
    --checkpoint "$ckpt_file" \
    --episodes 20 \
    --seed 1000 \
    --opponent "$OPPONENT" 2>&1 | tee "$OUT/eval-$alg_tag-$nome.log"
  mv logs/evaluation.csv "$OUT/evaluation-$alg_tag-$nome.csv"
}

run_alg() {
  local alg=$1 tag=$2 rotulo=$3 lr=$4
  local dir="$CKPT/$tag"
  local run_name="${alg}_configs_${OPPONENT}_fs4_fk3_lr${lr}_b64_seed${SEED}_${HOURS}h"

  if [ -e "logs/runs/$run_name" ]; then
    echo "ERRO: logs/runs/$run_name já existe; remova antes de repetir" >&2
    exit 1
  fi

  echo "=== [$(date +%H:%M:%S)] TREINO $rotulo: $alg (${HOURS}h, lr=$lr) ==="
  "$PY" train.py \
    --alg "$alg" \
    --checkpoint-dir "$dir" \
    --lr "$lr" \
    --run-name "$run_name" \
    "${COMMON[@]}" 2>&1 | tee "$OUT/train-$tag.log"

  if [ ! -f "$dir/best.pt" ] || [ ! -f "$dir/latest.pt" ]; then
    echo "ERRO: checkpoints de $tag não foram criados em $dir" >&2
    exit 1
  fi
  mv logs/training.csv "$OUT/training-$tag.csv"

  avaliar "$tag" "$rotulo" "$dir/best.pt" best
  avaliar "$tag" "$rotulo" "$dir/latest.pt" latest
}

# Mesmo protocolo da comparação: DQN/DDQN/PPO com lr=2.5e-4 e A2C com
# lr=7e-4 (valor usual para actor-critic com um único worker).
run_alg dqn dqn DQN 0.00025
run_alg doubledqn_per_dueling ddqn "Double DQN" 0.00025
run_alg ppo ppo PPO 0.00025
run_alg a2c a2c A2C 0.0007

echo "=== [$(date +%H:%M:%S)] TREINOS LONGOS CONCLUÍDOS ==="

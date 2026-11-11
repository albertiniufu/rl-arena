#!/usr/bin/env bash
# Comparação controlada dos algoritmos on-policy (PPO e A2C), no mesmo
# protocolo da comparação DQN/DDQN já existente. Roda depois que a fila
# anterior termina; não apaga nem move os checkpoints de DQN/Double DQN.
#
# Uso:
#   PYTHON=/caminho/venv/bin/python bash scripts/compare_onpolicy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
SEED=7
EPISODES=100
OPPONENT=rule
OUT=logs/compare

export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONUNBUFFERED=1

mkdir -p "$OUT" logs/runs checkpoints/compare

COMMON=(
  --seed "$SEED"
  --opponent "$OPPONENT"
  --frame-stack 4
  --frame-skip 3
  --batch-size 64
  --tensorboard
)

run_alg() {
  local alg=$1 ckpt=$2 tag=$3 lr=$4
  local run_name="${alg}_configs_${OPPONENT}_fs4_fk3_lr${lr}_b64_seed${SEED}"
  if [ -d "$ckpt" ]; then
    echo "ERRO: $ckpt já existe; remova ou use outro diretório para não misturar execuções" >&2
    exit 1
  fi
  if [ -d "logs/runs/$run_name" ]; then
    echo "ERRO: logs/runs/$run_name já existe; remova antes de repetir" >&2
    exit 1
  fi

  echo "=== [$(date +%H:%M:%S)] TREINO $tag: $alg (lr=$lr) ==="
  "$PY" train.py \
    --alg "$alg" \
    --episodes "$EPISODES" \
    --checkpoint-dir "$ckpt" \
    --lr "$lr" \
    --run-name "$run_name" \
    "${COMMON[@]}" 2>&1 | tee "$OUT/train-$tag.log"

  mv logs/training.csv "$OUT/training-$tag.csv"
  if [ ! -d "logs/runs/$run_name" ]; then
    echo "ERRO: logs/runs/$run_name não foi criado pelo treino $tag" >&2
    exit 1
  fi

  local pair target name
  for pair in "budget/ep050.pt:ep050" "budget/ep100.pt:ep100" "best.pt:best"; do
    target=${pair%%:*}
    name=${pair##*:}
    echo "=== [$(date +%H:%M:%S)] AVALIAÇÃO $tag: $target ==="
    "$PY" evaluate.py \
      --checkpoint "$ckpt/$target" \
      --episodes 20 \
      --seed 1000 \
      --opponent "$OPPONENT" 2>&1 | tee "$OUT/eval-$tag-$name.log"
    mv logs/evaluation.csv "$OUT/evaluation-$tag-$name.csv"
  done
}

# A2C usa um lr um pouco maior, como é usual para actor-critic com um único
# worker; o restante do protocolo é idêntico ao da comparação DQN/DDQN.
run_alg ppo checkpoints/compare/ppo ppo 0.00025
run_alg a2c checkpoints/compare/a2c a2c 0.0007

"$PY" scripts/summarize_all.py "$OUT" | tee "$OUT/summary-all.txt"
echo "=== [$(date +%H:%M:%S)] COMPARAÇÃO ON-POLICY CONCLUÍDA ==="

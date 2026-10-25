#!/usr/bin/env bash
# Gera dist/rl-arena-colab.zip com os arquivos rastreados do HEAD.
#
# O zip contém este diretório `colab/`, o notebook e o código necessário para
# treinar/avaliar no Colab (sem venv, checkpoints, logs ou demos).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! git diff --quiet HEAD -- colab; then
    echo "Há mudanças não commitadas em colab/; commite antes de gerar o zip." >&2
    exit 1
fi

mkdir -p dist
git archive --format=zip --prefix=rl-arena/ -o dist/rl-arena-colab.zip HEAD -- \
    agents arena colab alg_dqn alg_doubledqn_per_dueling alg_registry.py \
    tests train.py evaluate.py play.py requirements.txt README.md

echo "Gerado: $(pwd)/dist/rl-arena-colab.zip"

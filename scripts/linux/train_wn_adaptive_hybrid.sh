#!/usr/bin/env bash

# Training script for adaptive-hybrid DirectAU (learnable alpha over dual uniformity scales).
# Uses alignment + uniformity, no negative sampling.

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_adaptive_hybrid_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model distilbert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--valid-label-path "${DATA_DIR}/valid_w_label.txt" \
--task ${TASK} \
--batch-size 512 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 0 \
--finetune-t \
--loss-type alignment \
--use-uniformity-loss \
--adaptive-hybrid \
--directau-alpha 1 \
--directau-gamma 1 \
--directau-eps 1e-12 \
--directau-uniformity-scale-1 4 \
--directau-uniformity-scale-2 6 \
--no-negative-sampling \
--epochs 50 \
--workers 2 \
--max-to-keep 3 "$@"

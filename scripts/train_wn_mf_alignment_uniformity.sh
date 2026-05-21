#!/usr/bin/env bash

# Training script for MF encoder + Alignment + Uniformity loss (no InfoNCE)

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_mf_alignment_uniformity_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model distilbert-base-uncased \
--lr 5e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--valid-label-path "${DATA_DIR}/valid_w_label.txt" \
--task ${TASK} \
--batch-size 512 \
--print-freq 20 \
--use-amp \
--pre-batch 0 \
--finetune-t \
--loss-type alignment \
--use-uniformity-loss \
--directau-alpha 3.0 \
--directau-gamma 1.0 \
--directau-eps 1e-12 \
--no-negative-sampling \
--use-mf \
--mf-dim 256 \
--mf-init xavier \
--mf-dropout 0.0 \
--mf-head-mask \
--mf-head-mask-residual 1.0 \
--epochs 50 \
--workers 2 \
--max-to-keep 3 "$@"

#!/usr/bin/env bash

# Training script for Mode 011: InfoNCE + Negative Sampling + Uniformity Loss
# 
# All 8 supported training modes:
# Mode 000: --loss-type infonce (pairwise, no uniformity)
# Mode 001: --loss-type infonce --use-uniformity-loss
# Mode 010: --loss-type infonce --use-negative-sampling (standard InfoNCE)
# Mode 011: --loss-type infonce --use-negative-sampling --use-uniformity-loss (THIS SCRIPT)
# Mode 100: --loss-type alignment (pure alignment)
# Mode 101: --loss-type alignment --use-uniformity-loss (DirectAU traditional)
# Mode 110: --loss-type alignment --use-negative-sampling
# Mode 111: --loss-type alignment --use-negative-sampling --use-uniformity-loss

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_mode011_$(date +%F-%H%M.%S)"
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
--directau-gamma 1.0 \
--directau-eps 1e-12 \
--no-negative-sampling \
--epochs 50 \
--workers 2 \
--max-to-keep 3 "$@"
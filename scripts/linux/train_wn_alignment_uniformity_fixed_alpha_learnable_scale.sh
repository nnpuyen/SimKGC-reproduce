#!/usr/bin/env bash

# Training script: alignment loss with fixed DirectAU alpha and learnable uniformity scale
# Based on train_wn_alignment_uniformity_learnable_uniformity_scale.sh

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd ../.. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_mode011_fixed_alpha_learnable_scale_$(date +%F-%H%M.%S)"
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
--uniformity-on-query \
--uniformity-on-tail \
--directau-alpha 3 \
--directau-gamma 1 \
--directau-eps 1e-12 \
--directau-uniformity-scale 4 \
--learnable-directau-uniformity-scale \
--log-uniformity-lr 5e-5 \
--no-negative-sampling \
--epochs 50 \
--workers 2 \
--max-to-keep 3 "$@"

#!/usr/bin/env bash
# Train with DirectAU piecewise schedule (alpha and uniformity scale change by phase)
# Does NOT use linear schedule.

set -euo pipefail

DATA_DIR="data/WN18RR"
MODEL_DIR="checkpoint/WN18RR_directau_piecewise"
PRETRAINED="distilbert-base-uncased"
TASK="WN18RR"

mkdir -p ${MODEL_DIR}

python3 -u main.py \
  --model-dir "${MODEL_DIR}" \
  --pretrained-model ${PRETRAINED} \
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
  --directau-piecewise \
  --directau-piecewise-phases 15,15,20 \
  --directau-piecewise-values 4.5,5.5,6.5 \
  --directau-alpha 4.5 \
  --directau-gamma 1 \
  --directau-eps 1e-12 \
  --directau-uniformity-scale 4 \
  --no-negative-sampling \
  --epochs 50 \
  --workers 2 \
  --max-to-keep 3 "$@"

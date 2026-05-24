#!/usr/bin/env bash

# Resume training from a checkpoint path using alignment + uniformity (no negative sampling).
# Usage:
#   bash scripts/resume_wn_alignment_uniformity.sh <resume_path> <total_epochs> [output_dir]

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

RESUME_PATH="$1"
TOTAL_EPOCHS="$2"
OUTPUT_DIR="$3"

if [ -z "${RESUME_PATH}" ] || [ -z "${TOTAL_EPOCHS}" ]; then
  echo "Usage: $0 <resume_path> <total_epochs> [output_dir]"
  exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  if [ -d "${RESUME_PATH}" ]; then
    OUTPUT_DIR="${RESUME_PATH}"
  else
    OUTPUT_DIR="$( cd "$( dirname "${RESUME_PATH}" )" && pwd )"
  fi
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--resume \
--resume-path "${RESUME_PATH}" \
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
--directau-alpha 3.0 \
--directau-gamma 1.5 \
--directau-eps 1e-12 \
--directau-uniformity-scale 4 \
--no-negative-sampling \
--epochs "${TOTAL_EPOCHS}" \
--workers 2 \
--max-to-keep 3 "$@"

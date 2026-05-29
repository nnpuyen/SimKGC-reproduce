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
OUTPUT_DIR=""
EXTRA_ARGS=("${@:3}")

if [ -n "${EXTRA_ARGS[0]}" ] && [[ "${EXTRA_ARGS[0]}" != --* ]]; then
  OUTPUT_DIR="${EXTRA_ARGS[0]}"
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

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

RESUME_PATH="${RESUME_PATH}" python3 - <<'PY'
import os
import sys
import torch

ckt_path = os.environ.get('RESUME_PATH', '')
if not ckt_path:
  print('Resume path is empty; cannot read epoch')
  sys.exit(0)

try:
  ckt = torch.load(ckt_path, map_location='cpu')
  print('Checkpoint epoch =', ckt.get('epoch'))
except Exception as exc:
  print('Failed to read checkpoint epoch:', exc)
PY

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
--directau-alpha 1.0 \
--directau-gamma 1.0 \
--directau-eps 1e-12 \
--directau-uniformity-scale 5 \
--no-negative-sampling \
--epochs "${TOTAL_EPOCHS}" \
--workers 2 \
--max-to-keep 3 "${EXTRA_ARGS[@]}"

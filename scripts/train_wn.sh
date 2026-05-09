#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

choose_existing_file() {
    for path in "$@"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

TRAIN_PATH=$(choose_existing_file "${DATA_DIR}/train.txt.json" "${DATA_DIR}/train.txt")
VALID_PATH=$(choose_existing_file "${DATA_DIR}/valid.txt.json" "${DATA_DIR}/valid.txt" "${DATA_DIR}/valid_w_label.txt")

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model distilbert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${TRAIN_PATH}" \
--valid-path "${VALID_PATH}" \
--valid-label-path "${DATA_DIR}/valid_w_label.txt" \
--task ${TASK} \
--batch-size 512 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 0 \
--finetune-t \
--epochs 50 \
--workers 2 \
--max-to-keep 3 "$@"

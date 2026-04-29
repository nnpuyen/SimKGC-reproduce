#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

choose_existing_file() {
    for path in "$@"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

TRAIN_PATH="./data/${TASK}/train.txt"
VALID_PATH=$(choose_existing_file "./data/${TASK}/valid.txt" "./data/${TASK}/valid_w_label.txt")
TEST_PATH=$(choose_existing_file "./data/${TASK}/test.txt" "./data/${TASK}/test_w_label.txt")

python3 -u preprocess.py \
--task "${TASK}" \
--train-path "${TRAIN_PATH}" \
--valid-path "${VALID_PATH}" \
--test-path "${TEST_PATH}"

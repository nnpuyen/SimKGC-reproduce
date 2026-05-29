#!/usr/bin/env bash

set -x
set -e

model_path="bert"
task="wiki5m_trans"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

while [[ $# -ge 1 ]]; do
  case "$1" in
    --model_path|--model-path)
      model_path=$2
      shift 2
      ;;
    --task)
      task=$2
      shift 2
      ;;
    --test_path|--test-path)
      test_path=$2
      shift 2
      ;;
    --)
      shift
      break
      ;;
    --*)
      break
      ;;
    *)
      if [ "$model_path" = "bert" ]; then
        model_path=$1
      elif [ "$task" = "wiki5m_trans" ]; then
        task=$1
      elif [ -z "${test_path+x}" ]; then
        test_path=$1
      fi
      shift
      ;;
  esac
done

test_path="$DATA_DIR/test.txt.json"

neighbor_weight=0.05

python3 -u eval_wiki5m_trans.py \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "${test_path}" "$@"

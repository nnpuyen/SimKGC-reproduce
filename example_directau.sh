#!/bin/bash
# Example commands for training and evaluating with DirectAU

# Dataset: WN18RR
TASK="wn18rr"
TRAIN_PATH="data/WN18RR/train.txt"
VALID_PATH="data/WN18RR/valid.txt"
TEST_PATH="data/WN18RR/test.txt"
OUTPUT_DIR="./output_directau"

# Training with DirectAU (SimKGC encoder + DirectAU loss + strict normalization)
echo "=== Training with DirectAU ==="
python main.py \
    --task ${TASK} \
    --pretrained-model distilbert-base-uncased \
    --train-path ${TRAIN_PATH} \
    --valid-path ${VALID_PATH} \
    --model-dir ${OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 64 \
    --epochs 10 \
    --lr 2e-5 \
    --directau \
    --directau-gamma 1.0 \
    --directau-eps 1e-12 \
    --chunk-size 8192 \
    --eval-every-n-step 10000

# Evaluation with DirectAU
echo "=== Evaluation with DirectAU ==="
python evaluate.py \
    --task ${TASK} \
    --pretrained-model distilbert-base-uncased \
    --valid-path ${VALID_PATH} \
    --train-path ${TRAIN_PATH} \
    --eval-model-path ${OUTPUT_DIR}/checkpoint_best.mdl \
    --output-dir ${OUTPUT_DIR} \
    --directau \
    --chunk-size 8192

# Comparison: Training without DirectAU (baseline SimKGC)
echo "=== Training without DirectAU (baseline) ==="
python main.py \
    --task ${TASK} \
    --pretrained-model distilbert-base-uncased \
    --train-path ${TRAIN_PATH} \
    --valid-path ${VALID_PATH} \
    --model-dir ./output_simkgc \
    --output-dir ./output_simkgc \
    --batch-size 64 \
    --epochs 10 \
    --lr 2e-5

# Evaluation without DirectAU (baseline)
echo "=== Evaluation without DirectAU (baseline) ==="
python evaluate.py \
    --task ${TASK} \
    --pretrained-model distilbert-base-uncased \
    --valid-path ${VALID_PATH} \
    --train-path ${TRAIN_PATH} \
    --eval-model-path ./output_simkgc/checkpoint_best.mdl \
    --output-dir ./output_simkgc

## SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models

Official code repository for ACL 2022 paper 
"[SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models](https://aclanthology.org/2022.acl-long.295.pdf)".

The paper is available at [https://aclanthology.org/2022.acl-long.295.pdf](https://aclanthology.org/2022.acl-long.295.pdf).

In this paper,
we identify that one key issue for text-based knowledge graph completion is efficient contrastive learning.
By combining large number of negatives and hardness-aware InfoNCE loss,
SimKGC can substantially outperform existing methods on popular benchmark datasets.

## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with 4 V100(32GB) GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.

We also provide the predictions from our models in [predictions](predictions/) directory.

For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

### WN18RR dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh WN18RR
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```

Feel free to change the output directory to any path you think appropriate.

### FB15k-237 dataset

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh FB15k237
```

Step 2, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
```

### Wikidata5M transductive dataset

Step 0, download the dataset. 
We provide a script to download the [Wikidata5M dataset](https://deepgraphlearning.github.io/project/wikidata5m) from its official website.
This will download data for both transductive and inductive settings.
```
bash ./scripts/download_wikidata5m.sh
```

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh wiki5m_trans
```

Step 2, training the model and (optionally) specify the output directory (about 12 hours)
```
OUTPUT_DIR=./checkpoint/wiki5m_trans/ bash scripts/train_wiki.sh wiki5m_trans
```

Step 3, evaluate a trained model (it takes about 1 hour due to the large number of entities)
```
bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans/model_last.mdl
```

### Wikidata5M inductive dataset

Make sure you have run `scripts/download_wikidata5m.sh` to download Wikidata5M dataset.

Step 1, preprocess the dataset
```
bash scripts/preprocess.sh wiki5m_ind
```

Step 2, training the model and (optionally) specify the output directory (about 11 hours)
```
OUTPUT_DIR=./checkpoint/wiki5m_ind/ bash scripts/train_wiki.sh wiki5m_ind
```

Step 3, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wiki5m_ind/model_last.mdl wiki5m_ind
```

## Troubleshooting

1. I encountered "CUDA out of memory" when running the code.

We run experiments with 4 V100(32GB) GPUs, please reduce the batch size if you don't have enough resources. 
Be aware that smaller batch size will hurt the performance for contrastive training. 

2. Does this codebase support distributed data parallel(DDP) training?

No. Some input masks require access to batch data on all GPUs, 
so currently it only supports data parallel training for ease of implementation.

## DirectAU Integration

This codebase now includes an optional DirectAU mode that enhances SimKGC with:
- **Strict L2 normalization** on all encoder outputs for stable representation geometry
- **Alignment + Uniformity loss** replacing the InfoNCE loss for better representation learning (no negative sampling)
- **Chunked inference** for efficient evaluation on large entity sets
- **Keeps original pair-encoder architecture** — head+relation encoded together, tail encoded separately

Key design: DirectAU applies strict normalization + new loss to the same encoder architecture as SimKGC, without modifying how queries and targets are encoded.

### Quick Start with DirectAU

Training with DirectAU on WN18RR:
```bash
OUTPUT_DIR=./checkpoint/wn18rr_directau/ python main.py \
    --task wn18rr \
    --pretrained-model distilbert-base-uncased \
    --train-path data/WN18RR/train.txt \
    --valid-path data/WN18RR/valid.txt \
    --model-dir $OUTPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size 64 \
    --epochs 10 \
    --lr 2e-5 \
    --directau \
    --directau-gamma 1.0 \
    --directau-eps 1e-12 \
    --chunk-size 8192
```

Evaluating with DirectAU:
```bash
python evaluate.py \
    --task wn18rr \
    --pretrained-model distilbert-base-uncased \
    --valid-path data/WN18RR/valid.txt \
    --train-path data/WN18RR/train.txt \
    --eval-model-path $OUTPUT_DIR/checkpoint_best.mdl \
    --output-dir $OUTPUT_DIR \
    --directau \
    --chunk-size 8192
```

### Training Objective Configuration

SimKGC supports flexible training objectives through three independent flags:

**1. Loss Type** (`--loss-type`):
- `infonce` (default): Original InfoNCE contrastive loss
- `alignment`: DirectAU alignment loss (mean squared L2 distance)
- `bridge`: Bridged loss (alignment + cross-uniformity, negatives only)

**2. Negative Sampling** (`--use-negative-sampling`):
- `True` (default): Use in-batch negatives for contrastive learning
- `False`: Use only positive pairs (alignment or pairwise losses)

**3. Uniformity Loss** (`--use-uniformity-loss`):
- `False` (default): No regularization
- `True`: Add uniformity term to spread embeddings

**Eight Training Modes** (all combinations of 3 binary flags):

| Mode | Loss Type | Neg Sampling | Uniformity | Description |
|------|-----------|--------------|------------|-------------|
| **000** | infonce | ❌ | ❌ | Simple pairwise InfoNCE (no negatives, no uniformity) |
| **001** | infonce | ❌ | ✅ | Pairwise InfoNCE + uniformity regularization |
| **010** | infonce | ✅ | ❌ | Standard InfoNCE with in-batch negatives |
| **011** | infonce | ✅ | ✅ | InfoNCE + negatives + uniformity |
| **100** | alignment | ❌ | ❌ | Pure alignment loss (L2 distance only) |
| **101** | alignment | ❌ | ✅ | Alignment + uniformity (DirectAU traditional) |
| **110** | alignment | ✅ | ❌ | Alignment loss with in-batch negatives |
| **111** | alignment | ✅ | ✅ | Alignment + negatives + uniformity |

**Example Commands**:
```bash
# Mode 000: Pairwise InfoNCE (no negatives)
python main.py ... --loss-type infonce

# Mode 001: Pairwise InfoNCE + uniformity
python main.py ... --loss-type infonce --use-uniformity-loss

# Mode 010: Standard InfoNCE (with negatives)
python main.py ... --loss-type infonce --use-negative-sampling

# Mode 011: InfoNCE + negatives + uniformity
python main.py ... --loss-type infonce --use-negative-sampling --use-uniformity-loss

# Mode 100: Pure alignment
python main.py ... --loss-type alignment

# Mode 101: Alignment + uniformity (DirectAU)
python main.py ... --loss-type alignment --use-uniformity-loss

# Mode 110: Alignment with negatives
python main.py ... --loss-type alignment --use-negative-sampling

# Mode 111: Alignment + negatives + uniformity
python main.py ... --loss-type alignment --use-negative-sampling --use-uniformity-loss
```

**Legacy Compatibility**:
- `--directau`: Shorthand for `--loss-type alignment --use-uniformity-loss`

**Uniformity Configuration** (`--directau-gamma`, `--directau-eps`):
- `--directau-gamma`: Weight for uniformity term (default: 1.0)
- `--directau-eps`: Epsilon for numerical stability (default: 1e-12)

### Bridged Loss

The bridged objective combines alignment with a cross-uniformity term that pushes each query away from non-matching tails in the batch (positives are excluded from the denominator). This keeps the hypersphere geometry while using query-conditioned repulsion.

Key hyperparameters:
- `--bridge-alpha`: Weight for alignment term (default: 1.0)
- `--bridge-gamma`: Weight for cross-uniformity term (default: 1.0)
- `--bridge-beta`: Scale for squared distances inside the cross-uniformity term. If unset, it defaults to `1 / (2 * t)`.

Example:
```bash
python main.py ... --loss-type bridge --bridge-alpha 1.0 --bridge-gamma 1.0 --bridge-beta 10.0
```

### Key Differences from SimKGC

| Feature | SimKGC | DirectAU |
|---------|--------|----------|
| Query encoding | Pair encoder (head+relation together) | **Same** - pair encoder (head+relation together) |
| Tail encoding | Separate tail encoder | **Same** - separate tail encoder |
| Normalization | Implicit in dot-product | Explicit L2-norm after encoding |
| Loss function | InfoNCE with negative sampling | Alignment + Uniformity (no negative sampling) |
| Scoring | Dot product (descending) | Dot product (descending, on normalized vectors) |

### Testing DirectAU

Run the test suite to verify the DirectAU integration:
```bash
python test_directau.py
```

This tests initialization, loss computation, query construction, and chunked inference.

### Full Example

See `example_directau.sh` for complete examples including baseline comparison.

## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
@inproceedings{wang-etal-2022-simkgc,
    title = "{S}im{KGC}: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models",
    author = "Wang, Liang  and
      Zhao, Wei  and
      Wei, Zhuoyu  and
      Liu, Jingming",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.295",
    pages = "4281--4294",
}
```

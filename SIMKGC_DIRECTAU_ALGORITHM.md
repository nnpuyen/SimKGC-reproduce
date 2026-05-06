# SimKGC with DirectAU Loss: Algorithm

## Overview

**SimKGC-DirectAU** is a Knowledge Graph Completion model that combines:

1. **SimKGC Infrastructure**:
   - Pre-trained BERT encoder architecture (HR and Tail encoders)
   - Shared data pipeline and text-based entity representations
   - Link prediction and triple classification tasks

2. **DirectAU Loss Function**:
   - Alignment loss: minimizes L2 distance between matching (h,r,t) pairs
   - Uniformity loss: spreads embeddings uniformly on the hypersphere
   - No temperature scaling, no margin-based ranking
   - Explicit L2 normalization of all embeddings

The integration maintains SimKGC's **text-based encoder architecture** while replacing the contrastive loss with **DirectAU's alignment + uniformity approach** for more interpretable and theoretically motivated training.

### Eight Strategy Controls

In addition to the core loss-mode split, the implementation and documentation expose eight strategy switches that control data context, optimization, and memory behavior:

| # | Strategy | Primary use |
|---|---|---|
| 1 | Neighbor-based context augmentation | Enrich entity text with 1-hop graph neighbors |
| 2 | Triplet masking | Prevent in-batch leakage from known positives |
| 3 | Learning-rate scheduling | Apply linear or cosine warmup/decay |
| 4 | Gradient accumulation | Increase effective batch size under memory limits |
| 5 | Pre-batch negatives | Reuse cached vectors for InfoNCE-style training |
| 6 | Mixed precision | Reduce activation memory and improve throughput |
| 7 | Weight decay | Regularize model weights during optimization |
| 8 | Fine-tunable temperature | Make SimKGC temperature learnable when needed |

DirectAU primarily uses controls 1, 2, 3, 4, 6, and 7; controls 5 and 8 are SimKGC-oriented compatibility controls.

The three binary choices that define the eight modes are:

| Choice | Off | On |
|---|---|---|
| Loss family | InfoNCE | DirectAU |
| Context augmentation | Plain text only | `--use-link-graph` |
| Execution profile | Standard precision | `--use-amp` |

This gives $2^3 = 8$ concrete modes. The four InfoNCE modes keep the original SimKGC contrastive objective; the four DirectAU modes replace that objective with alignment + uniformity.

| Mode | Loss family | Link graph | AMP | Notes |
|---|---|---|---|---|
| 1 | InfoNCE | Off | Off | Base SimKGC training path |
| 2 | InfoNCE | Off | On | Same loss, lower precision for memory savings |
| 3 | InfoNCE | On | Off | Adds 1-hop neighbor context |
| 4 | InfoNCE | On | On | Graph context plus AMP |
| 5 | DirectAU | Off | Off | DirectAU baseline |
| 6 | DirectAU | Off | On | DirectAU with AMP |
| 7 | DirectAU | On | Off | DirectAU plus graph context |
| 8 | DirectAU | On | On | Full DirectAU configuration |

InfoNCE-specific knobs such as `--pre-batch`, `--additive-margin`, `--use-self-negative`, and `--finetune-t` remain orthogonal options that only matter when the loss family is InfoNCE.

---

## Architecture

### Text Encoding

```yaml
# Text Encoding
pretrained_model: "distilbert-base-uncased"
max_length: 128
pooling: "cls"              # CLS, mean, or max pooling

# Optional: Neighbor augmentation
use_link_graph: false       # If true: append neighbor entity names to descriptions
```

**With `--use-link-graph`**:
- Entity descriptions are enriched with 1-hop neighbors (max 10)
- Only added if description < 20 tokens (prevents over-tokenization)
- Example: "philosopher" → "philosopher Aristotle Plato Socrates"
- Prevents label leakage: excludes target entity during training

---

## Loss Functions

### 1. Alignment Loss

**Purpose**: Ensure matching (h, r, t) triples map close together in embedding space.

**Formula**:
$$L_{\text{align}} = \text{mean}\left(\| q - t \|_2^2\right)$$

Where:
- $q = \text{encode\_query}(h, r)$ (query embedding, L2-normalized)
- $t = \text{encode\_tail}(t)$ (tail embedding, L2-normalized)
- Both lie on the unit hypersphere

**Computation**:
```python
diff = q_batch - t_batch                    # [batch_size, 768]
squared_l2_dist = torch.sum(diff ** 2, dim=-1)  # [batch_size]
loss_align = torch.mean(squared_l2_dist)   # scalar
```

**Interpretation**: Minimizes squared Euclidean distance between matching pairs on the unit sphere. For L2-normalized vectors, this is equivalent to minimizing $1 - \cos(\theta)$ where $\theta$ is the angle between vectors.

---

### 2. Uniformity Loss

**Purpose**: Prevent representation collapse; ensure embeddings spread uniformly across the hypersphere.

**Formula**:
$$L_{\text{uni}}(x) = \log\left(\text{mean}_{i<j}\left[\exp\left(-2 \| x_i - x_j \|_2^2\right)\right]\right)$$

**High-level Idea**:
- Compute pairwise distances between all unique embeddings in a batch
- Apply exponential kernel: far-apart pairs get high weight, close pairs get low weight
- Take log of mean → higher value when embeddings are spread apart
- Maximizing this loss spreads embeddings uniformly

**Implementation** (Vectorized):
```python
def uniformity_loss(vectors, eps=1e-12):
    """
    vectors: [n_unique, 768], L2-normalized
    """
    if vectors.size(0) < 2:
        return torch.tensor(0.0, device=vectors.device)
    
    # Pairwise L2 distances
    pairwise_dists = torch.cdist(vectors, vectors, p=2)  # [n, n]
    
    # Exclude diagonal (same embeddings)
    mask = ~torch.eye(vectors.size(0), dtype=torch.bool, device=vectors.device)
    valid_dists = pairwise_dists[mask]
    
    # Exponential kernel: exp(-2 * dist²)
    exp_term = torch.exp(-2 * valid_dists ** 2)
    mean_exp = torch.mean(exp_term)
    
    # Log-mean
    loss_uni = torch.log(mean_exp + eps)
    
    return loss_uni
```

**Two Uniformity Terms** (Computed Separately):

1. **Query Uniformity** ($L_{\text{uni}}^q$):
    - Extract unique query embeddings by the composite key (head_id, relation)
   - Compute uniformity loss for unique queries
    - Prevents repeated (h, r) pairs from clustering

2. **Entity Uniformity** ($L_{\text{uni}}^t$):
   - Extract unique tail embeddings by tail_id
   - Compute uniformity loss for unique entities
   - Prevents multiple triples with same t from clustering

**Combined**:
$$L_{\text{uniform}} = L_{\text{uni}}^q + L_{\text{uni}}^t$$

---

### 3. Total Loss

**Formula**:
$$L_{\text{total}} = L_{\text{align}} + \gamma \cdot L_{\text{uniform}}$$

Where $\gamma = $ `--directau-gamma` (typically 1.0)

**Loss Value Ranges** (Typical):
- $L_{\text{align}}$ ≈ 0.1 - 1.0 (squared L2 distances)
- $L_{\text{uniform}}$ ≈ -1.0 to -5.0 (log of mean exponential)
- $L_{\text{total}}$ ≈ -5.0 to 1.0 (depends on $\gamma$)

**Trade-offs**:
- When $\gamma = 0$: Only alignment (triples match perfectly but may collapse)
- When $\gamma = 1$: Balanced alignment and uniformity (recommended)
- When $\gamma > 1$: Emphasis on uniformity (more spread, less tight alignment)

---

### 4. SimKGC InfoNCE Loss

**Purpose**: Train the query embedding to rank the positive tail above all negative candidates in the batch and optional buffers.

**Core Formula**:
$$L_{\text{infonce}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(s_{i,i})}{\sum_{j=1}^{M} \exp(s_{i,j})}$$

Where:
- $B$ is the current batch size
- $M$ is the number of candidate tails after adding optional negatives
- $s_{i,j}$ is the similarity score for query $i$ and candidate $j$

**Score Construction**:
1. Compute in-batch dot products: $q_i \cdot t_j$
2. Scale scores by inverse temperature $\exp(\text{log\_inv\_t})$
3. Subtract the additive margin from the diagonal during training
4. Append pre-batch negatives when `--pre-batch > 0`
5. Append self-negatives when `--use-self-negative`
6. Apply triplet masking to remove known positives
7. Compute cross entropy with labels `[0, 1, ..., B-1]`

**Behavioral Notes**:
- Lower temperature makes the distribution sharper
- Pre-batch negatives increase candidate diversity without new forward passes
- Self-negatives make the query compete against its own head embedding
- Triplet masking keeps known true triples out of the negative set

---

## Training Algorithm

### Data Preparation

1. **Index entities and relations** from train/valid/test files
2. **Load text descriptions** for each entity and relation (names, aliases)
3. **Tokenize descriptions** using BERT tokenizer (max_length=128)
4. **Cache token IDs** for efficient batch collation

### Training Loop

```
for epoch in range(n_epochs):
    
    # Shuffle training data
    shuffled_triples = shuffle(train_triples)
    
    lr_scheduler.step()
    epoch_loss = 0.0
    
    for batch_idx, batch_data in enumerate(batches):
        
        # STEP 1: Extract batch triplets
        h_batch, r_batch, t_batch = batch_data     # [batch_size]
        batch_exs = batch_data.examples             # Example objects with head_id, tail_id
        
        # STEP 2: Encode query pairs (head + relation)
        # Input: text descriptions of h_i and r_i
        # Output: dense embeddings via BERT
        q_batch = encode_query(h_batch, r_batch)    # [batch_size, 768]
        
        # STEP 3: Encode tail entities
        t_batch_emb = encode_tail(t_batch)          # [batch_size, 768]
        
        # STEP 4: Explicit L2 Normalization
        # Project all embeddings onto unit hypersphere
        q_batch = L2_normalize(q_batch, eps=1e-12)
        t_batch_emb = L2_normalize(t_batch_emb, eps=1e-12)
        
        # STEP 5: Compute Alignment Loss
        # Minimize squared L2 distance for matching (h,r,t) pairs
        diff = q_batch - t_batch_emb                    # [batch_size, 768]
        squared_l2_dist = torch.sum(diff ** 2, dim=-1)  # [batch_size]
        loss_align = torch.mean(squared_l2_dist)        # scalar
        
        # STEP 6: Extract unique embeddings for Uniformity Loss
        # Deduplicate queries by (head_id, relation)
        query_keys = [(ex.head_id, ex.relation) for ex in batch_exs]
        q_unique_idx = get_unique_indices(query_keys)
        q_unique = q_batch[q_unique_idx]
        
        # Deduplicate tails by tail_id
        tail_ids = [ex.tail_id for ex in batch_exs]
        t_unique_idx = get_unique_indices(tail_ids)
        t_unique = t_batch_emb[t_unique_idx]
        
        # STEP 7: Compute Uniformity Loss
        # Spread unique embeddings uniformly across hypersphere
        loss_uni_q = uniformity_loss(q_unique, eps=1e-12)
        loss_uni_t = uniformity_loss(t_unique, eps=1e-12)
        loss_uniform = loss_uni_q + loss_uni_t
        
        # STEP 8: Total Loss
        loss = loss_align + gamma * loss_uniform
        
        # STEP 9: Backward Pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        
        # STEP 10: Clear gradients
        optimizer.zero_grad()
        epoch_loss += loss.item() * batch_size
    
    # STEP 11: Validation
    valid_metrics = validate(valid_data, model)
    valid_loss = valid_metrics['loss']
    
    # STEP 12: Early Stopping
    if valid_loss < best_valid_loss:
        save_checkpoint(model, optimizer, epoch)
        best_valid_loss = valid_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Valid Loss = {valid_loss:.4f}")
```

### Helper Functions

#### L2 Normalization
```python
def L2_normalize(vectors, eps=1e-12):
    """
    Project vectors onto unit hypersphere.
    vectors: [batch_size, dim]
    Returns: [batch_size, dim], L2-normalized
    """
    norm = torch.sqrt(torch.sum(vectors ** 2, dim=-1, keepdim=True) + eps)
    return vectors / norm
```

#### Uniqueness Extraction
```python
def get_unique_indices(ids):
    """
    Find first occurrence index of each unique ID.
    ids: List of hashable keys, such as integers or (head_id, relation) tuples
    Returns: List of indices corresponding to first occurrence
    """
    seen = {}
    unique_idx = []
    for i, id_val in enumerate(ids):
        if id_val not in seen:
            seen[id_val] = i
            unique_idx.append(i)
    return unique_idx
```

#### Uniformity Loss Function
```python
def uniformity_loss(vectors, eps=1e-12):
    """
    Measure how uniformly vectors are spread on hypersphere.
    Higher value = more spread.
    vectors: [n_unique, 768], L2-normalized
    Returns: scalar
    """
    n = vectors.size(0)
    if n < 2:
        return torch.tensor(0.0, device=vectors.device)
    
    # Pairwise L2 distances
    dists = torch.cdist(vectors, vectors, p=2)  # [n, n]
    
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=vectors.device)
    valid_dists = dists[mask]
    
    # Exponential kernel: exp(-2 * dist²)
    exp_dists = torch.exp(-2 * (valid_dists ** 2))
    mean_exp = torch.mean(exp_dists)
    
    # Log-mean → higher when spread apart
    loss = torch.log(mean_exp + eps)
    
    return loss
```

---

## Inference & Evaluation

### Link Prediction

For each test triple (h, r, t), rank entities and compute metrics:

#### 1. Pre-compute All Entity Embeddings

```python
# Encode all entities once (vectorized)
all_entity_texts = [entity_dict[e] for e in range(n_entities)]
entity_embeddings = encode_tail_batch(all_entity_texts)  # [n_entities, 768]

# L2 Normalize
entity_embeddings = L2_normalize(entity_embeddings)     # [n_entities, 768]
```

#### 2. Tail Prediction (rank entities for "?" in (h, r, ?))

```python
for (h, r, t) in test_triples:
    
    # Encode query (h, r)
    q = encode_query(h, r)                              # [768]
    q = L2_normalize(q)                                 # Project to unit sphere
    
    # Score all entities: dot product (similarity on unit sphere)
    scores = q @ entity_embeddings.T                    # [n_entities]
    
    # Rank entities by score (descending: higher score = better match)
    ranked_indices = argsort(scores, descending=True)   # [n_entities]
    
    # Find rank of true entity
    true_entity_rank = position of t in ranked_indices + 1
    
    # Optional: Apply filtering (remove training seen entities)
    if use_filtering:
        valid_mask = get_unseen_mask(h, r)              # [n_entities, bool]
        scores[~valid_mask] = -inf
        ranked_indices = argsort(scores, descending=True)
        true_entity_rank = position of t in ranked_indices + 1
    
    # Record rank for metrics computation
    tail_ranks.append(true_entity_rank)
```

#### 3. Head Prediction (rank entities for "?" in (?, r, t))

```python
for (h, r, t) in test_triples:
    
    # Use inverse relation: (t, r_inv, ?)
    r_inv = "inverse_" + relation_text[r]
    
    # Encode query (t, r_inv)
    q = encode_query(t, r_inv)                          # [768]
    q = L2_normalize(q)
    
    # Score all entities
    scores = q @ entity_embeddings.T                    # [n_entities]
    
    # Rank by descending score
    ranked_indices = argsort(scores, descending=True)
    
    # Find rank of true head entity h
    true_head_rank = position of h in ranked_indices + 1
    
    # Apply filtering if needed
    if use_filtering:
        valid_mask = get_unseen_mask(t, r_inv)
        scores[~valid_mask] = -inf
        ranked_indices = argsort(scores, descending=True)
        true_head_rank = position of h in ranked_indices + 1
    
    head_ranks.append(true_head_rank)
```

#### 4. Compute Metrics

```python
def compute_metrics(tail_ranks, head_ranks):
    """Compute link prediction metrics."""
    all_ranks = tail_ranks + head_ranks
    
    # Mean Rank (MR)
    MR = mean(all_ranks)
    
    # Mean Reciprocal Rank (MRR)
    MRR = mean([1/r for r in all_ranks])
    
    # Hits@K
    Hits@1 = count(r == 1 for r in all_ranks) / len(all_ranks)
    Hits@3 = count(r <= 3 for r in all_ranks) / len(all_ranks)
    Hits@10 = count(r <= 10 for r in all_ranks) / len(all_ranks)
    
    return {
        'MR': MR,
        'MRR': MRR,
        'Hits@1': Hits@1,
        'Hits@3': Hits@3,
        'Hits@10': Hits@10
    }
```

#### 5. Chunked Inference (for Large KGs)

For datasets with millions of entities, process embeddings in chunks to save memory:

```python
def compute_scores_chunked(q, entity_embeddings, chunk_size=8192):
    """
    Compute scores for query against all entities in chunks.
    """
    n_entities = entity_embeddings.size(0)
    scores_list = []
    
    for chunk_start in range(0, n_entities, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_entities)
        chunk_embs = entity_embeddings[chunk_start:chunk_end]  # [chunk_size, 768]
        
        # Compute scores for this chunk
        chunk_scores = q @ chunk_embs.T                        # [chunk_size]
        scores_list.append(chunk_scores)
    
    # Concatenate scores
    all_scores = torch.cat(scores_list, dim=0)               # [n_entities]
    
    return all_scores
```

---

### Triple Classification Task

For labeled test triples with binary labels (valid/invalid):

#### Training

Same as link prediction: minimize alignment + uniformity loss

#### Testing

```python
# Find optimal classification threshold on validation set
best_threshold = find_optimal_threshold(val_scores, val_labels)

# Classify test triples
predictions = []
for (h, r, t, true_label) in test_triples_with_labels:
    
    # Encode query and tail
    q = encode_query(h, r)
    t_emb = encode_tail(t)
    
    # Normalize
    q = L2_normalize(q)
    t_emb = L2_normalize(t_emb)
    
    # Compute similarity
    score = q @ t_emb                                   # scalar in [-2, 2]
    
    # Classify
    pred_label = 1 if score >= best_threshold else 0
    predictions.append(pred_label)

# Compute classification metrics
accuracy = count(pred == true for pred, true) / len(test_triples_with_labels)
precision, recall, f1 = compute_classification_metrics(predictions, true_labels)
auc = compute_roc_auc(scores, true_labels)

return {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1': f1,
    'AUC': auc
}
```

---

## Command-Line Configuration

```bash
python main.py \
    --task wn18rr \
    --pretrained-model distilbert-base-uncased \
    --batch-size 512 \
    --max-length 128 \
    --learning-rate 3e-5 \
    --warmup-steps 400 \
    --epochs 20 \
    --early-stop-patience 5 \
    --directau-gamma 1.0 \
    --directau-eps 1e-12 \
    --chunk-size 8192 \
    --use-amp true
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained-model` | distilbert-base-uncased | Pre-trained BERT model |
| `--batch-size` | 512 | Training batch size |
| `--max-length` | 128 | Tokenizer max sequence length |
| `--learning-rate` | 3e-5 | AdamW optimizer learning rate |
| `--warmup` | 400 | Learning rate warmup steps |
| `--epochs` | 20 | Total training epochs |
| `--early-stop-patience` | 5 | Patience for early stopping |
| `--directau-gamma` | 1.0 | Weight for uniformity loss |
| `--directau-eps` | 1e-12 | Epsilon for L2 normalization |
| `--chunk-size` | 8192 | Entity chunk size for inference |
| `--use-amp` | False | Automatic Mixed Precision |
| **Advanced Techniques** | | |
| `--use-link-graph` | False | Enable neighbor context augmentation |
| `--lr-scheduler` | linear | Learning rate schedule (linear/cosine) |
| `--wd` / `--weight-decay` | 1e-4 | L2 regularization coefficient |
| `--pre-batch` | 0 | Pre-batch negative buffer multiplier |
| `--pre-batch-weight` | 0.5 | Weight for pre-batch negatives |
| `--finetune-t` | False | Make temperature learnable |
| `--t` | 0.05 | Initial temperature (InfoNCE only) |

---

## Algorithm Characteristics

### Properties

| Property | DirectAU |
|----------|----------|
| **Loss Function** | Alignment + Uniformity |
| **Normalization** | Explicit L2 normalization |
| **Hyperparameters** | Minimal (only γ for uniformity weight) |
| **Temperature Scaling** | No (fixed to 1.0) |
| **Negative Sampling** | Implicit via uniformity loss |
| **Margin-based Ranking** | No |
| **Triplet Masking** | No required |
| **Inference Ranking** | Descending (higher score better) |
| **Uniqueness Handling** | Required for uniformity |
| **Memory Usage** | Lower than InfoNCE with pre-batch |
| **Interpretability** | High (direct distance minimization) |

### Advantages

1. **Simpler Loss**: Direct geometric optimization (alignment + spread)
2. **Fewer Hyperparameters**: Only γ to tune (vs temperature, margin, pre-batch)
3. **Theoretical Grounding**: Based on contrastive learning principles
4. **Uniform Distribution**: Guaranteed via uniformity loss
5. **No Training Leakage**: Doesn't require triplet masking
6. **Better Scaling**: Linear complexity in entity count (no pre-batch buffer)

### Limitations

1. **Uniqueness Overhead**: Must deduplicate embeddings for uniformity
2. **Gradient Flow**: Pairwise distance computation can be expensive for large batches
3. **Loss Balancing**: Must tune γ for alignment-uniformity trade-off
4. **Batch Dependency**: Uniformity loss depends on batch composition

---

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Encode batch (BERT) | O(B × L × d) | O(B × L × d) |
| L2 Normalization | O(B × d) | O(B × d) |
| Alignment Loss | O(B × d) | O(B × d) |
| Uniqueness Extraction | O(B) | O(B) |
| Pairwise Distances | O(U² × d) | O(U²) |
| Uniformity Loss | O(U² × d) | O(U²) |
| Forward Pass Total | O(B × L × d) | O(B × d) |
| Inference (pre-compute) | O(N × L × d) | O(N × d) |
| Inference (per query) | O(N × d) | O(N) |

**Notation**:
- B = batch size
- L = max token length
- d = embedding dimension (768)
- N = total entities
- U = unique entities in batch (≤ B)

---

## Memory Optimizations

### 1. Automatic Mixed Precision (AMP)

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    q_batch = encode_query(h_batch, r_batch)
    t_batch_emb = encode_tail(t_batch)
    loss = compute_loss(q_batch, t_batch_emb)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- ~2x memory reduction (FP16 forward pass)
- ~20-30% speedup without accuracy loss
- FP32 backward for numerical stability

### 2. Gradient Accumulation

For larger effective batch sizes with limited GPU memory:

```python
# Effective batch_size = micro_batch * grad_accum_steps
accumulation_steps = 4
effective_batch_size = 512 * 4 = 2048

loss = compute_loss(...) / accumulation_steps
loss.backward()

if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 3. Chunked Inference

Process large entity sets without loading all embeddings:

```python
entity_scores = []

for chunk_start in range(0, n_entities, chunk_size):
    chunk_end = min(chunk_start + chunk_size, n_entities)
    chunk_embs = entity_embeddings[chunk_start:chunk_end]
    chunk_scores = q @ chunk_embs.T
    entity_scores.append(chunk_scores)

full_scores = torch.cat(entity_scores, dim=0)
```

**Example**: 
- Full entity set: Wikidata5M (5.9M entities × 768 = 4.5GB)
- Chunk size: 8192 → Process in ~720 chunks
- Per-chunk memory: 8192 × 768 × 4 bytes = 25MB

### 4. Selective Unique Deduplication

```python
# Only deduplicate if batch has many duplicates
unique_ratio = len(unique(head_ids)) / len(head_ids)

if unique_ratio < 0.5:  # More than 50% duplicates
    q_unique = q_batch[unique_indices]
    loss_uni = uniformity_loss(q_unique)
else:
    loss_uni = uniformity_loss(q_batch)  # Skip dedup
```

---

## Practical Implementation Tips

### 1. Data Loading

```python
# Use DataLoader with multiprocessing
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 2. Gradient Clipping

Prevent exploding gradients:

```python
max_grad_norm = 10.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
```

### 3. Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.WarmupLinearSchedule(
    optimizer,
    num_warmup_steps=400,
    num_training_steps=total_steps
)

for epoch in range(epochs):
    train_loop()
    scheduler.step()
```

### 4. Checkpoint Saving

```python
if valid_loss < best_valid_loss:
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': valid_loss
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

---

## Advanced Techniques

Beyond the core DirectAU loss, the codebase implements several advanced techniques to enhance training and inference:

### 1. Neighbor-Based Context Augmentation

**Purpose**: Enrich entity representations with 1-hop neighborhood information from the knowledge graph.

**Configuration**:
```bash
--use-link-graph        # Enable neighbor context (default: False)
```

**Implementation** (in `doc.py`):
```python
def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    """
    Retrieve neighboring entity names from link graph.
    """
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id, max_to_keep=10)
    
    # Prevent label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    
    # Convert to entity names
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    return ' '.join(entities)

# Usage in Example.vectorize():
if args.use_link_graph:
    if len(head_desc.split()) < 20:
        head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
    if len(tail_desc.split()) < 20:
        tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)
```

**Benefits**:
- Incorporates structural information from KG
- Prevents description over-tokenization (only adds if < 20 words)
- Ensures label leakage prevention during training
- Deterministic ordering for reproducibility

**Example**:
```
Original head_desc: "philosopher from ancient Greece"
With neighbors: "philosopher from ancient Greece Aristotle Plato Socrates"
```

---

### 2. Triplet Masking for Training Leakage Prevention

**Purpose**: Prevent training triples from being penalized as false negatives during in-batch negative sampling.

**Configuration**: Automatically applied during training (not applied at test time)

**Implementation** (in `triplet_mask.py`):
```python
def construct_mask(row_exs: List, col_exs: List = None) -> torch.tensor:
    """
    Build mask to prevent training leakage.
    Returns: [num_row × num_col] binary mask (True = valid negative, False = ignore)
    """
    # Start with exact match mask (diagonal = True, off-diagonal = True initially)
    row_entity_ids = torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in row_exs])
    col_entity_ids = row_entity_ids if col_exs is None else \
        torch.LongTensor([entity_dict.entity_to_idx(ex.tail_id) for ex in col_exs])
    
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    triplet_mask.fill_diagonal_(True)  # Positive at diagonal
    
    # Mask out other possible neighbors from training set
    for i in range(len(row_exs)):
        head_id, relation = row_exs[i].head_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(head_id, relation)
        
        for j in range(len(col_exs)):
            if i == j:
                continue
            tail_id = col_exs[j].tail_id
            if tail_id in neighbor_ids:
                triplet_mask[i][j] = False  # Ignore this negative
    
    return triplet_mask

# Usage in models.py:
if triplet_mask is not None:
    logits.masked_fill_(~triplet_mask, -1e4)  # Set masked positions to very negative
```

**Example**:
```
Training triple: (Washington, capital_of, USA)
Other capitals in batch: (Paris, capital_of, France), (Rome, capital_of, Italy)

Without masking: CrossEntropy penalizes (Washington, USA) as false negative
With masking: These known positives are ignored in loss computation
```

**Benefits**:
- Prevents artificial penalization of correct facts
- Ensures training focuses on genuine negatives
- Improves convergence and final performance

---

### 3. Learning Rate Scheduling

**Purpose**: Control learning rate decay throughout training for better convergence.

**Configuration**:
```bash
--lr-scheduler linear          # Linear decay (default)
--lr-scheduler cosine          # Cosine annealing (alternative)
--warmup 400                   # Warmup steps (default)
--lr 3e-5                      # Initial learning rate
```

**Implementation**:
```python
# Linear schedule with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=400,
    num_training_steps=total_training_steps
)

# or Cosine schedule with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=400,
    num_training_steps=total_training_steps
)

# Usage in training loop
for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()
```

**Learning Rate Curve**:

**Linear Warmup + Decay**:
```
LR ↑
   |     ╱╲
   |    ╱  ╲
   |   ╱    ╲
   |  ╱      ╲___
   |╱────────────────
   └────────────────→ steps
     warmup   training
```

**Benefits**:
- Warmup prevents large gradient updates early
- Controlled decay improves convergence
- Commonly used with transformer models

---

### 4. Gradient Accumulation

**Purpose**: Simulate a larger batch size when GPU memory cannot hold the full batch.

**Configuration**:
```bash
--grad-accum-steps 4      # Number of micro-batches to accumulate before optimizer.step()
```

**Implementation**:
```python
loss = loss / accumulation_steps
loss.backward()

if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**Benefits**:
- Improves gradient stability with large effective batches
- Keeps peak memory lower than a true large batch
- Works with both DirectAU and SimKGC training loops

---

### 5. Pre-batch Negatives: Historical Embedding Buffer

**Purpose**: Increase negative diversity without additional forward passes by maintaining a circular buffer of past embeddings.

**Configuration**:
```bash
--pre-batch 0                  # Number of pre-batch multipliers (default: 0, disabled)
--pre-batch-weight 0.5         # Weight for pre-batch negatives
```

**Implementation** (in `models.py`):
```python
# Initialize circular buffer
num_pre_batch_vectors = max(1, args.pre_batch) * batch_size
random_vector = torch.randn(num_pre_batch_vectors, hidden_size)
self.register_buffer("pre_batch_vectors", F.normalize(random_vector, dim=-1))
self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

# During training loop (commented out in DirectAU mode)
if self.pre_batch > 0 and self.training:
    pre_batch_logits = q_batch @ pre_batch_vectors.T
    pre_batch_logits *= exp(log_inv_t) * pre_batch_weight
    logits = torch.cat([logits, pre_batch_logits], dim=-1)
```

**Benefits**:
- Larger effective batch size for negatives
- Memory efficient (no additional forward passes)
- Improves negative sampling diversity

**Note**: Typically used with InfoNCE loss, not DirectAU (which has implicit negative sampling via uniformity loss).

---

### 6. Fine-tunable Temperature (InfoNCE only)

**Purpose**: Make temperature parameter learnable for better loss scaling.

**Configuration**:
```bash
--finetune-t            # Make temperature trainable (default: False)
--t 0.05                # Initial temperature value
```

**Implementation** (in `models.py`):
```python
if self.args.finetune_t:
    log_inv_t = torch.nn.Parameter(
        torch.tensor(1.0 / self.args.t).log(),
        requires_grad=True
    )
    optimizer.add_param_group({'params': [log_inv_t]})
```

**Effect**:
- Temperatures < 1.0 → sharper logits (lower entropy)
- Temperatures > 1.0 → softer logits (higher entropy)
- Learned temperature adapts to data characteristics

**Note**: Not used with DirectAU loss (which fixes temperature implicitly to 1.0).

---

### 7. Weight Decay (L2 Regularization)

**Purpose**: Prevent overfitting by penalizing large weight magnitudes.

**Configuration**:
```bash
--wd 1e-4               # Weight decay coefficient (default)
```

**Implementation**:
```python
optimizer = AdamW(
    params=trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay  # L2 penalty
)
```

**Effect**:
- Loss becomes: L_total = DirectAULoss + weight_decay * ||params||²
- Encourages smaller, simpler models
- Helps generalization to unseen data

**Typical values**:
- `--wd 1e-4` for transformer fine-tuning
- `--wd 1e-3` for aggressive regularization
- `--wd 0` to disable

---

### Combining Techniques: Example Configurations

#### Minimal (DirectAU only):
```bash
python main.py --task wn18rr --directau --epochs 20
```
- Uses DirectAU loss only
- No neighbor augmentation
- Standard learning rate scheduling

#### Enhanced (DirectAU + Neighbors):
```bash
python main.py --task wn18rr --directau --use-link-graph \
    --epochs 20 --warmup 400 --lr 2e-5
```
- Adds neighbor context enrichment
- Controlled warmup
- Lower learning rate for fine-tuning

#### Full Advanced (All techniques):
```bash
python main.py --task wn18rr --directau --use-link-graph \
    --epochs 20 --warmup 400 --lr 2e-5 \
    --lr-scheduler cosine --wd 1e-4 \
    --use-amp --batch-size 512 --chunk-size 8192
```
- Neighbor augmentation ✅
- Cosine LR scheduling ✅
- L2 regularization ✅
- Mixed precision ✅
- Large batch processing ✅

---

## Summary

**SimKGC with DirectAU Loss** provides a **simple, theoretically-grounded approach** to knowledge graph completion:

### Key Features

1. ✅ **Bi-encoder architecture**: HR encoder + Tail encoder (shared BERT initialization)
2. ✅ **Text-based representations**: Entity descriptions → dense embeddings
3. ✅ **DirectAU loss**: Alignment (minimize L2 distance) + Uniformity (spread on sphere)
4. ✅ **No complex hyperparameters**: Only γ for uniformity weight
5. ✅ **Explicit normalization**: All embeddings on unit hypersphere
6. ✅ **Scalable inference**: Chunked processing for large entity sets
7. ✅ **Supports both tasks**: Link prediction + triple classification
8. ✅ **Eight strategy switches**: Neighbor augmentation, triplet masking, LR scheduling, gradient accumulation, pre-batch negatives, AMP, weight decay, temperature tuning

### Advanced Techniques Available

- **Neighbor Context** (`--use-link-graph`): Enrich entity text with 1-hop neighbors
- **Triplet Masking**: Prevent training leakage in negative sampling
- **Learning Rate Scheduling**: Linear or cosine warmup with decay
- **Gradient Accumulation** (`--grad-accum-steps`): Simulate larger effective batches under tight memory budgets
- **Pre-batch Negatives** (`--pre-batch`): Historical embedding buffer for negatives
- **Mixed Precision** (`--use-amp`): FP16 forward, FP32 backward
- **Weight Decay** (`--wd`): L2 regularization
- **Fine-tunable Temperature** (`--finetune-t`): Learnable temperature for InfoNCE

### When to Use DirectAU

- ✅ Want simpler loss without temperature/margin tuning
- ✅ Need interpretable geometric optimization
- ✅ Prefer guaranteed uniform embedding distribution
- ✅ Working with large-scale KGs (millions of entities)
- ✅ Have sufficient GPU memory for batch processing

### Related Work

- **Contrastive Learning**: Schroff et al. 2015, Chen et al. 2020
- **Uniformity-Alignment**: Wang & Isola 2020
- **KG Embeddings**: TransE, DistMult, ComplEx, RotatE
- **BERT for KG**: SimKGC (Wang et al. 2021)

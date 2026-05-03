# SimKGC with DirectAU Loss: Flow Diagrams

## 1. Overall System Architecture

```
Knowledge Graph Data
        ↓
    ┌─────────────────────────────────────┐
    │ Parse Triples & Load Descriptions   │
    │ Entity/Relation Text (from files)    │
    └─────────────────────────────────────┘
        ↓
    ┌──────────────────┐    ┌──────────────────┐
    │ Head + Relation  │    │   Tail Entity    │
    │ Description Text │    │   Description    │
    └──────────────────┘    └──────────────────┘
         ↓                       ↓
    BERT Tokenization (max_length=128)
         ↓                       ↓
    [HR Encoder (BERT)]   [Tail Encoder (BERT)]
    (Separate weights)    (Same init, diff weights)
         ↓                       ↓
    Pooling (CLS Token)
         ↓                       ↓
    L2 Normalization (Explicit)
         ↓                       ↓
    Query Vector              Tail Vector
    [768 dimensions]          [768 dimensions]
    (unit hypersphere)        (unit hypersphere)
         ↓                       ↓
    ┌──────────────────────────────────────┐
    │      DirectAU Loss Computation       │
    │  ┌──────────────────────────────────┐│
    │  │ 1. Alignment Loss                ││
    │  │    L_align = mean(||q - t||²)    ││
    │  ├──────────────────────────────────┤│
    │  │ 2. Unique Query Embeddings       ││
    │  │    Deduplicate by head_id        ││
    │  ├──────────────────────────────────┤│
    │  │ 3. Unique Tail Embeddings        ││
    │  │    Deduplicate by tail_id        ││
    │  ├──────────────────────────────────┤│
    │  │ 4. Query Uniformity Loss         ││
    │  │    L_uni_q (pairwise distances)  ││
    │  ├──────────────────────────────────┤│
    │  │ 5. Tail Uniformity Loss          ││
    │  │    L_uni_t (pairwise distances)  ││
    │  ├──────────────────────────────────┤│
    │  │ 6. Total Loss                    ││
    │  │    L = Align + γ*Uniform         ││
    │  └──────────────────────────────────┘│
    └──────────────────────────────────────┘
         ↓
    Backpropagation (with AMP)
         ↓
    Optimizer Update
```

---

## 2. Training Loop: DirectAU-Only Execution

```
┌─────────────────────────────────────────┐
│         START TRAINING                  │
│  for epoch in range(num_epochs):        │
└─────────────────────────────────────────┘
    ↓
    [Shuffle Training Data]
    ↓
    FOR each batch of (h_ids, r_ids, t_ids):
    │
    ├─→ STEP 1: Encode Queries (h, r)
    │   └─ q_batch = encode_query(h, r)
    │      └─ Output: [batch_size, 768]
    │
    ├─→ STEP 2: Encode Tails (t)
    │   └─ t_batch = encode_tail(t)
    │      └─ Output: [batch_size, 768]
    │
    ├─→ STEP 3: Explicit L2 Normalization
    │   ├─ q_batch = L2_normalize(q_batch)
    │   └─ t_batch = L2_normalize(t_batch)
    │
    ├─→ STEP 4: Compute Alignment Loss
    │   ├─ diff = q_batch - t_batch
    │   ├─ squared_dist = sum(diff², dim=-1)
    │   └─ loss_align = mean(squared_dist)
    │
    ├─→ STEP 5: Extract Unique Embeddings
    │   ├─ head_ids = [ex.head_id for ex in batch]
    │   ├─ q_unique_idx = find_unique(head_ids)
    │   ├─ q_unique = q_batch[q_unique_idx]
    │   │
    │   ├─ tail_ids = [ex.tail_id for ex in batch]
    │   ├─ t_unique_idx = find_unique(tail_ids)
    │   └─ t_unique = t_batch[t_unique_idx]
    │
    ├─→ STEP 6: Compute Query Uniformity Loss
    │   ├─ dist_q = pairwise_distance(q_unique)
    │   ├─ exp_q = exp(-2 * dist_q²)
    │   └─ loss_uni_q = log(mean(exp_q))
    │
    ├─→ STEP 7: Compute Tail Uniformity Loss
    │   ├─ dist_t = pairwise_distance(t_unique)
    │   ├─ exp_t = exp(-2 * dist_t²)
    │   └─ loss_uni_t = log(mean(exp_t))
    │
    ├─→ STEP 8: Compute Total Uniformity Loss
    │   └─ loss_uniform = loss_uni_q + loss_uni_t
    │
    ├─→ STEP 9: Compute Total Loss
    │   ├─ gamma = directau_gamma (e.g., 1.0)
    │   └─ loss = loss_align + gamma * loss_uniform
    │
    ├─→ STEP 10: Backward Pass (with AMP)
    │   ├─ scaler.scale(loss).backward()
    │   ├─ clip_grad_norm_(max_norm=10.0)
    │   ├─ scaler.step(optimizer)
    │   ├─ scaler.update()
    │   └─ optimizer.zero_grad()
    │
    └─→ [END BATCH]
    ↓
    [Compute Validation Metrics]
    ├─→ IF validation_loss < best_loss:
    │   ├─ [Save Checkpoint]
    │   └─ patience_counter = 0
    │
    └─→ ELSE:
        └─ patience_counter += 1
    ↓
    [Early Stopping Check]
    ├─→ IF patience_counter >= early_stop_patience:
    │   └─ [BREAK and Load Best Checkpoint]
    │
    └─→ ELSE:
        └─ [Continue Next Epoch]
    ↓
└─────────────────────────────────────────┐
│         END TRAINING                    │
└─────────────────────────────────────────┘
```

---

## 3. DirectAU Loss Computation: Detailed Flow

```
╔════════════════════════════════════════════════════════════╗
║           DIRECTAU LOSS COMPUTATION FLOW                   ║
╚════════════════════════════════════════════════════════════╝

INPUT:
  • q_batch: [batch_size, 768], L2-normalized
  • t_batch: [batch_size, 768], L2-normalized
  • batch_exs: List[Example] with head_id, tail_id
  • gamma: uniformity weight (default 1.0)
  • eps: numerical stability (default 1e-12)

════════════════════════════════════════════════════════════

┌─ BRANCH 1: ALIGNMENT LOSS ─────────────────────────────────┐
│                                                             │
│  diff = q_batch - t_batch                                  │
│  → [batch_size, 768]                                       │
│                                                             │
│  squared_dists = torch.sum(diff ** 2, dim=-1)              │
│  → [batch_size]                                            │
│                                                             │
│  L_align = torch.mean(squared_dists)                       │
│  → scalar                                                  │
│                                                             │
│  Interpretation: Average squared L2 distance of matching   │
│                 pairs on unit hypersphere                  │
│                 Lower = better alignment                   │
│                                                             │
└────────────────────────────────────────────────────────────┘
              ↓
┌─ BRANCH 2: UNIQUENESS EXTRACTION ──────────────────────────┐
│                                                             │
│  # Extract unique query indices                            │
│  head_ids = [ex.head_id for ex in batch_exs]              │
│  → List[int] of length batch_size                          │
│                                                             │
│  q_unique_idx = get_first_unique_indices(head_ids)        │
│  → List[int], length ≤ batch_size                          │
│                                                             │
│  q_unique = q_batch[q_unique_idx]                          │
│  → [num_unique_heads, 768]                                 │
│                                                             │
│  ─────────────────────────────────────────────────         │
│                                                             │
│  # Extract unique tail indices                             │
│  tail_ids = [ex.tail_id for ex in batch_exs]              │
│  → List[int] of length batch_size                          │
│                                                             │
│  t_unique_idx = get_first_unique_indices(tail_ids)        │
│  → List[int], length ≤ batch_size                          │
│                                                             │
│  t_unique = t_batch[t_unique_idx]                          │
│  → [num_unique_tails, 768]                                 │
│                                                             │
└────────────────────────────────────────────────────────────┘
              ↓
┌─ BRANCH 3: QUERY UNIFORMITY LOSS ──────────────────────────┐
│                                                             │
│  Input: q_unique [num_unique_q, 768]                       │
│                                                             │
│  # Step 3a: Compute pairwise L2 distances                  │
│  dist_q = torch.cdist(q_unique, q_unique, p=2)            │
│  → [num_unique_q, num_unique_q]                            │
│                                                             │
│  # Step 3b: Create mask to exclude diagonal                │
│  mask = ~torch.eye(num_unique_q, dtype=bool)              │
│  → [num_unique_q, num_unique_q], True except diagonal     │
│                                                             │
│  # Step 3c: Extract off-diagonal distances                 │
│  valid_dists_q = dist_q[mask]                              │
│  → [num_unique_q * (num_unique_q - 1)]                     │
│                                                             │
│  # Step 3d: Apply exponential kernel                       │
│  exp_q = torch.exp(-2 * (valid_dists_q ** 2))            │
│  → Higher values = pairs farther apart                     │
│                                                             │
│  # Step 3e: Compute mean of exponentials                   │
│  mean_exp_q = torch.mean(exp_q)                            │
│  → scalar in (0, 1]                                        │
│                                                             │
│  # Step 3f: Take logarithm (log-mean)                      │
│  L_uni_q = torch.log(mean_exp_q + eps)                     │
│  → scalar (typically -1.0 to -5.0)                         │
│  → Higher (closer to 0) = more uniform                     │
│                                                             │
└────────────────────────────────────────────────────────────┘
              ↓
┌─ BRANCH 4: TAIL UNIFORMITY LOSS ───────────────────────────┐
│                                                             │
│  Input: t_unique [num_unique_t, 768]                       │
│                                                             │
│  # Same computation as queries                             │
│  dist_t = torch.cdist(t_unique, t_unique, p=2)            │
│  mask = ~torch.eye(num_unique_t, dtype=bool)              │
│  valid_dists_t = dist_t[mask]                              │
│  exp_t = torch.exp(-2 * (valid_dists_t ** 2))            │
│  mean_exp_t = torch.mean(exp_t)                            │
│  L_uni_t = torch.log(mean_exp_t + eps)                     │
│                                                             │
│  Interpretation: Encourage tail embeddings to spread      │
│                 uniformly across the hypersphere           │
│                                                             │
└────────────────────────────────────────────────────────────┘
              ↓
┌─ BRANCH 5: TOTAL UNIFORMITY LOSS ─────────────────────────┐
│                                                             │
│  L_uniform = L_uni_q + L_uni_t                             │
│  → scalar (typically -2.0 to -10.0)                        │
│                                                             │
│  Combines both query and tail uniformity                   │
│                                                             │
└────────────────────────────────────────────────────────────┘
              ↓
┌─ BRANCH 6: TOTAL LOSS ─────────────────────────────────────┐
│                                                             │
│  gamma = args.directau_gamma (e.g., 1.0)                   │
│                                                             │
│  L_total = L_align + gamma * L_uniform                     │
│  → scalar                                                  │
│                                                             │
│  Example values (typical):                                 │
│    L_align ≈ 0.5                                           │
│    L_uniform ≈ -3.0                                        │
│    gamma = 1.0                                             │
│    L_total ≈ 0.5 + 1.0 * (-3.0) = -2.5                    │
│                                                             │
│  Gradient flows back to both encoders:                     │
│    - Minimize alignment (bring pairs closer)               │
│    - Maximize uniformity (spread embeddings)               │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Inference: Link Prediction

```
╔════════════════════════════════════════════════════════════╗
║           LINK PREDICTION INFERENCE FLOW                   ║
╚════════════════════════════════════════════════════════════╝

STAGE 1: PRE-COMPUTATION (Once per dataset)
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  [Load all entity texts from KG]                           │
│  └─ entity_texts = [text_e0, text_e1, ..., text_e_N]      │
│                                                             │
│  [Batch-encode all entities]                               │
│  └─ for chunk in chunks(entity_texts, chunk_size=8192):   │
│       chunk_embeds = encode_tail(chunk)                    │
│       all_embeddings.append(chunk_embeds)                  │
│                                                             │
│  entity_embeddings = concat(all_embeddings)               │
│  → [num_entities, 768]                                     │
│                                                             │
│  [Explicit L2 Normalization]                               │
│  entity_embeddings = L2_normalize(entity_embeddings)       │
│  → All entities on unit hypersphere                        │
│                                                             │
└────────────────────────────────────────────────────────────┘

STAGE 2: TAIL PREDICTION for (h, r, ?)
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  for (h, r, t) in test_triples:                            │
│                                                             │
│    [Encode query]                                          │
│    q = encode_query(h, r)                                  │
│    → [768]                                                 │
│                                                             │
│    [Normalize]                                             │
│    q = L2_normalize(q)                                     │
│    → on unit sphere                                        │
│                                                             │
│    [Score all entities (dot product on sphere)]            │
│    scores = q @ entity_embeddings.T                        │
│    → [num_entities]                                        │
│    → Range: [-2, 2] (for normalized vectors)               │
│                                                             │
│    [Rank by descending score (higher = better)]            │
│    ranked_indices = argsort(scores, descending=True)       │
│    → Most similar entity first                             │
│                                                             │
│    [Find rank of true tail t]                              │
│    true_rank = position_of(t in ranked_indices) + 1        │
│    → Rank 1 means best match                               │
│                                                             │
│    [Optional: Apply filtering]                             │
│    if use_filtering:                                       │
│      seen = get_training_triples(h, r)                     │
│      scores[seen] = -inf                                   │
│      ranked_indices = argsort(scores, desc=True)           │
│      true_rank = position_of(t in ranked_indices) + 1      │
│                                                             │
│    tail_ranks.append(true_rank)                            │
│                                                             │
└────────────────────────────────────────────────────────────┘

STAGE 3: HEAD PREDICTION for (?, r, t)
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  for (h, r, t) in test_triples:                            │
│                                                             │
│    [Create inverse relation]                               │
│    r_inv = "inverse_" + relation_text[r]                   │
│                                                             │
│    [Encode query with inverse relation]                    │
│    q = encode_query(t, r_inv)                              │
│    → [768]                                                 │
│                                                             │
│    [Normalize]                                             │
│    q = L2_normalize(q)                                     │
│                                                             │
│    [Score all entities]                                    │
│    scores = q @ entity_embeddings.T                        │
│    → [num_entities]                                        │
│                                                             │
│    [Rank entities by descending score]                     │
│    ranked_indices = argsort(scores, descending=True)       │
│                                                             │
│    [Find rank of true head h]                              │
│    true_rank = position_of(h in ranked_indices) + 1        │
│                                                             │
│    [Optional: Apply filtering]                             │
│    if use_filtering:                                       │
│      seen = get_training_triples(t, r_inv)                 │
│      scores[seen] = -inf                                   │
│      ranked_indices = argsort(scores, desc=True)           │
│      true_rank = position_of(h in ranked_indices) + 1      │
│                                                             │
│    head_ranks.append(true_rank)                            │
│                                                             │
└────────────────────────────────────────────────────────────┘

STAGE 4: COMPUTE METRICS
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  all_ranks = tail_ranks + head_ranks                       │
│  → Combined ranking positions                              │
│                                                             │
│  MR = mean(all_ranks)                                      │
│  → Lower is better (average rank should be high)           │
│                                                             │
│  MRR = mean(1/r for r in all_ranks)                        │
│  → Mean reciprocal rank (harmonic mean of ranks)           │
│                                                             │
│  Hits@K = count(r ≤ K for r in all_ranks) / len(all_ranks)│
│  → Proportion of triples in top-K ranking                  │
│                                                             │
│  return {                                                  │
│    'MR': MR,                                               │
│    'MRR': MRR,                                             │
│    'Hits@1': Hits@1,                                       │
│    'Hits@3': Hits@3,                                       │
│    'Hits@10': Hits@10                                      │
│  }                                                         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Triple Classification Task

```
╔════════════════════════════════════════════════════════════╗
║        TRIPLE CLASSIFICATION TRAINING & EVALUATION         ║
╚════════════════════════════════════════════════════════════╝

TRAINING: Same as Link Prediction
├─ Minimize DirectAU Loss (alignment + uniformity)
└─ Triples have binary labels (valid/invalid)

EVALUATION: Classification on Labeled Test Set
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  for (h, r, t, true_label) in test_triples_with_labels:   │
│                                                             │
│    [Encode query]                                          │
│    q = encode_query(h, r)                                  │
│    q = L2_normalize(q)                                     │
│    → [768]                                                 │
│                                                             │
│    [Encode tail]                                           │
│    t_emb = encode_tail(t)                                  │
│    t_emb = L2_normalize(t_emb)                             │
│    → [768]                                                 │
│                                                             │
│    [Compute similarity score (dot product)]                │
│    score = q @ t_emb                                       │
│    → scalar in [-2, 2]                                     │
│    → Higher = more similar                                 │
│                                                             │
│    predictions.append((score, true_label))                 │
│                                                             │
│  [VALIDATION: Find optimal threshold]                      │
│  best_threshold = argmax_F1(threshold)                     │
│  for threshold in [-2, -1.9, -1.8, ..., 1.9, 2.0]:       │
│    pred_labels = [1 if score ≥ threshold else 0           │
│                   for score in val_scores]                │
│    F1 = compute_F1(pred_labels, true_labels)              │
│                                                             │
│  [TEST: Classify with best threshold]                      │
│  test_predictions = [1 if score ≥ best_threshold else 0   │
│                      for score in test_scores]             │
│                                                             │
│  [COMPUTE METRICS]                                         │
│  accuracy = count(pred==true) / len(test_set)             │
│  precision, recall, f1 = compute_metrics(pred, true)      │
│  auc = compute_roc_auc(scores, true_labels)               │
│                                                             │
│  return {                                                  │
│    'Accuracy': accuracy,                                   │
│    'Precision': precision,                                 │
│    'Recall': recall,                                       │
│    'F1': f1,                                               │
│    'AUC': auc,                                             │
│    'Threshold': best_threshold                             │
│  }                                                         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## 3. DirectAU Loss Computation (Detailed)

```
INPUT: q_batch [B × 768], t_batch [B × 768]
       (both L2-normalized)
       batch_exs (list of examples with entity IDs)

═════════════════════════════════════════════

STEP 1: ALIGNMENT LOSS
┌───────────────────────────────────────────┐
│ diff = q_batch - t_batch                  │
│ squared_dist = sum(diff², dim=-1)         │
│ L_align = mean(squared_dist)              │
└───────────────────────────────────────────┘
Output: scalar (e.g., 0.5-2.0)

═════════════════════════════════════════════

STEP 2: EXTRACT UNIQUE EMBEDDINGS (for uniformity)
┌───────────────────────────────────────────┐
│ # For query embeddings:                   │
│ head_ids = [ex.head_id for ex in batch]   │
│ unique_head_indices = unique_ids(head_ids)│
│ q_unique = q_batch[unique_head_indices]   │
│ # Size: [num_unique_heads, 768]           │
│                                           │
│ # For tail embeddings:                    │
│ tail_ids = [ex.tail_id for ex in batch]   │
│ unique_tail_indices = unique_ids(tail_ids)│
│ t_unique = t_batch[unique_tail_indices]   │
│ # Size: [num_unique_tails, 768]           │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 3: UNIFORMITY LOSS FOR QUERIES
┌───────────────────────────────────────────┐
│ # Pairwise distances for unique queries   │
│ pairwise_dist = cdist(q_unique, q_unique) │
│ # Size: [num_unique_q, num_unique_q]      │
│                                           │
│ # Create mask to exclude diagonal        │
│ mask = ~eye(num_unique_q)                │
│ valid_dist = pairwise_dist[mask]         │
│                                           │
│ # Exponential kernel                      │
│ exp_term = exp(-2 * valid_dist²)         │
│ mean_exp = mean(exp_term)                │
│                                           │
│ # Log-mean (higher = more uniform)        │
│ L_uni_q = log(mean_exp + eps)            │
└───────────────────────────────────────────┘
Output: scalar (typically -1.0 to -5.0)

═════════════════════════════════════════════

STEP 4: UNIFORMITY LOSS FOR TAILS
┌───────────────────────────────────────────┐
│ (Same computation as queries)             │
│ pairwise_dist = cdist(t_unique, t_unique) │
│ mask = ~eye(num_unique_t)                │
│ valid_dist = pairwise_dist[mask]         │
│ exp_term = exp(-2 * valid_dist²)         │
│ mean_exp = mean(exp_term)                │
│ L_uni_t = log(mean_exp + eps)            │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 5: TOTAL UNIFORMITY LOSS
┌───────────────────────────────────────────┐
│ L_uniform = L_uni_q + L_uni_t            │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 6: TOTAL LOSS
┌───────────────────────────────────────────┐
│ gamma = directau_gamma (typically 1.0)    │
│ L_total = L_align + gamma * L_uniform     │
└───────────────────────────────────────────┘
Output: scalar loss
```

## 4. SimKGC Loss Computation (Detailed)

```
INPUT: q_batch [B × 768], t_batch [B × 768]
       (implicitly L2-normalized)
       batch_exs, pre_batch_buffer

═════════════════════════════════════════════

STEP 1: IN-BATCH LOGITS
┌───────────────────────────────────────────┐
│ logits = q_batch @ t_batch.T              │
│ # Size: [B × B]                           │
│ # logits[i,j] = q_i · t_j (dot product)  │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 2: TEMPERATURE SCALING
┌───────────────────────────────────────────┐
│ inv_t = exp(log_inv_t)                    │
│ logits *= inv_t                           │
│ # Makes distribution sharper/softer      │
│ # inv_t > 1 → sharper, < 1 → softer     │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 3: ADDITIVE MARGIN (if training)
┌───────────────────────────────────────────┐
│ margin_mask = eye(B)                      │
│ logits[margin_mask] -= margin_value       │
│ # e.g., logits[diag] -= 0.02              │
│ # Makes positive pairs harder to classify│
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 4: PRE-BATCH NEGATIVES (if pre_batch > 0)
┌───────────────────────────────────────────┐
│ pre_batch_logits = q_batch @              │
│                    pre_batch_buffer.T     │
│ # Size: [B × pre_batch_size]              │
│                                           │
│ # Scale by weight                         │
│ pre_batch_logits *= inv_t * pre_batch_wt │
│                                           │
│ # Concatenate with in-batch               │
│ logits = cat([logits, pre_batch_logits],  │
│             dim=1)                        │
│ # Size: [B × (B + pre_batch_size)]        │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 5: SELF-NEGATIVES (if use_self_negative)
┌───────────────────────────────────────────┐
│ head_batch = encode_tail(head_ids)        │
│ head_batch = L2_normalize(head_batch)     │
│                                           │
│ self_neg_logits = sum(q_batch * head, -1)│
│ self_neg_logits *= inv_t                  │
│ self_neg_logits = self_neg_logits         │
│                   .unsqueeze(1)           │
│ # Size: [B × 1]                           │
│                                           │
│ logits = cat([logits, self_neg_logits],   │
│             dim=1)                        │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 6: TRIPLET MASKING
┌───────────────────────────────────────────┐
│ # Build mask: True=valid, False=training │
│ triplet_mask =                            │
│   construct_mask(current_batch,           │
│                  pre_batch_exs)           │
│ # Size: [B × total_negatives]             │
│                                           │
│ # Apply mask: -1e4 for training triplets  │
│ logits.masked_fill_(~triplet_mask, -1e4) │
│ # These will be ignored in softmax        │
└───────────────────────────────────────────┘

═════════════════════════════════════════════

STEP 7: INFONCE LOSS
┌───────────────────────────────────────────┐
│ labels = [0, 1, 2, ..., B-1]              │
│ # Labels point to diagonal (positive)    │
│                                           │
│ loss = CrossEntropyLoss(logits, labels)   │
│ # Equivalent to:                          │
│ # L = -log(exp(logits[i,i]) /             │
│ #          sum_j exp(logits[i,j]))        │
│ #                                         │
│ # Learns to match q_i with t_i (label i) │
│ # while pushing q_i away from t_j (j≠i)  │
└───────────────────────────────────────────┘
Output: scalar loss (typically 1.0-10.0)

═════════════════════════════════════════════

STEP 8: UPDATE PRE-BATCH BUFFER
┌───────────────────────────────────────────┐
│ # Circular replacement                    │
│ offset = current_offset                   │
│ pre_batch_buffer[offset:offset+B] =       │
│   t_batch.detach()                        │
│ pre_batch_exs[offset:offset+B] =          │
│   batch_exs                               │
│                                           │
│ offset = (offset + B) % buffer_size       │
└───────────────────────────────────────────┘
```

## 5. Inference Pipeline: Mode-Specific Ranking

```
Given: Test triple (h_test, r_test, t_test)

PRE-COMPUTATION (shared, done once):
┌─────────────────────────────────┐
│ For each entity e in KB:         │
│   ent_emb[e] = encode_tail(e)   │
│ entity_matrix = stack all        │
│ Shape: [n_entity, 768]           │
│                                  │
│ IF directau:                     │
│   entity_matrix = L2_normalize() │
│   explicit normalization         │
└─────────────────────────────────┘

╔══════════════════════════╗  ╔════════════════════════╗
║  DIRECTAU INFERENCE      ║  ║  SIMKGC INFERENCE      ║
╠══════════════════════════╣  ╠════════════════════════╣
║ TAIL PREDICTION:         ║  ║ TAIL PREDICTION:       ║
║ q = encode_query(h, r)   ║  ║ q = encode_query(h, r) ║
║ q = L2_normalize(q)      ║  ║                        ║
║ scores = q @ matrix.T    ║  ║ scores = -(q @         ║
║ (higher = better)        ║  ║           matrix.T)    ║
║                          ║  ║ (lower = better)       ║
║ rank = argsort(scores,   ║  ║ rank = argsort(scores) ║
║                desc=True)║  ║                        ║
║ target_rank =            ║  ║ target_rank =          ║
║   position(t in rank)+1  ║  ║   position(t in rank)+1║
║                          ║  ║                        ║
║ HEAD PREDICTION:         ║  ║ HEAD PREDICTION:       ║
║ inv_r = "inverse_" + r   ║  ║ inv_r = "inverse_" + r ║
║ q_h = encode_query(t,    ║  ║ q_h = encode_query(t,  ║
║                   inv_r) ║  ║                 inv_r) ║
║ q_h = L2_normalize(q_h)  ║  ║ scores = -(q_h @       ║
║ scores = q_h @ matrix.T  ║  ║           matrix.T)    ║
║ rank = argsort(scores,   ║  ║ rank = argsort(scores) ║
║                desc=True)║  ║                        ║
║ target_rank =            ║  ║ target_rank =          ║
║   position(h in rank)+1  ║  ║   position(h in rank)+1║
╚══════════════════════════╝  ╚════════════════════════╝
        ↓                              ↓
        └──────────────┬───────────────┘
                       ↓
        ┌──────────────────────────┐
        │  APPLY FILTERING (opt)   │
        │ Set invalid entities to  │
        │ -∞ (push to bottom of    │
        │ ranking)                 │
        │ Recompute ranks          │
        └──────────────────────────┘
                ↓
        ┌──────────────────────────┐
        │  COMPUTE METRICS         │
        │  MR, MRR, Hits@K         │
        └──────────────────────────┘
```

## 6. Complete Training vs Inference Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    START EXPERIMENT                         │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA LOADING & PREPARATION (shared)               │
│  • Load entity/relation descriptions                        │
│  • Create ID mappings                                       │
│  • Tokenize descriptions                                    │
│  • Prepare train/valid/test splits                          │
│  • IF SimKGC: Construct triplet masks                       │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: MODE SELECTION & INITIALIZATION                   │
│                                                              │
│  IF --directau:                                             │
│   • Load DirectAU loss module                               │
│   • Gamma = --directau-gamma (1.0)                          │
│   • Eps = --directau-eps (1e-12)                            │
│  ELSE (SimKGC):                                             │
│   • Use InfoNCE loss                                        │
│   • Initialize pre-batch buffer                            │
│   • Temperature parameter: t (0.05)                         │
│   • Additive margin: (0.02)                                 │
│                                                              │
│  • Load pre-trained BERT (HR & Tail)                        │
│  • Initialize optimizer (AdamW)                             │
│  • Setup AMP scaler                                         │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: TRAINING LOOP (Multiple Epochs)                  │
│                                                              │
│  FOR each epoch:                                            │
│   ├─ Shuffle train data                                     │
│   ├─ FOR each batch:                                        │
│   │   ├─ Encode (h,r) and t                                 │
│   │   │                                                     │
│   │   ├─ IF --directau:                                     │
│   │   │   • Compute Align Loss                              │
│   │   │   • Compute Uniform Loss                            │
│   │   │   • L = Align + gamma * Uniform                     │
│   │   │ ELSE:                                               │
│   │   │   • Compute InfoNCE Loss                            │
│   │   │   • (with all strategies)                           │
│   │   │                                                     │
│   │   ├─ Backward & optimize (AMP)                          │
│   │   └─ Log loss metrics                                   │
│   │                                                          │
│   ├─ Validation (link prediction, triple classification)    │
│   ├─ IF valid_perf > best:                                  │
│   │   └─ Save checkpoint                                    │
│   ├─ IF early_stop_patience exceeded:                       │
│   │   └─ Break                                              │
│   └─ Test evaluation (periodic)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: LOAD BEST MODEL                                   │
│  • Load weights from best validation checkpoint             │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: FINAL INFERENCE (Mode-Specific)                   │
│                                                              │
│  Pre-compute: Encode all entities                           │
│                                                              │
│  FOR each test triple:                                      │
│   ├─ Tail prediction (mode-specific ranking)                │
│   ├─ Head prediction (with inverse relation)                │
│   ├─ Apply filtering (optional)                             │
│   └─ Compute MR, MRR, Hits@K                                │
│                                                              │
│  Final Results:                                             │
│   • Mean MR (average rank)                                  │
│   • Mean MRR (average reciprocal rank)                      │
│   • Hits@1, Hits@3, Hits@10 (% in top-K)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                  END EXPERIMENT                             │
└─────────────────────────────────────────────────────────────┘
```

## 7. Decision Tree: Mode Selection Impact

```
                    MODE SELECTION
                          │
                ┌─────────┴─────────┐
                │                   │
            --directau          (default)
                │                   │
          ┌─────┴──────┐      ┌─────┴──────┐
          │            │      │            │
    Normalize      Extract    Temperature  Pre-batch
    Explicitly    Unique      Scaling      Negatives
          │            │      │            │
    ┌─────┴──────┐    │      │    ┌───────┴───────┐
    │            │    │      │    │               │
  Alignment    Uniformity   InfoNCE      Buffer
    Loss         Loss        Loss       Management
    │            │           │           │
    └────┬───────┘           └────┬──────┘
         │                        │
    Total DirectAU         Total SimKGC
    Loss Function           Loss Function
```

---

## Key Mathematical Notation (Integrated)

| Symbol | Meaning | Context |
|--------|---------|---------|
| $q$ | Query embedding (h,r pair) | Both modes |
| $t$ | Tail entity embedding | Both modes |
| $\tau = \exp(\text{log\_inv\_t})$ | Temperature parameter | SimKGC only |
| $\gamma$ | Uniformity weight | DirectAU only |
| $\|\cdot\|_2^2$ | Squared L2 distance | DirectAU align loss |
| $\times$ | Dot product similarity | Both modes |
| $\mathcal{M}$ | Triplet mask | SimKGC only |
| $B$ | Batch size | Both modes |
| $d$ | Embedding dimension | Both modes |
| $n$ | Number of entities | Both modes |

---

## Implementation Notes

1. **Shared Infrastructure**: Both modes use same tokenizer, encoders, and data pipeline
2. **Normalization Timing**: 
   - DirectAU: Explicit after pooling
   - SimKGC: Implicit during pooling (L2-normalize)
3. **Inference Scoring**: 
   - DirectAU: Higher score = better (direct similarity)
   - SimKGC: Lower score = better (negative ranking loss)
4. **Pre-batch Buffer**: Only used in SimKGC mode
5. **Triplet Mask**: Only used in SimKGC mode
6. **Unique Deduplication**: Only used in DirectAU mode (for uniformity)
7. **Temperature Tuning**: Only learnable in SimKGC mode

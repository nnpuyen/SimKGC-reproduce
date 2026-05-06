# SimKGC with DirectAU Loss: Pseudocode Reference

## Table of Contents
1. [Initialization](#initialization)
2. [Training](#training)
3. [Validation](#validation)
4. [Testing/Inference](#testing)
5. [Loss Functions](#loss-functions)

---

## Initialization

```python
# ============================================================================
# FUNCTION: Initialize DirectAU Model
# ============================================================================

FUNCTION initialize_model(config):
    """
    Initialize bi-encoder model with DirectAU loss.
    
    Args:
        config: Configuration object with model hyperparameters
    
    Returns:
        Dictionary with initialized components
    """
    
    // Load and index knowledge graph
    entity_index = index_entities(config.train_path, config.valid_path, config.test_path)
    relation_index = index_relations(config.train_path, config.valid_path, config.test_path)
    n_entity = len(entity_index)
    n_relation = len(relation_index)
    
    // Load text descriptions
    entity_texts = load_entity_descriptions(entity_index)           // [n_entity]
    relation_texts = load_relation_descriptions(relation_index)     // [n_relation]
    
    // Initialize tokenizer and encoders
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    
    // Two separate BERT encoders: one for (head, relation), one for tails
    hr_encoder = AutoModel.from_pretrained(config.pretrained_model)
    tail_encoder = deepcopy(hr_encoder)                             // Same init, different weights
    
    // Move to device
    hr_encoder = hr_encoder.to(device)
    tail_encoder = tail_encoder.to(device)
    
    // DirectAU loss module
    loss_module = DirectAULoss(
        gamma=config.directau_gamma,
        eps=config.directau_eps
    )
    
    // Optimizer (Adam with weight decay)
    trainable_params = (
        list(hr_encoder.parameters()) + 
        list(tail_encoder.parameters())
    )
    optimizer = AdamW(
        params=trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    // Learning rate scheduler with warmup
    lr_scheduler = WarmupLinearSchedule(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_training_steps
    )
    
    // Automatic Mixed Precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    RETURN {
        'hr_encoder': hr_encoder,
        'tail_encoder': tail_encoder,
        'tokenizer': tokenizer,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'scaler': scaler,
        'loss_module': loss_module,
        'entity_texts': entity_texts,
        'relation_texts': relation_texts,
        'n_entity': n_entity,
        'n_relation': n_relation
    }

END FUNCTION
```

---

## Training

```python
# ============================================================================
# FUNCTION: Train DirectAU Model
# ============================================================================

FUNCTION train_one_epoch(model_dict, train_loader, device, config):
    """
    Train for one epoch using DirectAU loss.
    
    Args:
        model_dict: Initialized model components
        train_loader: DataLoader for training triples
        device: GPU/CPU device
        config: Configuration object
    
    Returns:
        epoch_loss: Average loss across all batches
    """
    
    // Extract components
    hr_encoder = model_dict['hr_encoder']
    tail_encoder = model_dict['tail_encoder']
    optimizer = model_dict['optimizer']
    scaler = model_dict['scaler']
    loss_module = model_dict['loss_module']
    
    // Set to training mode
    hr_encoder.train()
    tail_encoder.train()
    
    // Initialize epoch tracking
    epoch_loss = 0.0
    num_batches = 0
    
    // Training loop
    FOR EACH batch IN train_loader:
        
        // Extract batch data
        h_ids, r_ids, t_ids = batch['head_ids'], batch['rel_ids'], batch['tail_ids']
        batch_exs = batch['examples']  // List[Example] with entity IDs
        
        // Get text descriptions
        h_texts = [entity_texts[h] for h in h_ids]
        r_texts = [relation_texts[r] for r in r_ids]
        t_texts = [entity_texts[t] for t in t_ids]
        
        // Tokenize
        h_tokens = tokenizer(h_texts, max_length=128, padding=True, truncation=True, return_tensors='pt')
        r_tokens = tokenizer(r_texts, max_length=128, padding=True, truncation=True, return_tensors='pt')
        t_tokens = tokenizer(t_texts, max_length=128, padding=True, truncation=True, return_tensors='pt')
        
        // Move to device
        h_tokens = move_to_device(h_tokens, device)
        r_tokens = move_to_device(r_tokens, device)
        t_tokens = move_to_device(t_tokens, device)
        
        // AUTOCAST + FORWARD PASS
        WITH torch.autocast(device_type='cuda', dtype=torch.float16) IF config.use_amp ELSE no_autocast():
            
            // Encode queries (head, relation)
            h_emb = hr_encoder(**h_tokens).last_hidden_state[:, 0, :]    // [B, 768] CLS token
            r_emb = hr_encoder(**r_tokens).last_hidden_state[:, 0, :]    // [B, 768] CLS token
            
            q_batch = L2_normalize(h_emb + r_emb)                        // [B, 768]
            
            // Encode tails
            t_batch_raw = tail_encoder(**t_tokens).last_hidden_state[:, 0, :]  // [B, 768]
            t_batch = L2_normalize(t_batch_raw)                          // [B, 768]
            
            // Compute DirectAU loss
            loss = loss_module(q_batch, t_batch, batch_exs)
        
        // BACKWARD PASS
        IF config.use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(hr_encoder.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(tail_encoder.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        ELSE:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hr_encoder.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(tail_encoder.parameters(), max_norm=10.0)
            optimizer.step()
        
        // Zero gradients
        optimizer.zero_grad()
        
        // Accumulate loss
        epoch_loss += loss.item()
        num_batches += 1
    
    // Return average loss
    RETURN epoch_loss / num_batches

END FUNCTION


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

FUNCTION main_training(config):
    """
    Main training loop with early stopping.
    """
    
    // Initialize
    model_dict = initialize_model(config)
    train_loader = create_dataloader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_data, batch_size=config.eval_batch_size, shuffle=False)
    
    // Early stopping tracking
    best_valid_loss = INFINITY
    patience_counter = 0
    best_checkpoint = None
    
    // Training epochs
    FOR epoch IN RANGE(config.num_epochs):
        
        // Train
        train_loss = train_one_epoch(model_dict, train_loader, device, config)
        
        // Validate
        valid_loss = validate(model_dict, valid_loader, device, config)
        
        // Learning rate scheduling
        model_dict['lr_scheduler'].step()
        
        // Early stopping check
        IF valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_checkpoint = deepcopy(model_dict)
            patience_counter = 0
            save_checkpoint(model_dict, optimizer, epoch, best_valid_loss)
            LOG "Best model updated at epoch {epoch}"
        ELSE:
            patience_counter += 1
        
        LOG "Epoch {epoch}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}"
        
        // Early stopping
        IF patience_counter >= config.early_stop_patience:
            LOG "Early stopping at epoch {epoch}"
            BREAK
    
    // Load best checkpoint
    model_dict = best_checkpoint
    
    RETURN model_dict

END FUNCTION
```

---

## Validation

```python
# ============================================================================
# FUNCTION: Validate DirectAU Model
# ============================================================================

FUNCTION validate(model_dict, valid_loader, device, config):
    """
    Compute validation loss (same as training loss computation).
    """
    
    // Set to eval mode
    model_dict['hr_encoder'].eval()
    model_dict['tail_encoder'].eval()
    
    valid_loss = 0.0
    num_batches = 0
    
    WITH torch.no_grad():
        FOR EACH batch IN valid_loader:
            
            // Encode
            h_texts, r_texts, t_texts = extract_texts(batch)
            h_tokens = tokenize_and_move(h_texts, device)
            r_tokens = tokenize_and_move(r_texts, device)
            t_tokens = tokenize_and_move(t_texts, device)
            
            // Forward pass
            h_emb = model_dict['hr_encoder'](**h_tokens).last_hidden_state[:, 0, :]
            r_emb = model_dict['hr_encoder'](**r_tokens).last_hidden_state[:, 0, :]
            q_batch = L2_normalize(h_emb + r_emb)
            
            t_batch_raw = model_dict['tail_encoder'](**t_tokens).last_hidden_state[:, 0, :]
            t_batch = L2_normalize(t_batch_raw)
            
            // Compute loss
            loss = model_dict['loss_module'](q_batch, t_batch, batch['examples'])
            
            valid_loss += loss.item()
            num_batches += 1
    
    // Set back to train mode
    model_dict['hr_encoder'].train()
    model_dict['tail_encoder'].train()
    
    RETURN valid_loss / num_batches

END FUNCTION
```

---

## Testing / Inference

```python
# ============================================================================
# FUNCTION: Inference - Link Prediction
# ============================================================================

FUNCTION link_prediction_inference(model_dict, test_triples, device, config):
    """
    Perform link prediction ranking for test triples.
    """
    
    // Pre-compute entity embeddings
    entity_embeddings = EMPTY LIST
    
    FOR chunk IN chunks(all_entity_texts, chunk_size=config.chunk_size):
        
        // Encode entities
        entity_tokens = tokenizer(chunk, max_length=128, padding=True, truncation=True, return_tensors='pt')
        entity_tokens = move_to_device(entity_tokens, device)
        
        WITH torch.no_grad():
            entity_embs = model_dict['tail_encoder'](**entity_tokens).last_hidden_state[:, 0, :]
            entity_embs = L2_normalize(entity_embs)
        
        entity_embeddings.APPEND(entity_embs)
    
    entity_embeddings = torch.cat(entity_embeddings, dim=0)  // [n_entities, 768]
    
    // Ranking for each test triple
    tail_ranks = EMPTY LIST
    head_ranks = EMPTY LIST
    
    FOR EACH (h, r, t) IN test_triples:
        
        // TAIL PREDICTION: (h, r, ?)
        h_text = entity_texts[h]
        r_text = relation_texts[r]
        hr_tokens = tokenizer(
            [h_text + " " + r_text],
            max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        hr_tokens = move_to_device(hr_tokens, device)
        
        WITH torch.no_grad():
            q_tail = model_dict['hr_encoder'](**hr_tokens).last_hidden_state[0, 0, :]
            q_tail = L2_normalize(q_tail.unsqueeze(0))  // [1, 768]
            
            // Score all entities
            scores = q_tail @ entity_embeddings.T  // [1, n_entities]
            scores = scores.squeeze()  // [n_entities]
        
        // Rank by descending score (higher = better on unit sphere)
        ranked_indices = argsort(scores, descending=True)
        
        // Find true entity rank
        true_rank = index_of(t, ranked_indices) + 1
        tail_ranks.APPEND(true_rank)
        
        // HEAD PREDICTION: (?, r, t)
        r_inv_text = "inverse_" + relation_texts[r]
        t_text = entity_texts[t]
        tr_tokens = tokenizer(
            [t_text + " " + r_inv_text],
            max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        tr_tokens = move_to_device(tr_tokens, device)
        
        WITH torch.no_grad():
            q_head = model_dict['hr_encoder'](**tr_tokens).last_hidden_state[0, 0, :]
            q_head = L2_normalize(q_head.unsqueeze(0))  // [1, 768]
            
            scores = q_head @ entity_embeddings.T
            scores = scores.squeeze()
        
        ranked_indices = argsort(scores, descending=True)
        true_rank = index_of(h, ranked_indices) + 1
        head_ranks.APPEND(true_rank)
    
    // Compute metrics
    all_ranks = tail_ranks + head_ranks
    MR = mean(all_ranks)
    MRR = mean([1.0/r for r in all_ranks])
    Hits@1 = count(r == 1 for r in all_ranks) / len(all_ranks)
    Hits@3 = count(r <= 3 for r in all_ranks) / len(all_ranks)
    Hits@10 = count(r <= 10 for r in all_ranks) / len(all_ranks)
    
    RETURN {
        'MR': MR,
        'MRR': MRR,
        'Hits@1': Hits@1,
        'Hits@3': Hits@3,
        'Hits@10': Hits@10,
        'tail_ranks': tail_ranks,
        'head_ranks': head_ranks
    }

END FUNCTION


# ============================================================================
# FUNCTION: Inference - Triple Classification
# ============================================================================

FUNCTION triple_classification_inference(model_dict, test_triples_labeled, device, config):
    """
    Classify triples as valid (1) or invalid (0).
    """
    
    predictions = EMPTY LIST
    
    FOR EACH (h, r, t, true_label) IN test_triples_labeled:
        
        h_text = entity_texts[h]
        r_text = relation_texts[r]
        t_text = entity_texts[t]
        
        // Encode query
        hr_tokens = tokenizer(
            [h_text + " " + r_text],
            max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        hr_tokens = move_to_device(hr_tokens, device)
        
        // Encode tail
        t_tokens = tokenizer(
            [t_text],
            max_length=128, padding=True, truncation=True, return_tensors='pt'
        )
        t_tokens = move_to_device(t_tokens, device)
        
        WITH torch.no_grad():
            q = model_dict['hr_encoder'](**hr_tokens).last_hidden_state[0, 0, :]
            q = L2_normalize(q.unsqueeze(0))  // [1, 768]
            
            t_emb = model_dict['tail_encoder'](**t_tokens).last_hidden_state[0, 0, :]
            t_emb = L2_normalize(t_emb.unsqueeze(0))  // [1, 768]
            
            // Similarity score (dot product on sphere)
            score = (q @ t_emb.T).item()  // scalar in [-2, 2]
        
        predictions.APPEND({
            'triple': (h, r, t),
            'score': score,
            'true_label': true_label
        })
    
    // Find optimal threshold on validation set
    best_threshold = find_optimal_threshold(predictions, metric='F1')
    
    // Classify using threshold
    for pred IN predictions:
        pred['pred_label'] = 1 IF pred['score'] >= best_threshold ELSE 0
    
    // Compute metrics
    accuracy = count(p['pred_label'] == p['true_label']) / len(predictions)
    precision, recall, f1 = compute_classification_metrics(predictions)
    auc = compute_roc_auc([p['score'] for p in predictions], 
                          [p['true_label'] for p in predictions])
    
    RETURN {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': best_threshold
    }

END FUNCTION
```

---

## Loss Functions

```python
# ============================================================================
# CLASS: DirectAULoss
# ============================================================================

CLASS DirectAULoss(Module):
    """
    Combined Alignment + Uniformity loss for DirectAU training.
    """
    
    FUNCTION __init__(gamma=1.0, eps=1e-12):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    
    FUNCTION forward(q_batch, t_batch, batch_exs):
        """
        Args:
            q_batch: [batch_size, 768], L2-normalized query embeddings
            t_batch: [batch_size, 768], L2-normalized tail embeddings
            batch_exs: List[Example] with head_id, relation, tail_id for uniqueness
        
        Returns:
            loss: scalar
        """
        
        // ALIGNMENT LOSS
        diff = q_batch - t_batch  // [B, 768]
        squared_l2_dist = torch.sum(diff ** 2, dim=-1)  // [B]
        loss_align = torch.mean(squared_l2_dist)
        
        // UNIQUENESS EXTRACTION
        query_keys = [(ex.head_id, ex.relation) for ex in batch_exs]
        q_unique_idx = get_unique_indices(query_keys)
        q_unique = q_batch[q_unique_idx]
        
        tail_ids = [ex.tail_id for ex in batch_exs]
        t_unique_idx = get_unique_indices(tail_ids)
        t_unique = t_batch[t_unique_idx]
        
        // UNIFORMITY LOSS
        loss_uni_q = self._uniformity_loss(q_unique)
        loss_uni_t = self._uniformity_loss(t_unique)
        loss_uniform = loss_uni_q + loss_uni_t
        
        // TOTAL LOSS
        loss = loss_align + self.gamma * loss_uniform
        
        RETURN loss
    
    FUNCTION _uniformity_loss(vectors):
        """
        Compute uniformity loss for a set of vectors on unit sphere.
        """
        
        n = vectors.size(0)
        IF n < 2:
            RETURN torch.tensor(0.0, device=vectors.device)
        
        // Pairwise distances
        dist = torch.cdist(vectors, vectors, p=2)  // [n, n]
        
        // Exclude diagonal
        mask = ~torch.eye(n, dtype=torch.bool, device=vectors.device)
        valid_dist = dist[mask]
        
        // Exponential kernel
        exp_dist = torch.exp(-2 * (valid_dist ** 2))
        mean_exp = torch.mean(exp_dist)
        
        // Log-mean
        loss = torch.log(mean_exp + self.eps)
        
        RETURN loss

END CLASS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

FUNCTION L2_normalize(vectors, eps=1e-12):
    """
    Project vectors onto unit hypersphere.
    """
    norm = torch.sqrt(torch.sum(vectors ** 2, dim=-1, keepdim=True) + eps)
    RETURN vectors / norm

END FUNCTION


FUNCTION get_unique_indices(ids):
    """
    Get indices of first occurrence of each unique ID.
    """
    seen = {}
    unique_idx = []
    FOR i, id_val IN ENUMERATE(ids):
        IF id_val NOT IN seen:
            seen[id_val] = i
            unique_idx.APPEND(i)
    RETURN unique_idx

END FUNCTION


FUNCTION find_optimal_threshold(predictions, metric='F1'):
    """
    Find optimal classification threshold.
    """
    scores = [p['score'] for p in predictions]
    labels = [p['true_label'] for p in predictions]
    
    best_threshold = 0.0
    best_metric = -INFINITY
    
    FOR threshold IN range(-2.0, 2.0, 0.01):
        pred_labels = [1 IF s >= threshold ELSE 0 for s in scores]
        current_metric = compute_metric(pred_labels, labels, metric)
        
        IF current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
    
    RETURN best_threshold

END FUNCTION
```
        'n_relation': n_relation,
    }

END FUNCTION
```

---

## Training

```python
# ============================================================================
# FUNCTION: Text Encoding (Shared)
# ============================================================================

FUNCTION encode_text(encoder, token_ids, attention_mask, token_type_ids, pooling='cls'):
    """Encode text through BERT, return pooled and L2-normalized (for SimKGC)"""
    
    outputs = encoder(
        input_ids=token_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True
    )
    
    last_hidden_state = outputs.last_hidden_state
    
    // Pooling
    IF pooling == 'cls':
        pooled = last_hidden_state[:, 0, :]
    ELSE IF pooling == 'mean':
        mask = attention_mask.unsqueeze(-1)
        sum_embeddings = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_embeddings / denom
    ELSE IF pooling == 'max':
        masked_hidden = last_hidden_state.masked_fill(attention_mask==0, -inf)
        pooled = torch.max(masked_hidden, dim=1)[0]
    
    // L2 normalization (implicit for SimKGC via F.normalize)
    normalized = F.normalize(pooled, p=2, dim=-1)
    
    RETURN normalized

END FUNCTION


# ============================================================================
# FUNCTION: Encode Query (Shared)
# ============================================================================

FUNCTION encode_query(head_ids, relation_ids, encoder, tokenizer, 
                     entity_texts, relation_texts, config):
    """Encode (head, relation) pair"""
    
    head_texts = [entity_texts[i] for i in head_ids]
    rel_texts = [relation_texts[i] for i in relation_ids]
    
    encoded = tokenizer(
        head_texts, rel_texts,
        padding=True, truncation=True,
        max_length=config.max_length,
        return_tensors='pt'
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    query_emb = encode_text(encoder, encoded['input_ids'],
                           encoded['attention_mask'],
                           encoded.get('token_type_ids'), config.pooling)
    
    RETURN query_emb  // [batch_size, 768]

END FUNCTION


# ============================================================================
# FUNCTION: Encode Tail (Shared)
# ============================================================================

FUNCTION encode_tail(tail_ids, encoder, tokenizer, entity_texts, config):
    """Encode tail entity descriptions"""
    
    tail_texts = [entity_texts[i] for i in tail_ids]
    
    encoded = tokenizer(
        tail_texts,
        padding=True, truncation=True,
        max_length=config.max_length,
        return_tensors='pt'
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    tail_emb = encode_text(encoder, encoded['input_ids'],
                          encoded['attention_mask'],
                          encoded.get('token_type_ids'), config.pooling)
    
    RETURN tail_emb  // [batch_size, 768]

END FUNCTION


# ============================================================================
# FUNCTION: DirectAU Loss Computation
# ============================================================================

FUNCTION directau_loss_forward(q_batch, t_batch, batch_exs, 
                              gamma=1.0, eps=1e-12):
    """
    Compute DirectAU loss: alignment + uniformity
    Input: q_batch, t_batch already L2-normalized via encode_text
    """
    
    // ALIGNMENT LOSS
    squared_l2_dist = torch.sum((q_batch - t_batch) ** 2, dim=-1)
    loss_align = torch.mean(squared_l2_dist)
    
    // UNIFORMITY LOSS FOR QUERIES
    IF batch_exs IS NOT None:
        query_keys = [(ex.head_id, ex.relation) for ex in batch_exs]
        unique_head_idx = torch.tensor(
            unique_indices_by_id(query_keys), device=q_batch.device
        )
        q_unique = q_batch[unique_head_idx]
    ELSE:
        q_unique = q_batch
    
    IF q_unique.size(0) >= 2:
        // Pairwise distances
        pairwise_dists = torch.cdist(q_unique, q_unique, p=2)
        mask = ~torch.eye(q_unique.size(0), dtype=torch.bool, device=q_batch.device)
        valid_dists = pairwise_dists[mask]
        
        // Exponential kernel and log-mean
        exp_term = torch.exp(-2 * valid_dists ** 2)
        mean_exp = torch.mean(exp_term)
        loss_uni_q = torch.log(mean_exp + eps)
    ELSE:
        loss_uni_q = torch.tensor(0.0, device=q_batch.device)
    
    // UNIFORMITY LOSS FOR TAILS (same as queries)
    IF batch_exs IS NOT None:
        tail_ids = [ex.tail_id for ex in batch_exs]
        unique_tail_idx = torch.tensor(
            unique_indices_by_id(tail_ids), device=t_batch.device
        )
        t_unique = t_batch[unique_tail_idx]
    ELSE:
        t_unique = t_batch
    
    IF t_unique.size(0) >= 2:
        pairwise_dists = torch.cdist(t_unique, t_unique, p=2)
        mask = ~torch.eye(t_unique.size(0), dtype=torch.bool, device=t_batch.device)
        valid_dists = pairwise_dists[mask]
        exp_term = torch.exp(-2 * valid_dists ** 2)
        mean_exp = torch.mean(exp_term)
        loss_uni_t = torch.log(mean_exp + eps)
    ELSE:
        loss_uni_t = torch.tensor(0.0, device=t_batch.device)
    
    // TOTAL LOSS
    loss_uni = loss_uni_q + loss_uni_t
    total_loss = loss_align + gamma * loss_uni
    
    RETURN {
        'loss': total_loss,
        'align_loss': loss_align,
        'uniform_loss': loss_uni,
    }

END FUNCTION


# ============================================================================
# FUNCTION: SimKGC Loss Computation
# ============================================================================

FUNCTION simkgc_loss_forward(q_batch, t_batch, head_batch, batch_exs,
                            log_inv_t, pre_batch_buffer, config, 
                            batch_dict, training_triples_set):
    """
    Compute SimKGC InfoNCE loss with all strategies
    """
    
    batch_size = q_batch.size(0)
    inv_t = torch.exp(log_inv_t)
    
    // 1. IN-BATCH LOGITS
    logits = q_batch @ t_batch.T
    logits *= inv_t
    
    // 2. ADDITIVE MARGIN (if training)
    IF model.training:
        margin_mask = torch.eye(batch_size, dtype=torch.bool, device=q_batch.device)
        logits[margin_mask] -= config.additive_margin
    
    // 3. PRE-BATCH NEGATIVES
    IF config.pre_batch > 0:
        pre_batch_logits = q_batch @ pre_batch_buffer.T
        pre_batch_logits *= inv_t * config.pre_batch_weight
        logits = torch.cat([logits, pre_batch_logits], dim=1)
    
    // 4. SELF-NEGATIVES (optional)
    IF config.use_self_negative AND head_batch IS NOT None:
        self_neg_logits = torch.sum(q_batch * head_batch, dim=1) * inv_t
        logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=1)
    
    // 5. TRIPLET MASKING
    triplet_mask = batch_dict.get('triplet_mask', None)
    IF triplet_mask IS NOT None:
        logits.masked_fill_(~triplet_mask, -1e4)
    
    // 6. INFONCE LOSS
    labels = torch.arange(batch_size, device=q_batch.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    RETURN loss

END FUNCTION


# ============================================================================
# FUNCTION: Main Training Step (Hybrid)
# ============================================================================

FUNCTION train_one_epoch(train_loader, model, optimizer, scaler, config, mode,
                        training_triples_set):
    """
    Train for one epoch - mode-aware loss computation
    """
    
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    
    // Initialize pre-batch buffer (SimKGC only)
    IF mode == "simkgc":
        pre_batch_offset = 0
    ELSE:
        pre_batch_offset = None
    
    FOR batch_idx, batch_dict IN enumerate(train_loader):
        
        WITH torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.use_amp):
            
            // Extract from batch
            head_ids = batch_dict['head_ids'].to(device)
            relation_ids = batch_dict['relation_ids'].to(device)
            tail_ids = batch_dict['tail_ids'].to(device)
            batch_exs = batch_dict.get('batch_exs', None)
            
            // 1. SHARED ENCODING
            q_batch = encode_query(head_ids, relation_ids, model.hr_encoder, ...)
            t_batch = encode_tail(tail_ids, model.tail_encoder, ...)
            
            // 2. MODE-SPECIFIC NORMALIZATION
            IF mode == "directau":
                // Explicit L2 normalization
                q_batch = F.normalize(q_batch, p=2, dim=-1, eps=config.directau_eps)
                t_batch = F.normalize(t_batch, p=2, dim=-1, eps=config.directau_eps)
            ELSE:  // simkgc
                // Implicit (already normalized via encode_text)
                head_batch = encode_tail(head_ids, model.tail_encoder, ...) 
                                        IF config.use_self_negative ELSE None
            
            // 3. MODE-SPECIFIC LOSS COMPUTATION
            IF mode == "directau":
                loss_dict = directau_loss_forward(
                    q_batch, t_batch, batch_exs,
                    gamma=config.directau_gamma,
                    eps=config.directau_eps
                )
                loss = loss_dict['loss']
            ELSE:  // simkgc
                loss = simkgc_loss_forward(
                    q_batch, t_batch, head_batch, batch_exs,
                    model.log_inv_t, model.pre_batch_buffer, config,
                    batch_dict, training_triples_set
                )
        
        // 4. SHARED BACKWARD PASS
        IF config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        ELSE:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        // 5. MODE-SPECIFIC POST-PROCESSING
        IF mode == "simkgc":
            // Update pre-batch buffer
            model.pre_batch_vectors[offset:(offset+batch_size)] = t_batch.detach()
            model.pre_batch_exs[offset:(offset+batch_size)] = batch_exs
            offset = (offset + batch_size) % len(model.pre_batch_vectors)
        
        epoch_loss += loss.item() * batch_size
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    RETURN avg_loss

END FUNCTION


# ============================================================================
# FUNCTION: Complete Training Loop
# ============================================================================

FUNCTION train_model(train_loader, valid_loader, test_loader, model, config, mode):
    """Complete training with validation and early stopping"""
    
    best_valid_perf = 0.0
    patience_counter = 0
    
    FOR epoch FROM 1 TO config.epochs:
        
        // Train one epoch
        avg_loss = train_one_epoch(train_loader, model, config, mode)
        LOG: "Epoch {epoch}, Loss = {avg_loss:.6f}, Mode = {mode}"
        
        // Validation (shared)
        valid_perf = validate(model, valid_loader, config, mode)
        LOG: "Epoch {epoch}, Valid MRR = {valid_perf:.4f}"
        
        // Early stopping
        IF valid_perf > best_valid_perf:
            best_valid_perf = valid_perf
            patience_counter = 0
            model.save_checkpoint()
        ELSE:
            patience_counter += 1
        
        IF patience_counter >= config.early_stop_patience:
            LOG: "Early stopping at epoch {epoch}"
            BREAK
    
    model.load_checkpoint()
    RETURN model

END FUNCTION
```

---

## Validation

```python
# ============================================================================
# FUNCTION: Validation (Mode-Aware)
# ============================================================================

FUNCTION validate(model, valid_loader, config, mode):
    """Evaluate on validation set"""
    
    model.eval()
    WITH torch.no_grad():
        metrics = evaluate_link_prediction(
            model=model,
            data_loader=valid_loader,
            config=config,
            mode=mode
        )
    
    valid_mrr = metrics['mrr']
    LOG: f"Valid MRR = {valid_mrr:.4f}, MR = {metrics['mr']:.1f}"
    
    RETURN valid_mrr

END FUNCTION
```

---

## Testing

```python
# ============================================================================
# FUNCTION: Link Prediction Testing (Mode-Aware)
# ============================================================================

FUNCTION test_link_prediction(model, test_triples, config, mode):
    """
    Evaluate on all test triples with mode-specific scoring
    """
    
    model.eval()
    WITH torch.no_grad():
        
        // PRE-COMPUTE: Encode all entities once
        print(f"Pre-computing entity embeddings (mode={mode})...")
        entity_embeddings_list = []
        
        FOR batch_start FROM 0 TO n_entity STEP config.batch_size:
            batch_end = min(batch_start + config.batch_size, n_entity)
            batch_ids = torch.arange(batch_start, batch_end, device=device)
            batch_ent_emb = encode_tail(batch_ids, model.tail_encoder, ...)
            
            IF mode == "directau":
                batch_ent_emb = F.normalize(batch_ent_emb, p=2, dim=-1, 
                                           eps=config.directau_eps)
            
            entity_embeddings_list.append(batch_ent_emb)
        
        entity_matrix = torch.cat(entity_embeddings_list, dim=0)
        
        // Initialize metrics
        mr_list, mrr_list = [], []
        hits = {1: 0, 3: 0, 10: 0}
        
        // Process test set
        FOR (head, relation, tail) IN test_triples:
            
            // TAIL PREDICTION
            q_tail = encode_query(head, relation, model.hr_encoder, ...)
            
            IF mode == "directau":
                q_tail = F.normalize(q_tail, p=2, dim=-1)
                tail_scores = q_tail @ entity_matrix.T  // higher = better
                tail_rank = rank_position(tail_scores, tail, descending=True)
            ELSE:  // simkgc
                tail_scores = -(q_tail @ entity_matrix.T)  // lower = better
                tail_rank = rank_position(tail_scores, tail, descending=False)
            
            mr_list.append(tail_rank)
            mrr_list.append(1.0 / tail_rank)
            IF tail_rank <= 10: hits[10] += 1
            IF tail_rank <= 3: hits[3] += 1
            IF tail_rank <= 1: hits[1] += 1
            
            // HEAD PREDICTION
            inv_rel = "inverse_" + get_relation_text(relation)
            q_head = encode_query(tail, inv_rel, model.hr_encoder, ...)
            
            IF mode == "directau":
                q_head = F.normalize(q_head, p=2, dim=-1)
                head_scores = q_head @ entity_matrix.T
                head_rank = rank_position(head_scores, head, descending=True)
            ELSE:  // simkgc
                head_scores = -(q_head @ entity_matrix.T)
                head_rank = rank_position(head_scores, head, descending=False)
            
            mr_list.append(head_rank)
            mrr_list.append(1.0 / head_rank)
            IF head_rank <= 10: hits[10] += 1
            IF head_rank <= 3: hits[3] += 1
            IF head_rank <= 1: hits[1] += 1
        
        // Aggregate
        results = {
            'mr': mean(mr_list),
            'mrr': mean(mrr_list),
            'hits@1': hits[1] / len(mr_list),
            'hits@3': hits[3] / len(mr_list),
            'hits@10': hits[10] / len(mr_list),
        }
        
        LOG: f"Test MRR = {results['mrr']:.4f}, MR = {results['mr']:.1f} (mode={mode})"
    
    RETURN results

END FUNCTION
```

---

## Loss Functions

```python
# ============================================================================
# QUICK REFERENCE: Integrated Loss Functions
# ============================================================================

LOSS FUNCTION directau_total:
    L_total = L_align + gamma * L_uniform
    
    L_align = mean(||q - t||_2^2)
    L_uniform = L_uni_q + L_uni_t
    L_uni_x = log(mean(exp(-2 * pairwise_dist²)))

END FUNCTION


LOSS FUNCTION simkgc_infonce:
    L_infonce = CrossEntropy(logits, labels)
    where loss_i = -log(exp(s_i,i) / sum_j exp(s_i,j))

    logits are built from:
    • In-batch negatives (Q @ T^T)
    • Temperature scaling (inv_t = exp(log_inv_t))
    • Additive margin on the diagonal (training only)
    • Pre-batch negatives (Q @ Buffer^T)
    • Self-negatives (optional)
    • Triplet masking (prevent leakage)
    
    This is the original SimKGC contrastive objective.

END FUNCTION
```

---

## Key Concepts Summary

| Concept | DirectAU | SimKGC |
|---------|----------|--------|
| **Normalization** | Explicit after pooling | Implicit via F.normalize |
| **Similarity** | Direct dot product | Negative for ranking |
| **Loss** | Align + Uniform | InfoNCE Contrastive |
| **Ranking** | Descending (↓) | Ascending (↑) |
| **Temperature** | None (fixed 1.0) | Learnable $\tau$ |
| **Margin** | None | Optional margin |
| **Negatives** | Only in-batch | In-batch + Pre-batch + Self |
| **Masking** | None | Triplet mask |
| **Unique Dedup** | Yes (for uniformity) | Optional |

---

## Eight Strategy Controls

| # | Strategy | Pseudocode hook |
|---|---|---|
| 1 | Neighbor-based context augmentation | `use_link_graph` branch in text preparation |
| 2 | Triplet masking | `construct_mask(...)` during SimKGC-style training |
| 3 | Learning-rate scheduling | `lr_scheduler.step()` in the epoch loop |
| 4 | Gradient accumulation | Scale loss, delay `optimizer.step()` |
| 5 | Pre-batch negatives | `pre_batch_buffer` and `pre_batch_logits` |
| 6 | Mixed precision | `torch.autocast(...)` and `GradScaler` |
| 7 | Weight decay | `AdamW(..., weight_decay=...)` |
| 8 | Fine-tunable temperature | `log_inv_t` when `finetune_t` is enabled |

---

## Three Binary Choices -> Eight Modes

| Binary choice | Off | On |
|---|---|---|
| Loss family | InfoNCE | DirectAU |
| Context augmentation | Plain text only | `--use-link-graph` |
| Execution profile | Standard precision | `--use-amp` |

| Mode | Loss family | Link graph | AMP | Notes |
|---|---|---|---|---|
| 1 | InfoNCE | Off | Off | Base SimKGC path |
| 2 | InfoNCE | Off | On | Same loss, lower precision |
| 3 | InfoNCE | On | Off | Adds 1-hop context |
| 4 | InfoNCE | On | On | Graph context plus AMP |
| 5 | DirectAU | Off | Off | DirectAU baseline |
| 6 | DirectAU | Off | On | DirectAU with AMP |
| 7 | DirectAU | On | Off | DirectAU plus graph context |
| 8 | DirectAU | On | On | Full DirectAU configuration |

InfoNCE-only knobs such as temperature, additive margin, pre-batch negatives, and self-negatives are orthogonal to this 8-mode matrix.

---

## Implementation Workflow Summary

```
1. Load data → Initialize mode-specific components
2. FOR each epoch:
    a. FOR each batch:
        i.   Encode queries and tails (shared)
        ii.  Mode-specific normalization
        iii. Mode-specific loss computation
        iv.  Backward & optimize (shared)
        v.   Mode-specific post-processing
    b. Validate (shared)
    c. Check early stopping
3. Load best checkpoint
4. Evaluate on test set (mode-aware ranking)
```

---

## Command-Line Examples

```bash
# SimKGC mode (default)
python main.py \
    --task wn18rr \
    --pretrained-model distilbert-base-uncased \
    --batch-size 512 \
    --epochs 20 \
    --t 0.05 \
    --additive-margin 0.02 \
    --pre-batch 1 \
    --use-self-negative

# DirectAU mode
python main.py \
    --task wn18rr \
    --pretrained-model distilbert-base-uncased \
    --batch-size 512 \
    --epochs 20 \
    --directau \
    --directau-gamma 1.0 \
    --directau-eps 1e-12
```

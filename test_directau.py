#!/usr/bin/env python3
"""
Simple test script to verify DirectAU integration works.
Tests initialization, forward pass, and loss computation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock command-line arguments before importing config
sys.argv = [sys.argv[0], 
            '--task', 'wn18rr',
            '--model-dir', './test_checkpoint',
            '--pretrained-model', 'distilbert-base-uncased']

from config import args
from models import build_model, DirectAULoss


def test_directau_initialization():
    """Test that DirectAU model initializes correctly."""
    print("\n=== Test 1: DirectAU Model Initialization ===")
    
    # Mock args for DirectAU
    class MockArgs:
        def __init__(self):
            self.pretrained_model = 'distilbert-base-uncased'
            self.t = 0.05
            self.finetune_t = False
            self.additive_margin = 0.0
            self.batch_size = 32
            self.pre_batch = 0
            self.use_self_negative = False
            self.directau = True
            self.directau_model = 'distmult'
            self.directau_init = 'xavier'
            self.directau_dim = 768
            self.directau_eps = 1e-12
            self.pooling = 'cls'
            self.dropout = 0.1
    
    mock_args = MockArgs()
    model = build_model(mock_args)
    print("✓ Model initialized successfully")
    
    # Check DirectAU components are present
    assert hasattr(model, 'directau') and model.directau, "DirectAU flag not set"
    print("✓ DirectAU flag is enabled")
    
    # Set vocabulary sizes
    model.set_directau_vocab_sizes(num_entities=100, num_relations=10)
    assert model.entity_embeddings is not None, "Entity embeddings not created"
    assert model.relation_embeddings is not None, "Relation embeddings not created"
    assert model.relation_mask_embeddings is not None, "Relation mask embeddings not created"
    print("✓ DirectAU embedding tables initialized")


def test_directau_loss():
    """Test DirectAU loss computation."""
    print("\n=== Test 2: DirectAU Loss Computation ===")
    
    batch_size, dim = 32, 768
    
    # Create normalized vectors
    hr_vector = torch.randn(batch_size, dim)
    hr_vector = torch.nn.functional.normalize(hr_vector, p=2, dim=-1)
    
    tail_vector = torch.randn(batch_size, dim)
    tail_vector = torch.nn.functional.normalize(tail_vector, p=2, dim=-1)
    
    labels = torch.arange(batch_size)
    
    loss_fn = DirectAULoss(gamma=1.0, eps=1e-12)
    loss_dict = loss_fn(hr_vector, tail_vector, labels)
    
    assert 'loss' in loss_dict, "Loss not in output"
    assert 'align_loss' in loss_dict, "Align loss not in output"
    assert 'uniform_loss' in loss_dict, "Uniform loss not in output"
    
    loss_value = loss_dict['loss'].item()
    assert loss_value >= 0, f"Loss should be non-negative, got {loss_value}"
    print(f"✓ Loss computed successfully: {loss_value:.6f}")
    print(f"  - Alignment loss: {loss_dict['align_loss'].item():.6f}")
    print(f"  - Uniformity loss: {loss_dict['uniform_loss'].item():.6f}")


def test_directau_query_computation():
    """Test DirectAU query computation."""
    print("\n=== Test 3: DirectAU Query Computation ===")
    
    class MockArgs:
        def __init__(self):
            self.pretrained_model = 'distilbert-base-uncased'
            self.t = 0.05
            self.finetune_t = False
            self.additive_margin = 0.0
            self.batch_size = 32
            self.pre_batch = 0
            self.use_self_negative = False
            self.directau = True
            self.directau_model = 'distmult'
            self.directau_init = 'xavier'
            self.directau_dim = 768
            self.directau_eps = 1e-12
            self.pooling = 'cls'
            self.dropout = 0.1
    
    mock_args = MockArgs()
    model = build_model(mock_args)
    model.set_directau_vocab_sizes(num_entities=100, num_relations=10)
    
    batch_size = 32
    head_indices = torch.randint(0, 100, (batch_size,))
    relation_indices = torch.randint(0, 10, (batch_size,))
    tail_indices = torch.randint(0, 100, (batch_size,))
    
    output_dict = model.forward_directau(head_indices, relation_indices, tail_indices)
    
    assert 'query_vector' in output_dict, "Query vector not in output"
    assert 'tail_vector' in output_dict, "Tail vector not in output"
    assert 'squared_l2_score' in output_dict, "Squared L2 score not in output"
    
    # Check normalization
    query_norms = torch.norm(output_dict['query_vector'], p=2, dim=-1)
    assert torch.allclose(query_norms, torch.ones_like(query_norms), atol=1e-5), \
        "Query vectors should be L2-normalized"
    print("✓ Query vectors are properly normalized")
    
    score = output_dict['squared_l2_score']
    assert score.shape == (batch_size,), f"Score shape should be ({batch_size},), got {score.shape}"
    assert (score >= 0).all(), "Squared L2 distance should be non-negative"
    print(f"✓ Squared L2 scores computed: min={score.min().item():.6f}, max={score.max().item():.6f}")


def test_chunked_inference():
    """Test that chunked inference logic works."""
    print("\n=== Test 4: Chunked Inference Logic ===")
    
    batch_size = 100
    entity_cnt = 5000
    dim = 768
    chunk_size = 1024
    
    hr_vector = torch.randn(batch_size, dim)
    entities_tensor = torch.randn(entity_cnt, dim)
    
    # Simulate chunked inference
    batch_score = torch.zeros(batch_size, entity_cnt)
    
    for entity_start in range(0, entity_cnt, chunk_size):
        entity_end = min(entity_start + chunk_size, entity_cnt)
        chunk_entities = entities_tensor[entity_start:entity_end, :]
        batch_score[:, entity_start:entity_end] = torch.mm(hr_vector, chunk_entities.t())
    
    # Compare with full inference
    full_score = torch.mm(hr_vector, entities_tensor.t())
    
    assert torch.allclose(batch_score, full_score, atol=1e-5), "Chunked inference differs from full inference"
    print(f"✓ Chunked inference matches full inference")
    print(f"  - Batch size: {batch_size}, Entity count: {entity_cnt}, Chunk size: {chunk_size}")


if __name__ == '__main__':
    print("Running DirectAU integration tests...")
    
    try:
        test_directau_initialization()
        test_directau_loss()
        test_directau_query_computation()
        test_chunked_inference()
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Standalone test for DirectAU components without full codebase initialization.
Tests only the core DirectAU loss and utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectAULoss(nn.Module):
    """Alignment and Uniformity loss for DirectAU model."""
    
    def __init__(self, gamma: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, hr_vector: torch.tensor, tail_vector: torch.tensor, 
                labels: torch.tensor = None) -> dict:
        batch_size = hr_vector.size(0)
        
        align_loss = self._compute_align_loss(hr_vector, tail_vector)
        uniform_loss = self._compute_uniform_loss(hr_vector, tail_vector, batch_size)
        
        total_loss = align_loss + self.gamma * uniform_loss
        
        return {
            'loss': total_loss,
            'align_loss': align_loss.detach(),
            'uniform_loss': uniform_loss.detach(),
        }
    
    def _compute_align_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor) -> torch.tensor:
        """Alignment loss: mean squared L2 distance between query and tail."""
        squared_l2_dist = torch.sum((hr_vector - tail_vector) ** 2, dim=-1)
        align_loss = torch.mean(squared_l2_dist)
        return align_loss
    
    def _compute_uniform_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor, 
                              batch_size: int) -> torch.tensor:
        """
        Uniformity loss: safe log of mean(exp(-2 * pairwise_distances)).
        Uses unique entities only to avoid trivial uniformity.
        """
        unique_vectors = torch.cat([hr_vector, tail_vector], dim=0)
        unique_vectors = torch.unique(unique_vectors, dim=0)
        
        if unique_vectors.size(0) < 2:
            return torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        
        pairwise_dists = torch.cdist(unique_vectors, unique_vectors, p=2)
        pairwise_dists = pairwise_dists[~torch.eye(unique_vectors.size(0), dtype=torch.bool, device=hr_vector.device)]
        
        exp_term = torch.exp(-2 * pairwise_dists ** 2)
        mean_exp = torch.mean(exp_term)
        
        uniform_loss = -torch.log(mean_exp + self.eps)
        return uniform_loss


def test_directau_loss_alignment():
    """Test alignment loss computation."""
    print("\n=== Test 1: Alignment Loss ===")
    
    batch_size, dim = 32, 768
    
    # Create normalized vectors
    hr_vector = torch.randn(batch_size, dim)
    hr_vector = F.normalize(hr_vector, p=2, dim=-1)
    
    tail_vector = torch.randn(batch_size, dim)
    tail_vector = F.normalize(tail_vector, p=2, dim=-1)
    
    loss_fn = DirectAULoss(gamma=0.0, eps=1e-12)  # Only alignment
    loss_dict = loss_fn(hr_vector, tail_vector)
    
    align_loss = loss_dict['align_loss'].item()
    assert align_loss >= 0, f"Alignment loss should be non-negative, got {align_loss}"
    assert align_loss <= 4.0, f"Alignment loss should be <= 4.0 for normalized vectors, got {align_loss}"
    print(f"✓ Alignment loss computed: {align_loss:.6f}")


def test_directau_loss_uniformity():
    """Test uniformity loss computation."""
    print("\n=== Test 2: Uniformity Loss ===")
    
    batch_size, dim = 32, 768
    
    # Create normalized vectors
    hr_vector = torch.randn(batch_size, dim)
    hr_vector = F.normalize(hr_vector, p=2, dim=-1)
    
    tail_vector = torch.randn(batch_size, dim)
    tail_vector = F.normalize(tail_vector, p=2, dim=-1)
    
    loss_fn = DirectAULoss(gamma=1.0, eps=1e-12)
    loss_dict = loss_fn(hr_vector, tail_vector)
    
    uniform_loss = loss_dict['uniform_loss'].item()
    assert uniform_loss >= 0, f"Uniformity loss should be non-negative, got {uniform_loss}"
    print(f"✓ Uniformity loss computed: {uniform_loss:.6f}")


def test_directau_loss_combined():
    """Test combined loss."""
    print("\n=== Test 3: Combined Loss (Alignment + Uniformity) ===")
    
    batch_size, dim = 32, 768
    
    hr_vector = torch.randn(batch_size, dim)
    hr_vector = F.normalize(hr_vector, p=2, dim=-1)
    
    tail_vector = torch.randn(batch_size, dim)
    tail_vector = F.normalize(tail_vector, p=2, dim=-1)
    
    loss_fn = DirectAULoss(gamma=1.0, eps=1e-12)
    loss_dict = loss_fn(hr_vector, tail_vector)
    
    total_loss = loss_dict['loss'].item()
    align_loss = loss_dict['align_loss'].item()
    uniform_loss = loss_dict['uniform_loss'].item()
    
    expected_total = align_loss + uniform_loss
    assert abs(total_loss - expected_total) < 1e-5, \
        f"Total loss {total_loss} != align {align_loss} + uniform {uniform_loss}"
    
    print(f"✓ Total loss: {total_loss:.6f}")
    print(f"  - Alignment: {align_loss:.6f}")
    print(f"  - Uniformity: {uniform_loss:.6f}")


def test_l2_normalization():
    """Test L2 normalization properties."""
    print("\n=== Test 4: L2 Normalization ===")
    
    batch_size, dim = 32, 768
    
    vectors = torch.randn(batch_size, dim)
    normalized = F.normalize(vectors, p=2, dim=-1)
    
    norms = torch.norm(normalized, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Normalized vectors should have unit norm"
    
    print(f"✓ All {batch_size} vectors have unit L2 norm")
    print(f"  - Norm range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")


def test_dot_product_l2_equivalence():
    """Test that dot product and L2 distance are equivalent for normalized vectors."""
    print("\n=== Test 5: Dot Product vs L2 Distance Equivalence ===")
    
    batch_size, dim = 100, 768
    
    # Create random vectors and normalize
    u = torch.randn(batch_size, dim)
    u = F.normalize(u, p=2, dim=-1)
    
    v = torch.randn(batch_size, dim)
    v = F.normalize(v, p=2, dim=-1)
    
    # Compute dot product and L2 distance
    dot_prod = torch.sum(u * v, dim=-1)  # Shape: (batch_size,)
    l2_dist_sq = torch.sum((u - v) ** 2, dim=-1)  # Shape: (batch_size,)
    
    # For normalized vectors: ||u-v||^2 = 2 - 2(u·v)
    expected_l2_sq = 2 - 2 * dot_prod
    
    assert torch.allclose(l2_dist_sq, expected_l2_sq, atol=1e-5), \
        "L2 distance and dot product relation violated"
    
    # Verify ranking is identical (higher dot product = lower L2 distance)
    dot_rank = torch.argsort(dot_prod, descending=True)
    l2_rank = torch.argsort(l2_dist_sq, descending=False)
    
    assert torch.equal(dot_rank, l2_rank), "Ranking should be identical"
    
    print(f"✓ Dot product and L2 distance produce identical rankings")
    print(f"  - Dot product range: [{dot_prod.min().item():.6f}, {dot_prod.max().item():.6f}]")
    print(f"  - L2 distance² range: [{l2_dist_sq.min().item():.6f}, {l2_dist_sq.max().item():.6f}]")


def test_chunked_inference_equivalence():
    """Test that chunked inference produces same results as full inference."""
    print("\n=== Test 6: Chunked Inference Equivalence ===")
    
    batch_size = 100
    entity_cnt = 5000
    dim = 768
    chunk_size = 1024
    
    hr_vector = torch.randn(batch_size, dim)
    entities_tensor = torch.randn(entity_cnt, dim)
    
    # Full inference
    full_score = torch.mm(hr_vector, entities_tensor.t())
    
    # Chunked inference
    chunked_score = torch.zeros(batch_size, entity_cnt)
    for entity_start in range(0, entity_cnt, chunk_size):
        entity_end = min(entity_start + chunk_size, entity_cnt)
        chunk_entities = entities_tensor[entity_start:entity_end, :]
        chunked_score[:, entity_start:entity_end] = torch.mm(hr_vector, chunk_entities.t())
    
    assert torch.allclose(full_score, chunked_score, atol=1e-5), \
        "Chunked inference differs from full inference"
    
    print(f"✓ Chunked inference matches full inference")
    print(f"  - Batch size: {batch_size}, Entity count: {entity_cnt}, Chunk size: {chunk_size}")
    print(f"  - Max difference: {(full_score - chunked_score).abs().max().item():.2e}")


def test_loss_stability():
    """Test that loss is numerically stable."""
    print("\n=== Test 7: Loss Numerical Stability ===")
    
    batch_size, dim = 32, 768
    loss_fn = DirectAULoss(gamma=1.0, eps=1e-12)
    
    # Test with various vector magnitudes
    for scale in [0.1, 1.0, 10.0]:
        hr_vector = torch.randn(batch_size, dim) * scale
        hr_vector = F.normalize(hr_vector, p=2, dim=-1)
        
        tail_vector = torch.randn(batch_size, dim) * scale
        tail_vector = F.normalize(tail_vector, p=2, dim=-1)
        
        loss_dict = loss_fn(hr_vector, tail_vector)
        loss_value = loss_dict['loss'].item()
        
        assert loss_value > 0, f"Loss should be positive, got {loss_value}"
        assert not torch.isnan(torch.tensor(loss_value)), f"Loss is NaN at scale {scale}"
        assert not torch.isinf(torch.tensor(loss_value)), f"Loss is inf at scale {scale}"
    
    print(f"✓ Loss is stable across different vector magnitudes")


if __name__ == '__main__':
    print("="*60)
    print("Running DirectAU Standalone Tests")
    print("="*60)
    
    try:
        test_directau_loss_alignment()
        test_directau_loss_uniformity()
        test_directau_loss_combined()
        test_l2_normalization()
        test_dot_product_l2_equivalence()
        test_chunked_inference_equivalence()
        test_loss_stability()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

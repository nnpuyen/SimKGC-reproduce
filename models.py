from abc import ABC
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask

class DirectAULoss(nn.Module):
    """Alignment and Uniformity loss for DirectAU model."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, eps: float = 1e-12,
                 uniformity_scale: float = 4.0, use_alignment: bool = True, use_uniformity: bool = True,
                 use_uniformity_query: bool = True, use_uniformity_tail: bool = True,
                 use_uniformity_head: bool = False, use_uniformity_entity: bool = False,
                 use_uniformity_cross: bool = False, cross_uniformity_beta: float = None,
                 gamma_cross: float = None,
                 learnable_uniformity_scale: bool = False,
                 use_uniformity_as_alignment: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_cross = gamma if gamma_cross is None else gamma_cross
        self.eps = eps
        self.cross_uniformity_beta = cross_uniformity_beta
        # Support an optional learnable uniformity scale via re-parameterization
        # If learnable_uniformity_scale is True, we store log(scale) as an nn.Parameter
        # and expose a `uniformity_scale` property that returns exp(log_scale).
        if learnable_uniformity_scale:
            self.log_uniformity_scale = nn.Parameter(torch.tensor(math.log(float(uniformity_scale))))
        else:
            self._uniformity_scale = float(uniformity_scale)
        self.use_alignment = use_alignment
        self.use_uniformity = use_uniformity
        self.use_uniformity_query = use_uniformity_query
        self.use_uniformity_tail = use_uniformity_tail
        self.use_uniformity_head = use_uniformity_head
        self.use_uniformity_entity = use_uniformity_entity
        self.use_uniformity_cross = use_uniformity_cross
        self.use_uniformity_as_alignment = use_uniformity_as_alignment
    
    def forward(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                labels: torch.tensor = None, batch_exs: list = None,
                head_vector: torch.tensor = None, triplet_mask: torch.tensor = None) -> dict:
        """
        Compute DirectAU loss: alignment and/or uniformity based on flags.
        
        Args:
            hr_vector: query vectors (batch_size, dim), normalized
            tail_vector: tail entity vectors (batch_size, dim), normalized
            labels: optional labels for batch (batch_size,)
            batch_exs: batch examples for deduplication
        
        Returns:
            dict with 'loss', 'align_loss', 'uniform_loss'
        """
        batch_size = hr_vector.size(0)
        
        align_loss = self._compute_align_loss(hr_vector, tail_vector) if self.use_alignment else torch.tensor(0.0, device=hr_vector.device)
        uniform_components = self._compute_uniform_loss(
            hr_vector,
            tail_vector,
            batch_size,
            batch_exs=batch_exs,
            head_vector=head_vector,
            triplet_mask=triplet_mask,
        ) if self.use_uniformity else {
            'total': torch.tensor(0.0, device=hr_vector.device),
            'query': torch.tensor(0.0, device=hr_vector.device),
            'tail': torch.tensor(0.0, device=hr_vector.device),
            'head': torch.tensor(0.0, device=hr_vector.device),
            'entity': torch.tensor(0.0, device=hr_vector.device),
            'cross': torch.tensor(0.0, device=hr_vector.device),
        }
        uniform_loss = uniform_components['total'] + uniform_components['cross']
        # Determine alignment scaling: either fixed `alpha` or (optionally)
        # reuse the uniformity scale as the alignment multiplier.
        if self.use_uniformity_as_alignment:
            align_scale = self.uniformity_scale if hasattr(self, 'log_uniformity_scale') else self.alpha
        else:
            align_scale = self.alpha
        scaled_align = align_scale * align_loss
        scaled_intra = self.gamma * uniform_components['total']
        scaled_cross = self.gamma_cross * uniform_components['cross']
        scaled_uniform = scaled_intra + scaled_cross
        total_loss = scaled_align + scaled_uniform

        return {
            'loss': total_loss,
            'align_loss': align_loss.detach(),
            'align_loss_scaled': scaled_align.detach(),
            'uniform_loss': uniform_loss.detach(),
            'uniform_loss_scaled': scaled_uniform.detach(),
            'uniform_loss_query': uniform_components['query'].detach(),
            'uniform_loss_tail': uniform_components['tail'].detach(),
            'uniform_loss_head': uniform_components['head'].detach(),
            'uniform_loss_entity': uniform_components['entity'].detach(),
            'uniform_loss_cross': uniform_components['cross'].detach(),
            'uniform_loss_query_scaled': (self.gamma * uniform_components['query']).detach(),
            'uniform_loss_tail_scaled': (self.gamma * uniform_components['tail']).detach(),
            'uniform_loss_head_scaled': (self.gamma * uniform_components['head']).detach(),
            'uniform_loss_entity_scaled': (self.gamma * uniform_components['entity']).detach(),
            'uniform_loss_cross_scaled': (self.gamma_cross * uniform_components['cross']).detach(),
        }
    
    def _compute_align_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor) -> torch.tensor:
        """Alignment loss: mean squared L2 distance between query and tail."""
        squared_l2_dist = torch.sum((hr_vector - tail_vector) ** 2, dim=-1)
        align_loss = torch.mean(squared_l2_dist)
        return align_loss
    
    def _compute_uniform_loss_for_vectors(self, vectors: torch.tensor) -> torch.tensor:
        """
        Uniformity loss for a single set of vectors: log of mean(exp(-2 * pairwise_distances)).
        Assumes the input vectors have already been deduplicated or pooled upstream.
        """
        if vectors.size(0) < 2:
            return torch.tensor(0.0, device=vectors.device, dtype=vectors.dtype)

        pairwise_dists = torch.cdist(vectors, vectors, p=2)
        pairwise_mask = ~torch.eye(vectors.size(0), dtype=torch.bool, device=vectors.device)
        pairwise_dists = pairwise_dists[pairwise_mask]

        scale = self.uniformity_scale
        # `scale` may be a python float or a torch tensor; ensure correct dtype for computation
        exp_term = torch.exp(-scale * pairwise_dists ** 2)
        mean_exp = torch.mean(exp_term)

        uniform_loss = torch.log(mean_exp + self.eps)
        return uniform_loss

    @property
    def uniformity_scale(self):
        if hasattr(self, 'log_uniformity_scale'):
            return torch.exp(self.log_uniformity_scale)
        return self._uniformity_scale

    @uniformity_scale.setter
    def uniformity_scale(self, value):
        # Allow external callers (e.g., scheduler) to assign a float value.
        if hasattr(self, 'log_uniformity_scale'):
            with torch.no_grad():
                self.log_uniformity_scale.data.fill_(math.log(float(value)))
        else:
            self._uniformity_scale = float(value)

    def _resolve_cross_uniformity_beta(self):
        if self.cross_uniformity_beta is not None:
            return self.cross_uniformity_beta
        return self.uniformity_scale

    def _compute_cross_uniformity_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                                       triplet_mask: torch.tensor = None) -> torch.tensor:
        """Cross-uniformity: push each query away from non-matching tail entities."""
        if hr_vector.size(0) == 0:
            return torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)

        dist = torch.cdist(hr_vector, tail_vector, p=2)
        dist2 = dist ** 2
        beta = self._resolve_cross_uniformity_beta()
        if torch.is_tensor(beta):
            beta = beta.to(device=hr_vector.device, dtype=hr_vector.dtype)

        if triplet_mask is not None:
            mask = triplet_mask.to(hr_vector.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.clone()
            if mask.shape == dist2.shape:
                mask.fill_diagonal_(False)
            else:
                mask = None
        else:
            mask = None

        if mask is None:
            mask = torch.ones_like(dist2, dtype=torch.bool, device=hr_vector.device)
            mask.fill_diagonal_(False)

        neg_scores = (-beta * dist2).masked_fill(~mask, float('-inf'))
        logsumexp = torch.logsumexp(neg_scores, dim=1)
        num_neg = mask.sum(dim=1).clamp(min=1)
        logmeanexp = logsumexp - torch.log(num_neg.to(dist2.dtype))
        logmeanexp = torch.where(torch.isfinite(logmeanexp), logmeanexp, torch.zeros_like(logmeanexp))
        return torch.mean(logmeanexp)

    def _compute_uniform_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                              batch_size: int, batch_exs: list = None,
                              head_vector: torch.tensor = None,
                              triplet_mask: torch.tensor = None) -> torch.tensor:
        """
        Uniformity loss: compute separately for hr_vector and tail_vector, then sum.
        If `batch_exs` is provided, deduplicate query vectors by the composite key
        (head_id, relation) and tail vectors by tail_id before computing uniformity so that
        repeated query or entity occurrences in a batch are treated as one.
        """
        if batch_exs is not None:
            query_keys = [(ex.head_id, ex.relation) for ex in batch_exs]
            tail_ids = [ex.tail_id for ex in batch_exs]

            def unique_indices_by_id(ids):
                seen = set()
                uniq_idx = []
                for i, idv in enumerate(ids):
                    if idv not in seen:
                        seen.add(idv)
                        uniq_idx.append(i)
                return torch.tensor(uniq_idx, dtype=torch.long, device=hr_vector.device)

            hr_idx = unique_indices_by_id(query_keys)
            tail_idx = unique_indices_by_id(tail_ids)

            hr_unique = hr_vector[hr_idx]
            tail_unique = tail_vector[tail_idx]
        else:
            hr_unique = hr_vector
            tail_unique = tail_vector

        total_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        query_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        tail_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        head_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        entity_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        cross_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)

        if self.use_uniformity_query:
            query_uniform_loss = self._compute_uniform_loss_for_vectors(hr_unique)
            total_uniform_loss = total_uniform_loss + query_uniform_loss
        if self.use_uniformity_tail:
            tail_uniform_loss = self._compute_uniform_loss_for_vectors(tail_unique)
            total_uniform_loss = total_uniform_loss + tail_uniform_loss

        if self.use_uniformity_cross:
            cross_uniform_loss = self._compute_cross_uniformity_loss(hr_vector, tail_vector, triplet_mask)

        if self.use_uniformity_head and head_vector is not None:
            if batch_exs is not None:
                head_ids = [ex.head_id for ex in batch_exs]
                def unique_indices_by_id(ids):
                    seen = set()
                    uniq_idx = []
                    for i, idv in enumerate(ids):
                        if idv not in seen:
                            seen.add(idv)
                            uniq_idx.append(i)
                    return torch.tensor(uniq_idx, dtype=torch.long, device=head_vector.device)
                head_idx = unique_indices_by_id(head_ids)
                head_unique = head_vector[head_idx]
                head_uniform_loss = self._compute_uniform_loss_for_vectors(head_unique)
            else:
                head_uniform_loss = self._compute_uniform_loss_for_vectors(head_vector)
            total_uniform_loss = total_uniform_loss + head_uniform_loss

        if self.use_uniformity_entity and head_vector is not None:
            if batch_exs is not None:
                seen = set()
                entity_vectors = []
                for i, ex in enumerate(batch_exs):
                    if ex.head_id not in seen:
                        seen.add(ex.head_id)
                        entity_vectors.append(head_vector[i])
                    if ex.tail_id not in seen:
                        seen.add(ex.tail_id)
                        entity_vectors.append(tail_vector[i])
                if len(entity_vectors) >= 2:
                    entity_stack = torch.stack(entity_vectors, dim=0)
                    entity_uniform_loss = self._compute_uniform_loss_for_vectors(entity_stack)
                else:
                    entity_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
            else:
                entity_stack = torch.cat([head_vector, tail_vector], dim=0)
                entity_uniform_loss = self._compute_uniform_loss_for_vectors(entity_stack)
            total_uniform_loss = total_uniform_loss + entity_uniform_loss

        return {
            'total': total_uniform_loss,
            'query': query_uniform_loss,
            'tail': tail_uniform_loss,
            'head': head_uniform_loss,
            'entity': entity_uniform_loss,
            'cross': cross_uniform_loss,
        }


class StaticHybridDirectAULoss(nn.Module):
    """Alignment plus two static uniformity terms with separate scales and weights."""

    def __init__(self, alpha: float = 1.0, gamma1: float = 0.5, gamma2: float = 0.5,
                 eps: float = 1e-12, uniformity_scale1: float = 4.0, uniformity_scale2: float = 6.0,
                 use_alignment: bool = True, use_uniformity: bool = True,
                 use_uniformity_query: bool = True, use_uniformity_tail: bool = True,
                 use_uniformity_head: bool = False, use_uniformity_entity: bool = False,
                 use_uniformity_cross: bool = False, cross_uniformity_beta: float = None,
                 gamma_cross: float = None):
        super().__init__()
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        default_gamma_cross = (gamma1 + gamma2) / 2.0
        self.gamma_cross = default_gamma_cross if gamma_cross is None else gamma_cross
        self.eps = eps
        self.uniformity_scale1 = uniformity_scale1
        self.uniformity_scale2 = uniformity_scale2
        self.cross_uniformity_beta = cross_uniformity_beta
        self.use_alignment = use_alignment
        self.use_uniformity = use_uniformity
        self.use_uniformity_query = use_uniformity_query
        self.use_uniformity_tail = use_uniformity_tail
        self.use_uniformity_head = use_uniformity_head
        self.use_uniformity_entity = use_uniformity_entity
        self.use_uniformity_cross = use_uniformity_cross

    def forward(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                labels: torch.tensor = None, batch_exs: list = None,
                head_vector: torch.tensor = None, triplet_mask: torch.tensor = None) -> dict:
        align_loss = self._compute_align_loss(hr_vector, tail_vector) if self.use_alignment else torch.tensor(0.0, device=hr_vector.device)

        if self.use_uniformity:
            uniform_components_1 = self._compute_uniform_loss(hr_vector, tail_vector, batch_exs, self.uniformity_scale1, head_vector)
            uniform_components_2 = self._compute_uniform_loss(hr_vector, tail_vector, batch_exs, self.uniformity_scale2, head_vector)
        else:
            uniform_components_1 = {
                'total': torch.tensor(0.0, device=hr_vector.device),
                'query': torch.tensor(0.0, device=hr_vector.device),
                'tail': torch.tensor(0.0, device=hr_vector.device),
                'head': torch.tensor(0.0, device=hr_vector.device),
                'entity': torch.tensor(0.0, device=hr_vector.device),
            }
            uniform_components_2 = {
                'total': torch.tensor(0.0, device=hr_vector.device),
                'query': torch.tensor(0.0, device=hr_vector.device),
                'tail': torch.tensor(0.0, device=hr_vector.device),
                'head': torch.tensor(0.0, device=hr_vector.device),
                'entity': torch.tensor(0.0, device=hr_vector.device),
            }

        cross_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        if self.use_uniformity and self.use_uniformity_cross:
            cross_uniform_loss = self._compute_cross_uniformity_loss(hr_vector, tail_vector, triplet_mask)

        uniform_loss_1 = uniform_components_1['total']
        uniform_loss_2 = uniform_components_2['total']

        scaled_align = self.alpha * align_loss
        scaled_uniform_1 = self.gamma1 * uniform_loss_1
        scaled_uniform_2 = self.gamma2 * uniform_loss_2
        scaled_cross = self.gamma_cross * cross_uniform_loss
        total_loss = scaled_align + scaled_uniform_1 + scaled_uniform_2 + scaled_cross

        return {
            'loss': total_loss,
            'align_loss': align_loss.detach(),
            'align_loss_scaled': scaled_align.detach(),
            'uniform_loss': (uniform_loss_1 + uniform_loss_2 + cross_uniform_loss).detach(),
            'uniform_loss_scaled': (scaled_uniform_1 + scaled_uniform_2 + scaled_cross).detach(),
            'uniform_loss_1': uniform_loss_1.detach(),
            'uniform_loss_2': uniform_loss_2.detach(),
            'uniform_loss_1_scaled': scaled_uniform_1.detach(),
            'uniform_loss_2_scaled': scaled_uniform_2.detach(),
            'uniform_loss_query': (uniform_components_1['query'] + uniform_components_2['query']).detach(),
            'uniform_loss_tail': (uniform_components_1['tail'] + uniform_components_2['tail']).detach(),
            'uniform_loss_head': (uniform_components_1['head'] + uniform_components_2['head']).detach(),
            'uniform_loss_entity': (uniform_components_1['entity'] + uniform_components_2['entity']).detach(),
            'uniform_loss_cross': cross_uniform_loss.detach(),
            'uniform_loss_query_scaled': (self.gamma1 * uniform_components_1['query'] + self.gamma2 * uniform_components_2['query']).detach(),
            'uniform_loss_tail_scaled': (self.gamma1 * uniform_components_1['tail'] + self.gamma2 * uniform_components_2['tail']).detach(),
            'uniform_loss_head_scaled': (self.gamma1 * uniform_components_1['head'] + self.gamma2 * uniform_components_2['head']).detach(),
            'uniform_loss_entity_scaled': (self.gamma1 * uniform_components_1['entity'] + self.gamma2 * uniform_components_2['entity']).detach(),
            'uniform_loss_cross_scaled': scaled_cross.detach(),
        }

    def _compute_align_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor) -> torch.tensor:
        squared_l2_dist = torch.sum((hr_vector - tail_vector) ** 2, dim=-1)
        return torch.mean(squared_l2_dist)

    def _resolve_cross_uniformity_beta(self):
        if self.cross_uniformity_beta is not None:
            return self.cross_uniformity_beta
        return self.uniformity_scale1

    def _compute_cross_uniformity_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                                       triplet_mask: torch.tensor = None) -> torch.tensor:
        if hr_vector.size(0) == 0:
            return torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)

        dist = torch.cdist(hr_vector, tail_vector, p=2)
        dist2 = dist ** 2
        beta = self._resolve_cross_uniformity_beta()
        if torch.is_tensor(beta):
            beta = beta.to(device=hr_vector.device, dtype=hr_vector.dtype)

        if triplet_mask is not None:
            mask = triplet_mask.to(hr_vector.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.clone()
            if mask.shape == dist2.shape:
                mask.fill_diagonal_(False)
            else:
                mask = None
        else:
            mask = None

        if mask is None:
            mask = torch.ones_like(dist2, dtype=torch.bool, device=hr_vector.device)
            mask.fill_diagonal_(False)

        neg_scores = (-beta * dist2).masked_fill(~mask, float('-inf'))
        logsumexp = torch.logsumexp(neg_scores, dim=1)
        num_neg = mask.sum(dim=1).clamp(min=1)
        logmeanexp = logsumexp - torch.log(num_neg.to(dist2.dtype))
        logmeanexp = torch.where(torch.isfinite(logmeanexp), logmeanexp, torch.zeros_like(logmeanexp))
        return torch.mean(logmeanexp)

    def _compute_uniform_loss_for_vectors(self, vectors: torch.tensor, scale: float) -> torch.tensor:
        if vectors.size(0) < 2:
            return torch.tensor(0.0, device=vectors.device, dtype=vectors.dtype)

        pairwise_dists = torch.cdist(vectors, vectors, p=2)
        pairwise_mask = ~torch.eye(vectors.size(0), dtype=torch.bool, device=vectors.device)
        pairwise_dists = pairwise_dists[pairwise_mask]

        exp_term = torch.exp(-scale * pairwise_dists ** 2)
        mean_exp = torch.mean(exp_term)
        return torch.log(mean_exp + self.eps)

    def _compute_uniform_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                              batch_exs: list, scale: float,
                              head_vector: torch.tensor = None) -> torch.tensor:
        if batch_exs is not None:
            query_keys = [(ex.head_id, ex.relation) for ex in batch_exs]
            tail_ids = [ex.tail_id for ex in batch_exs]

            def unique_indices_by_id(ids):
                seen = set()
                uniq_idx = []
                for i, idv in enumerate(ids):
                    if idv not in seen:
                        seen.add(idv)
                        uniq_idx.append(i)
                return torch.tensor(uniq_idx, dtype=torch.long, device=hr_vector.device)

            hr_idx = unique_indices_by_id(query_keys)
            tail_idx = unique_indices_by_id(tail_ids)

            hr_unique = hr_vector[hr_idx]
            tail_unique = tail_vector[tail_idx]
        else:
            hr_unique = hr_vector
            tail_unique = tail_vector

        total_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        query_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        tail_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        head_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        entity_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        if self.use_uniformity_query:
            query_uniform_loss = self._compute_uniform_loss_for_vectors(hr_unique, scale)
            total_uniform_loss = total_uniform_loss + query_uniform_loss
        if self.use_uniformity_tail:
            tail_uniform_loss = self._compute_uniform_loss_for_vectors(tail_unique, scale)
            total_uniform_loss = total_uniform_loss + tail_uniform_loss

        if self.use_uniformity_head and head_vector is not None:
            if batch_exs is not None:
                head_ids = [ex.head_id for ex in batch_exs]
                def unique_indices_by_id(ids):
                    seen = set()
                    uniq_idx = []
                    for i, idv in enumerate(ids):
                        if idv not in seen:
                            seen.add(idv)
                            uniq_idx.append(i)
                    return torch.tensor(uniq_idx, dtype=torch.long, device=head_vector.device)
                head_idx = unique_indices_by_id(head_ids)
                head_unique = head_vector[head_idx]
                head_uniform_loss = self._compute_uniform_loss_for_vectors(head_unique, scale)
            else:
                head_uniform_loss = self._compute_uniform_loss_for_vectors(head_vector, scale)
            total_uniform_loss = total_uniform_loss + head_uniform_loss

        if self.use_uniformity_entity and head_vector is not None:
            if batch_exs is not None:
                seen = set()
                entity_vectors = []
                for i, ex in enumerate(batch_exs):
                    if ex.head_id not in seen:
                        seen.add(ex.head_id)
                        entity_vectors.append(head_vector[i])
                    if ex.tail_id not in seen:
                        seen.add(ex.tail_id)
                        entity_vectors.append(tail_vector[i])
                if len(entity_vectors) >= 2:
                    entity_stack = torch.stack(entity_vectors, dim=0)
                    entity_uniform_loss = self._compute_uniform_loss_for_vectors(entity_stack, scale)
                else:
                    entity_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
            else:
                entity_stack = torch.cat([head_vector, tail_vector], dim=0)
                entity_uniform_loss = self._compute_uniform_loss_for_vectors(entity_stack, scale)
            total_uniform_loss = total_uniform_loss + entity_uniform_loss

        return {
            'total': total_uniform_loss,
            'query': query_uniform_loss,
            'tail': tail_uniform_loss,
            'head': head_uniform_loss,
            'entity': entity_uniform_loss,
        }


class AdaptiveHybridDirectAULoss(nn.Module):
    """Alignment plus adaptive mix of two uniformity scales with learnable alpha."""

    def __init__(self, alpha_init: float = 0.5, align_alpha: float = 1.0, gamma: float = 1.0, eps: float = 1e-12,
                 uniformity_scale1: float = 4.0, uniformity_scale2: float = 6.0,
                 use_alignment: bool = True, use_uniformity: bool = True,
                 use_uniformity_query: bool = True, use_uniformity_tail: bool = True,
                 use_uniformity_head: bool = False, use_uniformity_entity: bool = False,
                 use_uniformity_cross: bool = False, cross_uniformity_beta: float = None,
                 gamma_cross: float = None):
        super().__init__()
        alpha_init = float(alpha_init)
        alpha_init = min(max(alpha_init, 1e-6), 1.0 - 1e-6)
        self.alpha_logit = nn.Parameter(torch.log(torch.tensor(alpha_init / (1.0 - alpha_init))))
        self.align_alpha = align_alpha
        self.gamma = gamma
        self.gamma_cross = gamma if gamma_cross is None else gamma_cross
        self.eps = eps
        self.uniformity_scale1 = uniformity_scale1
        self.uniformity_scale2 = uniformity_scale2
        self.cross_uniformity_beta = cross_uniformity_beta
        self.use_alignment = use_alignment
        self.use_uniformity = use_uniformity
        self.use_uniformity_query = use_uniformity_query
        self.use_uniformity_tail = use_uniformity_tail
        self.use_uniformity_head = use_uniformity_head
        self.use_uniformity_entity = use_uniformity_entity
        self.use_uniformity_cross = use_uniformity_cross

    def _alpha(self) -> torch.tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                labels: torch.tensor = None, batch_exs: list = None,
                head_vector: torch.tensor = None, triplet_mask: torch.tensor = None) -> dict:
        align_loss = self._compute_align_loss(hr_vector, tail_vector) if self.use_alignment else torch.tensor(0.0, device=hr_vector.device)

        if self.use_uniformity:
            uniform_components_1 = self._compute_uniform_loss(hr_vector, tail_vector, batch_exs, self.uniformity_scale1, head_vector)
            uniform_components_2 = self._compute_uniform_loss(hr_vector, tail_vector, batch_exs, self.uniformity_scale2, head_vector)
        else:
            uniform_components_1 = {
                'total': torch.tensor(0.0, device=hr_vector.device),
                'query': torch.tensor(0.0, device=hr_vector.device),
                'tail': torch.tensor(0.0, device=hr_vector.device),
                'head': torch.tensor(0.0, device=hr_vector.device),
                'entity': torch.tensor(0.0, device=hr_vector.device),
            }
            uniform_components_2 = {
                'total': torch.tensor(0.0, device=hr_vector.device),
                'query': torch.tensor(0.0, device=hr_vector.device),
                'tail': torch.tensor(0.0, device=hr_vector.device),
                'head': torch.tensor(0.0, device=hr_vector.device),
                'entity': torch.tensor(0.0, device=hr_vector.device),
            }

        cross_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        if self.use_uniformity and self.use_uniformity_cross:
            cross_uniform_loss = self._compute_cross_uniformity_loss(hr_vector, tail_vector, triplet_mask)

        alpha = self._alpha()
        uniform_loss_1 = uniform_components_1['total']
        uniform_loss_2 = uniform_components_2['total']
        intra_uniform_loss = alpha * uniform_loss_1 + (1.0 - alpha) * uniform_loss_2
        uniform_loss = intra_uniform_loss + cross_uniform_loss

        scaled_align = self.align_alpha * align_loss
        scaled_intra = self.gamma * intra_uniform_loss
        scaled_cross = self.gamma_cross * cross_uniform_loss
        scaled_uniform = scaled_intra + scaled_cross
        total_loss = scaled_align + scaled_uniform

        return {
            'loss': total_loss,
            'align_loss': align_loss.detach(),
            'align_loss_scaled': scaled_align.detach(),
            'uniform_loss': uniform_loss.detach(),
            'uniform_loss_scaled': scaled_uniform.detach(),
            'uniform_loss_1': uniform_loss_1.detach(),
            'uniform_loss_2': uniform_loss_2.detach(),
            'uniform_loss_1_scaled': (self.gamma * uniform_loss_1).detach(),
            'uniform_loss_2_scaled': (self.gamma * uniform_loss_2).detach(),
            'uniform_loss_query': (uniform_components_1['query'] + uniform_components_2['query']).detach(),
            'uniform_loss_tail': (uniform_components_1['tail'] + uniform_components_2['tail']).detach(),
            'uniform_loss_head': (uniform_components_1['head'] + uniform_components_2['head']).detach(),
            'uniform_loss_entity': (uniform_components_1['entity'] + uniform_components_2['entity']).detach(),
            'uniform_loss_cross': cross_uniform_loss.detach(),
            'uniform_loss_query_scaled': (self.gamma * (uniform_components_1['query'] + uniform_components_2['query'])).detach(),
            'uniform_loss_tail_scaled': (self.gamma * (uniform_components_1['tail'] + uniform_components_2['tail'])).detach(),
            'uniform_loss_head_scaled': (self.gamma * (uniform_components_1['head'] + uniform_components_2['head'])).detach(),
            'uniform_loss_entity_scaled': (self.gamma * (uniform_components_1['entity'] + uniform_components_2['entity'])).detach(),
            'uniform_loss_cross_scaled': (self.gamma_cross * cross_uniform_loss).detach(),
            'uniform_alpha': alpha.detach(),
        }

    def _compute_align_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor) -> torch.tensor:
        squared_l2_dist = torch.sum((hr_vector - tail_vector) ** 2, dim=-1)
        return torch.mean(squared_l2_dist)

    def _resolve_cross_uniformity_beta(self):
        if self.cross_uniformity_beta is not None:
            return self.cross_uniformity_beta
        return self.uniformity_scale1

    def _compute_cross_uniformity_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                                       triplet_mask: torch.tensor = None) -> torch.tensor:
        if hr_vector.size(0) == 0:
            return torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)

        dist = torch.cdist(hr_vector, tail_vector, p=2)
        dist2 = dist ** 2
        beta = self._resolve_cross_uniformity_beta()
        if torch.is_tensor(beta):
            beta = beta.to(device=hr_vector.device, dtype=hr_vector.dtype)

        if triplet_mask is not None:
            mask = triplet_mask.to(hr_vector.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.clone()
            if mask.shape == dist2.shape:
                mask.fill_diagonal_(False)
            else:
                mask = None
        else:
            mask = None

        if mask is None:
            mask = torch.ones_like(dist2, dtype=torch.bool, device=hr_vector.device)
            mask.fill_diagonal_(False)

        neg_scores = (-beta * dist2).masked_fill(~mask, float('-inf'))
        logsumexp = torch.logsumexp(neg_scores, dim=1)
        num_neg = mask.sum(dim=1).clamp(min=1)
        logmeanexp = logsumexp - torch.log(num_neg.to(dist2.dtype))
        logmeanexp = torch.where(torch.isfinite(logmeanexp), logmeanexp, torch.zeros_like(logmeanexp))
        return torch.mean(logmeanexp)

    def _compute_uniform_loss_for_vectors(self, vectors: torch.tensor, scale: float) -> torch.tensor:
        if vectors.size(0) < 2:
            return torch.tensor(0.0, device=vectors.device, dtype=vectors.dtype)

        pairwise_dists = torch.cdist(vectors, vectors, p=2)
        pairwise_mask = ~torch.eye(vectors.size(0), dtype=torch.bool, device=vectors.device)
        pairwise_dists = pairwise_dists[pairwise_mask]

        exp_term = torch.exp(-scale * pairwise_dists ** 2)
        mean_exp = torch.mean(exp_term)
        return torch.log(mean_exp + self.eps)

    def _compute_uniform_loss(self, hr_vector: torch.tensor, tail_vector: torch.tensor,
                              batch_exs: list, scale: float,
                              head_vector: torch.tensor = None) -> dict:
        if batch_exs is not None:
            query_keys = [(ex.head_id, ex.relation) for ex in batch_exs]
            tail_ids = [ex.tail_id for ex in batch_exs]

            def unique_indices_by_id(ids):
                seen = set()
                uniq_idx = []
                for i, idv in enumerate(ids):
                    if idv not in seen:
                        seen.add(idv)
                        uniq_idx.append(i)
                return torch.tensor(uniq_idx, dtype=torch.long, device=hr_vector.device)

            hr_idx = unique_indices_by_id(query_keys)
            tail_idx = unique_indices_by_id(tail_ids)

            hr_unique = hr_vector[hr_idx]
            tail_unique = tail_vector[tail_idx]
        else:
            hr_unique = hr_vector
            tail_unique = tail_vector

        total_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        query_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        tail_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        head_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
        entity_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)

        if self.use_uniformity_query:
            query_uniform_loss = self._compute_uniform_loss_for_vectors(hr_unique, scale)
            total_uniform_loss = total_uniform_loss + query_uniform_loss
        if self.use_uniformity_tail:
            tail_uniform_loss = self._compute_uniform_loss_for_vectors(tail_unique, scale)
            total_uniform_loss = total_uniform_loss + tail_uniform_loss

        if self.use_uniformity_head and head_vector is not None:
            if batch_exs is not None:
                head_ids = [ex.head_id for ex in batch_exs]
                def unique_indices_by_id(ids):
                    seen = set()
                    uniq_idx = []
                    for i, idv in enumerate(ids):
                        if idv not in seen:
                            seen.add(idv)
                            uniq_idx.append(i)
                    return torch.tensor(uniq_idx, dtype=torch.long, device=head_vector.device)
                head_idx = unique_indices_by_id(head_ids)
                head_unique = head_vector[head_idx]
                head_uniform_loss = self._compute_uniform_loss_for_vectors(head_unique, scale)
            else:
                head_uniform_loss = self._compute_uniform_loss_for_vectors(head_vector, scale)
            total_uniform_loss = total_uniform_loss + head_uniform_loss

        if self.use_uniformity_entity and head_vector is not None:
            if batch_exs is not None:
                seen = set()
                entity_vectors = []
                for i, ex in enumerate(batch_exs):
                    if ex.head_id not in seen:
                        seen.add(ex.head_id)
                        entity_vectors.append(head_vector[i])
                    if ex.tail_id not in seen:
                        seen.add(ex.tail_id)
                        entity_vectors.append(tail_vector[i])
                if len(entity_vectors) >= 2:
                    entity_stack = torch.stack(entity_vectors, dim=0)
                    entity_uniform_loss = self._compute_uniform_loss_for_vectors(entity_stack, scale)
                else:
                    entity_uniform_loss = torch.tensor(0.0, device=hr_vector.device, dtype=hr_vector.dtype)
            else:
                entity_stack = torch.cat([head_vector, tail_vector], dim=0)
                entity_uniform_loss = self._compute_uniform_loss_for_vectors(entity_stack, scale)
            total_uniform_loss = total_uniform_loss + entity_uniform_loss

        return {
            'total': total_uniform_loss,
            'query': query_uniform_loss,
            'tail': tail_uniform_loss,
            'head': head_uniform_loss,
            'entity': entity_uniform_loss,
        }


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


def filter_shared_encoder_state_dict(state_dict: dict, shared_encoder: bool) -> dict:
    if not shared_encoder:
        return state_dict
    filtered = {k: v for k, v in state_dict.items() if not k.startswith('tail_bert.')}
    if len(filtered) < len(state_dict):
        from logger_config import logger
        logger.info('Dropped %d tail_bert keys for shared-encoder loading',
                    len(state_dict) - len(filtered))
    return filtered


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    head_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        
        loss_type = getattr(args, 'loss_type', 'infonce')
        self.use_uniformity_loss = bool(getattr(args, 'use_uniformity_loss', False))
        self.use_negative_sampling = bool(getattr(args, 'use_negative_sampling', True))
        
        self.use_infonce_loss = (loss_type in ['infonce', 'all'])
        self.use_alignment_loss = (loss_type in ['alignment', 'all'])
        self.use_bridge_loss = (loss_type == 'bridge')
        if loss_type == 'all':
            self.use_uniformity_loss = True
        self.directau = self.use_alignment_loss or self.use_bridge_loss
        self.directau_eps = float(getattr(args, 'directau_eps', 1e-12))
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.shared_encoder = bool(getattr(args, 'shared_encoder', False))
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        if self.shared_encoder:
            self.tail_bert = None
        else:
            self.tail_bert = deepcopy(self.hr_bert)

    @property
    def ent_encoder(self):
        return self.hr_bert if self.shared_encoder else self.tail_bert

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.ent_encoder,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.ent_encoder,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        if self.use_alignment_loss or self.use_uniformity_loss or self.use_bridge_loss:
            hr_vector = F.normalize(hr_vector, p=2, dim=-1, eps=self.directau_eps)
            tail_vector = F.normalize(tail_vector, p=2, dim=-1, eps=self.directau_eps)
            head_vector = F.normalize(head_vector, p=2, dim=-1, eps=self.directau_eps)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())

        # If alignment-only mode (DirectAU replacing InfoNCE), return early with embeddings
        # Note: do NOT early-return when only uniformity is enabled — uniformity should be
        # applied as an auxiliary term alongside InfoNCE when configured.
        if (self.use_alignment_loss or self.use_bridge_loss) and not self.use_infonce_loss:
            return {'logits': logits,
                    'labels': labels,
                    'inv_t': torch.tensor(1.0, device=hr_vector.device),
                    'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': output_dict['head_vector']}
        
        # For InfoNCE mode (default)
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        # Apply triplet mask only if negative sampling is enabled
        if self.use_negative_sampling:
            triplet_mask = batch_dict.get('triplet_mask', None)
            if triplet_mask is not None:
                logits.masked_fill_(~triplet_mask, -1e4)

        # Pre-batch negatives: only if negative sampling is enabled
        if self.pre_batch > 0 and self.training and self.use_negative_sampling:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        # Self-negatives: only if negative sampling is enabled
        if self.args.use_self_negative and self.training and self.use_negative_sampling:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict.get('self_negative_mask', None)
            if self_negative_mask is None:
                # Keep behavior stable when mask is unavailable (e.g., misconfigured test mode during training).
                self_negative_mask = torch.ones(batch_size, dtype=torch.bool, device=hr_vector.device)
            else:
                self_negative_mask = self_negative_mask.to(hr_vector.device).bool()
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        # Keep gradients for auxiliary losses (alignment/uniformity) in "all" mode.
        # Detach only when no auxiliary objective is active to reduce graph retention.
        if self.use_alignment_loss or self.use_uniformity_loss or self.use_bridge_loss:
            out_hr_vector = hr_vector
            out_tail_vector = tail_vector
            out_head_vector = output_dict['head_vector']
        else:
            out_hr_vector = hr_vector.detach()
            out_tail_vector = tail_vector.detach()
            out_head_vector = output_dict['head_vector'].detach()

        return {'logits': logits,
            'labels': labels,
            'inv_t': self.log_inv_t.detach().exp(),
            'hr_vector': out_hr_vector,
            'tail_vector': out_tail_vector,
            'head_vector': out_head_vector}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.ent_encoder,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector

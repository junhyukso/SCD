# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative sampler for Jacobi iteration."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


class CosmosSpeculativeSampler:
    """Speculative sampler implementing accept/reject logic for Jacobi iteration."""
    
    def __init__(
        self,
        generator: Optional[torch.Generator] = None,
        max_collected_logits: int = 2,
    ):
        """Initialize the speculative sampler.
        
        Args:
            generator: Random generator for reproducibility
            max_collected_logits: Maximum number of logits to collect
        """
        self.generator = generator
        self.max_collected_logits = max_collected_logits
        self.collected_draft_logits = []
        self.collected_advanced_logits = []
        
    def collect_logits(self, logits: torch.Tensor, collection_type: str = 'draft') -> Optional[torch.Tensor]:
        """Collect logits for future use.
        
        Args:
            logits: Logits tensor to collect
            collection_type: Either 'draft' or 'advanced'
            
        Returns:
            Oldest logits if collection is full, None otherwise
        """
        if collection_type == 'draft':
            collected_logits = self.collected_draft_logits
        elif collection_type == 'advanced':
            collected_logits = self.collected_advanced_logits
        else:
            raise ValueError(f"Invalid collection_type: {collection_type}")
        
        if logits is not None:
            collected_logits.append(logits)
        
        if len(collected_logits) > self.max_collected_logits:
            return collected_logits.pop(0)
        else:
            return None
    
    def reject_sampling(
        self,
        advanced_prob: torch.Tensor,
        draft_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform reject sampling for a single token position.
        
        Args:
            advanced_prob: Probability distribution from advanced model
            draft_prob: Probability distribution from draft model
            
        Returns:
            Tuple of (resampled_token, resampled_scores)
        """
        # Compute corrected probabilities: max(0, p_advanced - p_draft)
        corrected_prob = (advanced_prob - draft_prob).clamp(min=0) + 1e-10
        
        # Renormalize
        corrected_prob = corrected_prob / corrected_prob.sum(dim=-1, keepdim=True)
        
        # Sample from corrected distribution
        if len(corrected_prob.shape) == 1:
            corrected_prob = corrected_prob.unsqueeze(0)
            
        resampled_token = torch.multinomial(
            corrected_prob, 
            num_samples=1, 
            generator=self.generator
        ).squeeze(-1)
        
        return resampled_token, corrected_prob
    
    def __call__(
        self,
        draft_tokens: torch.Tensor,
        advanced_tokens: torch.Tensor, 
        draft_prob: torch.Tensor,
        advanced_prob: torch.Tensor,
        maximal_coupling: bool = True,
        **kwargs
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Main accept/reject sampling logic.
        
        Args:
            draft_tokens: Draft token predictions [batch_size, seq_len]
            advanced_tokens: Advanced model token predictions [batch_size, seq_len]
            draft_prob: Draft model probabilities [batch_size, seq_len, vocab_size]
            advanced_prob: Advanced model probabilities [batch_size, seq_len, vocab_size]
            
        Returns:
            Tuple of (first_mismatch_positions, accepted_tokens, accepted_scores)
        """
        batch_size, seq_len = draft_tokens.shape
        device = draft_tokens.device
        
        # Vectorized accept/reject over batch and time (positions 1..L-1 affect output 0..L-2)
        # Random numbers for positions 1..L-1
        if seq_len > 1:
            random_vals = torch.rand(
                (batch_size, seq_len - 1),
                device=device,
                generator=self.generator,
            )
        else:
            random_vals = torch.empty((batch_size, 0), device=device)

        accepted_tokens = advanced_tokens.clone()
        accepted_scores = advanced_prob.clone()

        if seq_len > 1:
            # Slices
            draft_next_tokens = draft_tokens[:, 1:]  # [B, L-1]
            adv_prev_prob = advanced_prob[:, :-1, :]  # [B, L-1, V]
            draft_next_prob = draft_prob[:, 1:, :]  # [B, L-1, V]

            # Neighborhood-summed probabilities around the draft token based on advanced ranking
            G = 4
            if G < 0:
                G = 0
            vocab_size = adv_prev_prob.shape[-1]

            if G == 0:
                # Original single-index gather
                p_adv_on_draft = adv_prev_prob.gather(-1, draft_next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
                p_draft_on_draft = draft_next_prob.gather(-1, draft_next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            else:
                # Sort token indices by advanced probability for each (batch, time)
                adv_sort_indices = adv_prev_prob.argsort(dim=-1)  # [B, L-1, V], ascending ranks -> token_id

                # Invert permutation to get rank for each token_id
                rank_values = torch.arange(vocab_size, device=device).view(1, 1, vocab_size)
                rank_values = rank_values.expand(batch_size, seq_len - 1, vocab_size)
                inv_rank = torch.empty_like(adv_sort_indices)
                inv_rank.scatter_(-1, adv_sort_indices, rank_values)  # inv_rank[b,t,token_id] = rank

                draft_token_ranks = inv_rank.gather(-1, draft_next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

                # Build fixed window of neighbor ranks [r-G, r+G] clamped to [0, V-1]
                window_width = 2 * G + 1
                offsets = torch.arange(-G, G + 1, device=device).view(1, 1, window_width)  # [1,1,W]
                neighbor_ranks = (draft_token_ranks.unsqueeze(-1) + offsets).clamp(min=0, max=vocab_size - 1)  # [B,L-1,W]

                # Map neighbor ranks back to token ids
                neighbor_token_ids = adv_sort_indices.gather(-1, neighbor_ranks)  # [B, L-1, W]

                # Sum probabilities over the neighborhood
                p_adv_on_draft = adv_prev_prob.gather(-1, neighbor_token_ids).sum(dim=-1)  # [B, L-1]
                p_draft_on_draft = draft_next_prob.gather(-1, neighbor_token_ids).sum(dim=-1)  # [B, L-1]

            # Acceptance ratio and mask
            ratio = (p_adv_on_draft / (p_draft_on_draft + 1e-10)).clamp(max=1.0)
            accept_mask = random_vals < ratio  # [B, L-1]

            # Reject-sampling distribution for all positions
            corrected_prob = (adv_prev_prob - draft_next_prob).clamp(min=0) + 1e-10  # [B, L-1, V]
            corrected_prob = corrected_prob / corrected_prob.sum(dim=-1, keepdim=True)

            # Sample for all positions in one call
            flat_corr = corrected_prob.reshape(-1, corrected_prob.shape[-1])  # [B*(L-1), V]
            resampled_flat = torch.multinomial(flat_corr, num_samples=1, generator=self.generator)  # [B*(L-1), 1]
            resampled_tokens = resampled_flat.view(batch_size, seq_len - 1)  # [B, L-1]

            if maximal_coupling:
                # Merge accepted draft tokens and rejected resampled tokens
                merged_tokens = torch.where(accept_mask, draft_next_tokens, resampled_tokens)
                accepted_tokens[:, :-1] = merged_tokens
            else:
                # If not maximal coupling: to the right of first mismatch, fall back to advanced tokens
                # Build accepted as advanced by default, then apply only accepts up to first mismatch
                accepted_tokens[:, :-1] = advanced_tokens[:, :-1]

            # First mismatch positions per batch (default to L if none)
            mismatch_mask = ~accept_mask  # [B, L-1]
            if mismatch_mask.any():
                pos_idx = torch.arange(1, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, L-1]
                filled = torch.where(mismatch_mask, pos_idx, torch.full_like(pos_idx, seq_len))
                first_mismatch_positions = filled.min(dim=1).values.tolist()
            else:
                first_mismatch_positions = [seq_len for _ in range(batch_size)]

            if not maximal_coupling:
                # For each batch, keep accepted draft tokens only up to first mismatch-1, else advanced
                if mismatch_mask.any():
                    # create mask up to first mismatch per batch
                    cut_points = torch.tensor(first_mismatch_positions, device=device).unsqueeze(1)  # [B, 1]
                    time_idx = torch.arange(1, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, L-1]
                    left_mask = time_idx < cut_points  # [B, L-1]
                    accepted_tokens[:, :-1] = torch.where(left_mask, draft_next_tokens, advanced_tokens[:, :-1])
                else:
                    # No mismatch: accept all draft tokens
                    accepted_tokens[:, :-1] = draft_next_tokens
                # accepted_scores remains as advanced_prob clone; right side is already advanced
        else:
            # No positions to update; accept all by default
            first_mismatch_positions = [seq_len for _ in range(batch_size)]

        return first_mismatch_positions, accepted_tokens, accepted_scores

    def clear_collected_logits(self):
        """Clear collected logits."""
        self.collected_draft_logits.clear()
        self.collected_advanced_logits.clear() 
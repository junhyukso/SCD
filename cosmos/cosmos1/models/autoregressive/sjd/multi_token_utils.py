# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-token utilities for Speculative Jacobi Decoding."""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def get_multi_token_initialization(
    input_ids: torch.Tensor,
    num_tokens: int,
    vocab_size: int,
    init_scheme: str = 'random',
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize multiple tokens for Jacobi iteration.
    
    Args:
        input_ids: Current input token ids [batch_size, seq_len]
        num_tokens: Number of tokens to initialize
        vocab_size: Vocabulary size for token sampling
        init_scheme: Initialization scheme ('random' or 'repeat_horizon')
        generator: Random generator for reproducibility
        device: Device to place tensors on
        
    Returns:
        Tuple of (initialized_tokens, token_scores)
    """
    if device is None:
        device = input_ids.device
    
    batch_size = input_ids.shape[0]
    
    if init_scheme == 'random':
        # Random initialization from full vocabulary
        rand_tokens = torch.randint(
            0, vocab_size,
            (batch_size, num_tokens),
            device=device,
            generator=generator,
            dtype=input_ids.dtype
        )
        
        # Create one-hot scores for the random tokens
        token_scores = torch.zeros(
            (batch_size, num_tokens, vocab_size),
            device=device,
            dtype=torch.float32
        )
        token_scores.scatter_(-1, rand_tokens.unsqueeze(-1), 1.0)
        
    elif init_scheme == 'repeat_horizon':
        # For video generation, repeat the last token
        if input_ids.shape[1] > 0:
            last_token = input_ids[:, -1:].expand(-1, num_tokens)
        else:
            # Fallback to random if no input tokens
            last_token = torch.randint(
                0, vocab_size,
                (batch_size, num_tokens), 
                device=device,
                generator=generator,
                dtype=input_ids.dtype
            )
        
        rand_tokens = last_token
        
        # Create one-hot scores
        token_scores = torch.zeros(
            (batch_size, num_tokens, vocab_size),
            device=device,
            dtype=torch.float32
        )
        token_scores.scatter_(-1, rand_tokens.unsqueeze(-1), 1.0)
        
    else:
        raise ValueError(f"Unknown init_scheme: {init_scheme}")
    
    return rand_tokens, token_scores


def prepare_multi_token_inputs(
    input_ids: torch.Tensor,
    additional_tokens: torch.Tensor,
    cache_position: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare inputs for multi-token prediction.
    
    Args:
        input_ids: Current input tokens [batch_size, seq_len]
        additional_tokens: Additional tokens to append [batch_size, num_additional]
        cache_position: Current cache positions
        
    Returns:
        Tuple of (extended_input_ids, updated_cache_position)
    """
    # Concatenate input_ids with additional tokens
    extended_input_ids = torch.cat([input_ids, additional_tokens], dim=-1)
    
    # Update cache position to include new tokens
    num_additional = additional_tokens.shape[-1]
    additional_positions = torch.arange(
        cache_position[-1].item() + 1,
        cache_position[-1].item() + 1 + num_additional,
        device=cache_position.device,
        dtype=cache_position.dtype
    )
    updated_cache_position = torch.cat([cache_position, additional_positions])
    
    return extended_input_ids, updated_cache_position


def rollback_kv_cache(model, num_tokens_to_remove: int):
    """Rollback KV cache by removing the last N tokens.
    
    Args:
        model: The autoregressive model with attention layers
        num_tokens_to_remove: Number of tokens to remove from cache
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'attention'):
                attention = layer.attention
                if hasattr(attention, 'cache_k') and hasattr(attention, 'cache_v'):
                    # Remove last num_tokens_to_remove entries
                    attention.cache_k = attention.cache_k[..., :-num_tokens_to_remove, :]
                    attention.cache_v = attention.cache_v[..., :-num_tokens_to_remove, :]


def find_first_mismatch(predicted_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Find the first position where predicted and target tokens mismatch.
    
    Args:
        predicted_tokens: Predicted token sequence [batch_size, seq_len]
        target_tokens: Target token sequence [batch_size, seq_len]
        
    Returns:
        Tensor of first mismatch positions for each batch element
    """
    batch_size = predicted_tokens.shape[0]
    seq_len = min(predicted_tokens.shape[1], target_tokens.shape[1])
    
    mismatch_positions = []
    for b in range(batch_size):
        mismatch_pos = seq_len  # Default to end if no mismatch
        for i in range(seq_len):
            if predicted_tokens[b, i] != target_tokens[b, i]:
                mismatch_pos = i
                break
        mismatch_positions.append(mismatch_pos)
    
    return torch.tensor(mismatch_positions, device=predicted_tokens.device) 
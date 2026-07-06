# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for Speculative Jacobi Decoding."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SJDConfig:
    """Configuration for Speculative Jacobi Decoding.
    
    Args:
        enable_sjd: Whether to enable SJD (default: True)
        max_num_new_tokens: Maximum number of tokens to generate per SJD iteration (default: 16)
        multi_token_init_scheme: Token initialization scheme ('random' or 'repeat_horizon')
        guidance_scale: CFG guidance scale (default: 3.0)
        seed: Random seed for reproducibility (default: None)
        image_top_k: Top-k for image tokens (default: 2000)
        text_top_k: Top-k for text tokens (default: 10)
        max_jacobi_iterations: Maximum Jacobi iterations per step (default: 200)
        prefix_token_sampler_scheme: Sampling scheme ('speculative_jacobi' or 'jacobi')
        maximal_coupling: If True, use maximal-coupling accept/reject; if False, fall back to advanced tokens to the right of first mismatch
    """
    enable_sjd: bool = True
    max_num_new_tokens: int = 16
    multi_token_init_scheme: str = 'random'
    guidance_scale: float = 3.0
    seed: Optional[int] = None
    image_top_k: int = 2000
    text_top_k: int = 10
    max_jacobi_iterations: int = 200
    prefix_token_sampler_scheme: str = 'speculative_jacobi'
    maximal_coupling: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.multi_token_init_scheme in ['random', 'repeat_horizon'], \
            f"Invalid multi_token_init_scheme: {self.multi_token_init_scheme}"
        assert self.prefix_token_sampler_scheme in ['speculative_jacobi', 'jacobi'], \
            f"Invalid prefix_token_sampler_scheme: {self.prefix_token_sampler_scheme}"
        assert self.max_num_new_tokens > 0, "max_num_new_tokens must be positive" 
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SJD-enabled world generation pipeline."""

import gc
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange

from .world_generation_pipeline import ARBaseGenerationPipeline, create_inference_config, detect_model_size_from_ckpt_path
from cosmos1.models.autoregressive.sjd import SJDConfig
from cosmos1.models.autoregressive.configs.inference.inference_config import SamplingConfig
from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.autoregressive.utils.inference import _SUPPORTED_CONTEXT_LEN, prepare_video_batch_for_saving
from cosmos1.utils import log, misc


class ARSJDGenerationPipeline(ARBaseGenerationPipeline):
    """SJD-enabled autoregressive world generation pipeline.
    
    Extends the base pipeline to support Speculative Jacobi Decoding for faster generation.
    """
    
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        sjd_config: Optional[SJDConfig] = None,
        has_text_input: bool = False,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        disable_diffusion_decoder: bool = False,
        offload_guardrail_models: bool = False,
        offload_diffusion_decoder: bool = False,
    ):
        """Initialize the SJD world generation pipeline.
        
        Args:
            inference_type: Type of world generation ('base' or 'video2world')
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the AR checkpoint to load
            sjd_config: SJD configuration (if None, SJD is disabled)
            has_text_input: Whether the pipeline takes text input for world generation
            disable_diffusion_decoder: Whether to disable the diffusion decoder stage
            offload_network: Whether to offload AR model from GPU after use
            offload_guardrail_models: Whether to offload content filtering models
            offload_diffusion_decoder: Whether to offload diffusion decoder
        """
        # Initialize SJD components first (before super().__init__)
        self.sjd_config = sjd_config
        self.jacobi_generator = None
        
        if self.sjd_config is not None and self.sjd_config.enable_sjd:
            log.info(f"SJD enabled with config: {self.sjd_config}")
        else:
            log.info("SJD disabled, using standard generation")
        
        # Initialize parent class
        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            has_text_input=has_text_input,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            disable_diffusion_decoder=disable_diffusion_decoder,
            offload_guardrail_models=offload_guardrail_models,
            offload_diffusion_decoder=offload_diffusion_decoder,
        )
        
    def generate_video_from_tokens(
        self,
        prompt_tokens: list[torch.Tensor],
        latent_shape: list[int],
        video_start_boundary: int,
        max_gen_len: int,
        sampling_config: SamplingConfig,
        logit_clipping_range: list[int],
        seed: int = 0,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Function to generate video from input tokens. These input tokens can be initial text tokens (in case of text to video),
        or partial ground truth tokens.

        Handles the core token-to-video generation process:
        1. Generates new tokens using the autoregressive model
        2. Handles padding and token sequence completion
        3. Reshapes and processes generated tokens
        4. Decodes final tokens into video frames

        Args:
            model (AutoRegressiveModel): LLama model instance
            prompt_tokens (list): Prompt tokens used by the model
            latent_shape (list): Shape of the video latents
            video_start_boundary (int): Index where the video tokens start
            max_gen_len (int): Maximum length of the tokens that needs to be generated
            sampling_config (SamplingConfig): Config used by sampler during inference
            logit_clipping_range (list): Range of indices in the logits to be clipped, e.g. [video_token_start, video_token_end]
            context (Optional[torch.Tensor]): The context tensor added via cross-attn.
            context_mask (Optional[torch.Tensor]): The context cross-attn mask tensor.
        Returns:
            tuple containing:
                - List[torch.Tensor]: Generated videos
                - List[torch.Tensor]: Generated tokens
                - List[torch.Tensor]: Token index tensors
        """
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        total_seq_len = np.prod(latent_shape)

        assert not sampling_config.logprobs

        stop_tokens = self.model.tokenizer.stop_tokens
        if self.offload_tokenizer:
            self._offload_tokenizer()
        if self.offload_network:
            self._load_network()

        generation_tokens, _ = self.model.generate(
            prompt_tokens=prompt_tokens,
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            echo=sampling_config.echo,
            seed=seed,
            context=context,
            sjd_config=self.sjd_config,
            context_mask=context_mask,
            max_gen_len=max_gen_len,
            compile_sampling=sampling_config.compile_sampling,
            compile_prefill=sampling_config.compile_prefill,
            stop_tokens=stop_tokens,
            verbose=True,
        )
        generation_tokens = generation_tokens[:, video_start_boundary:]
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        if generation_tokens.shape[1] < total_seq_len:
            log.warning(
                f"Generated video tokens (shape:{generation_tokens.shape}) shorted than expected {total_seq_len}. Could be the model produce end token early. Repeat the last token to fill the sequence in order for decoding."
            )
            padding_len = total_seq_len - generation_tokens.shape[1]
            padding_tokens = generation_tokens[:, [-1]].repeat(1, padding_len)
            generation_tokens = torch.cat([generation_tokens, padding_tokens], dim=1)
        # Cast to LongTensor
        indices_tensor = generation_tokens.long()
        # First, we reshape the generated tokens into batch x time x height x width
        indices_tensor = rearrange(
            indices_tensor,
            "B (T H W) -> B T H W",
            T=latent_shape[0],
            H=latent_shape[1],
            W=latent_shape[2],
        )
        log.debug(f"generated video tokens {len(generation_tokens[0])} -> reshape: {indices_tensor.shape}")
        # If logit clipping range is specified, offset the generated indices by the logit_clipping_range[0]
        # Video decoder always takes tokens in the range (0, N-1). So, this offset is needed.
        if len(logit_clipping_range) > 0:
            indices_tensor = indices_tensor - logit_clipping_range[0]

        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._load_tokenizer()

        # Now decode the video using tokenizer.
        video_decoded = self.model.tokenizer.video_tokenizer.decode(indices_tensor.cuda())
        # Normalize decoded video from [-1, 1] to [0, 1], and clip value
        video_decoded = (video_decoded * 0.5 + 0.5).clamp_(0, 1)
        return video_decoded, indices_tensor, generation_tokens

class ARSJDVideo2WorldGenerationPipeline(ARSJDGenerationPipeline):
    """Video-to-world generation pipeline with text conditioning capabilities.

    Extends the base autoregressive generation pipeline by adding:
    - Text prompt processing and embedding
    - Text-conditioned video generation
    - Additional safety checks for text input
    - Memory management for text encoder model

    Enables generating video continuations that are guided by both
    input video frames and text descriptions.

    Additional attributes compared to ARBaseGenerationPipeline:
        offload_text_encoder_model (bool): Whether to offload text encoder from GPU after use
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        sjd_config: Optional[SJDConfig] = None,
        inference_type: str = "video2world",
        has_text_input: bool = True,
        disable_diffusion_decoder: bool = False,
        offload_guardrail_models: bool = False,
        offload_diffusion_decoder: bool = False,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
    ):
        """Initialize text-conditioned video generation pipeline.

        Args:
            checkpoint_dir: Base directory containing model checkpoints
            checkpoint_name: Name of the checkpoint to load
            sjd_config: SJD configuration
            inference_type: Type of world generation workflow
            has_text_input: Whether the pipeline takes text input for world generation
            disable_diffusion_decoder: Whether to disable diffusion decoder stage
            offload_guardrail_models: Whether to offload content filtering models
            offload_diffusion_decoder: Whether to offload diffusion decoder
            offload_network: Whether to offload AR model from GPU
            offload_tokenizer: Whether to offload tokenizer from GPU
            offload_text_encoder_model: Whether to offload text encoder
        """
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            inference_type=inference_type,
            sjd_config=sjd_config,
            has_text_input=has_text_input,
            disable_diffusion_decoder=disable_diffusion_decoder,
            offload_guardrail_models=offload_guardrail_models,
            offload_diffusion_decoder=offload_diffusion_decoder,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
        )
        self.sampling_config = SamplingConfig()
        self.offload_text_encoder_model = offload_text_encoder_model
        if not self.offload_text_encoder_model:
            self._load_text_encoder_model()

    def _run_model_with_offload(
        self,
        prompt_embedding: torch.Tensor,
        prompt_mask: torch.Tensor,
        inp_vid: torch.Tensor,
        num_input_frames: int,
        seed: int,
        sampling_config: SamplingConfig,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Run model generation with memory management.

        Executes generation process and handles model offloading to manage GPU memory.

        Args:
            prompt_embedding: Text prompt embeddings tensor
            prompt_mask: Attention mask for prompt embeddings
            inp_vid: Input video tensor
            num_input_frames: Number of input frames to use
            seed: Random seed for reproducibility
            sampling_config: Configuration for sampling parameters

        Returns:
            tuple: (
                List of generated video tensors
                List of token index tensors
                List of prompt embedding tensors
            )
        """
        out_videos, indices_tensor, prompt_embedding, _ = self._run_model(
            prompt_embedding, prompt_mask, inp_vid, num_input_frames, seed, sampling_config
        )
        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._offload_tokenizer()
        return out_videos, indices_tensor, prompt_embedding

    def _run_model(
        self,
        prompt_embedding: torch.Tensor,
        prompt_mask: torch.Tensor,
        inp_vid: torch.Tensor,
        num_input_frames: int,
        seed: int,
        sampling_config: SamplingConfig,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Run core model generation process.

        Handles text-conditioned video generation:
        1. Prepares data batch with text embeddings and video
        2. Determines appropriate context length
        3. Generates video tokens with text conditioning
        4. Processes output tensors

        Args:
            prompt_embedding: Text prompt embeddings tensor
            prompt_mask: Attention mask for prompt embeddings
            inp_vid: Input video tensor
            num_input_frames: Number of input frames to use
            seed: Random seed for reproducibility
            sampling_config: Configuration for sampling parameters,
                uses default config if None

        Returns:
            tuple: (
                List of generated video tensors
                List of token index tensors
                Text context tensor
            )
        """
        data_batch = {}
        data_batch["context"], data_batch["context_mask"] = prompt_embedding, prompt_mask
        T, H, W = self.latent_shape

        if sampling_config is None:
            sampling_config = self.sampling_config
        if type(inp_vid) is list:
            batch_size = len(inp_vid)
        elif type(inp_vid) is torch.Tensor:
            batch_size = 1
        else:
            batch_size = inp_vid.shape[0]
        data_batch["context"] = data_batch["context"].repeat(batch_size, 1, 1)
        data_batch["context_mask"] = data_batch["context_mask"].repeat(batch_size, 1)
        data_batch["context_mask"] = torch.ones_like(data_batch["context_mask"]).bool()

        latent_context_t_size = 0

        # Choosing the context length from list of available contexts
        context_used = 0
        for _clen in self._supported_context_len:
            if num_input_frames >= _clen:
                context_used = _clen
                latent_context_t_size += 1
        log.info(f"Using context of {context_used} frames")

        num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))

        data_batch["video"] = inp_vid
        data_batch["video"] = data_batch["video"].repeat(batch_size, 1, 1, 1, 1)

        data_batch = misc.to(data_batch, "cuda")

        log.debug(f"  num_tokens_to_generate: {num_gen_tokens}")
        log.debug(f"  sampling_config: {sampling_config}")
        log.debug(f"  tokenizer_config: {self.tokenizer_config}")
        log.debug(f"  latent_shape: {self.latent_shape}")
        log.debug(f"  latent_context_t_size: {latent_context_t_size}")
        log.debug(f"  seed: {seed}")

        (
            out_videos_cur_batch,
            indices_tensor_cur_batch,
        ) = self.generate_partial_tokens_from_data_batch(
            data_batch=data_batch,
            num_tokens_to_generate=num_gen_tokens,
            sampling_config=sampling_config,
            tokenizer_config=self.tokenizer_config,
            latent_shape=self.latent_shape,
            task_condition="text_and_video",
            seed=seed,
        )
        return out_videos_cur_batch, indices_tensor_cur_batch

    def generate(
        self,
        inp_prompt: str,
        inp_vid: torch.Tensor,
        num_input_frames: int = 9,
        seed: int = 0,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> np.ndarray | None:
        """Generate a video guided by text prompt and input frames.

        Pipeline steps:
        1. Validates text prompt safety if enabled
        2. Converts text to embeddings
        3. Generates video with text conditioning
        4. Enhances quality via diffusion decoder
        5. Applies video safety checks if enabled

        Args:
            inp_prompt: Text prompt to guide the generation
            inp_vid: Input video tensor with shape (batch_size, time, channels=3, height, width)
            num_input_frames: Number of frames to use as context (default: 9)
            seed: Random seed for reproducibility (default: 0)
            sampling_config: Configuration for sampling parameters,
                uses default config if None

        Returns:
            np.ndarray | None: Generated video as numpy array (time, height, width, channels)
                if generation successful, None if safety checks fail
        """
        log.info("Run guardrail on prompt")
        is_safe = self._run_guardrail_on_prompt_with_offload(inp_prompt)
        if not is_safe:
            log.critical("Input text prompt is not safe")
            return None
        log.info("Pass guardrail on prompt")

        log.info("Run text embedding on prompt")
        prompt_embeddings, prompt_masks = self._run_text_embedding_on_prompt_with_offload([inp_prompt])
        prompt_embedding = prompt_embeddings[0]
        prompt_mask = prompt_masks[0]
        log.info("Finish text embedding on prompt")

        log.info("Run generation")
        if sampling_config is None:
            sampling_config = self.sampling_config
        out_videos_cur_batch, indices_tensor_cur_batch, prompt_embedding = self._run_model_with_offload(
            prompt_embedding, prompt_mask, inp_vid, num_input_frames, seed, sampling_config
        )
        log.info("Finish AR model generation")

        if not self.disable_diffusion_decoder:
            log.info("Run diffusion decoder on generated tokens")
            out_videos_cur_batch = self._run_diffusion_decoder_with_offload(
                out_videos_cur_batch, indices_tensor_cur_batch, [prompt_embedding]
            )
            log.info("Finish diffusion decoder on generated tokens")
        out_videos_cur_batch = prepare_video_batch_for_saving(out_videos_cur_batch)
        output_video = out_videos_cur_batch[0]

        log.info("Run guardrail on generated video")
        output_video = self._run_guardrail_on_video_with_offload(output_video)
        if output_video is None:
            log.critical("Generated video is not safe")
            return None
        log.info("Finish guardrail on generated video")

        return output_video

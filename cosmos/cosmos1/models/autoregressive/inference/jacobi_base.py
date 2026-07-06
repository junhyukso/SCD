# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jacobi-based video generation demo script with SJD support."""

import argparse
import os

import imageio
import torch

from cosmos1.models.autoregressive.inference.sjd_world_generation_pipeline import ARSJDGenerationPipeline
from cosmos1.models.autoregressive.sjd import SJDConfig
from cosmos1.models.autoregressive.utils.inference import add_common_arguments, load_vision_input, validate_args
from cosmos1.utils import log


def add_sjd_arguments(parser):
    """Add SJD-specific command line arguments.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    sjd_group = parser.add_argument_group('SJD Parameters')
    
    sjd_group.add_argument(
        "--enable_sjd", 
        action="store_true", 
        default=True,
        help="Enable Speculative Jacobi Decoding"
    )
    
    sjd_group.add_argument(
        "--disable_sjd", 
        action="store_true",
        help="Disable Speculative Jacobi Decoding (overrides --enable_sjd)"
    )
    
    sjd_group.add_argument(
        "--sjd_max_tokens", 
        type=int, 
        default=16,
        help="Maximum number of tokens to generate per SJD iteration"
    )
    
    sjd_group.add_argument(
        "--multi_token_init_scheme", 
        type=str, 
        default="random",
        choices=["random", "repeat_horizon"],
        help="Token initialization scheme for SJD"
    )
    
    sjd_group.add_argument(
        "--sjd_sampler_scheme", 
        type=str, 
        default="speculative_jacobi",
        choices=["speculative_jacobi", "jacobi"],
        help="Sampling scheme for SJD"
    )
    
    sjd_group.add_argument(
        "--sjd_maximal_coupling",
        default=False,
        action="store_true",
        help="Use maximal-coupling accept/reject (default True). If not set and --no_sjd_maximal_coupling provided, disables it."
    )

    sjd_group.add_argument(
        "--sjd_guidance_scale", 
        type=float, 
        default=3.0,
        help="Guidance scale for SJD"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Jacobi video generation demo script")
    
    # Add common arguments from base.py
    add_common_arguments(parser)
    
    # Add SJD-specific arguments
    add_sjd_arguments(parser)
    
    # Add base-specific arguments
    parser.add_argument(
        "--ar_model_dir",
        type=str,
        default="Cosmos-1.0-Autoregressive-4B",
        help="Autoregressive model directory name"
    )
    
    parser.add_argument(
        "--input_type", 
        type=str, 
        default="video", 
        help="Type of input",
        choices=["image", "video"]
    )
    
    args = parser.parse_args()
    return args


def create_sjd_config(args) -> SJDConfig:
    """Create SJD configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        SJDConfig instance
    """
    # Determine if SJD should be enabled
    enable_sjd = args.enable_sjd and not args.disable_sjd
    
    sjd_config = SJDConfig(
        enable_sjd=enable_sjd,
        max_num_new_tokens=args.sjd_max_tokens,
        multi_token_init_scheme=args.multi_token_init_scheme,
        guidance_scale=args.sjd_guidance_scale,
        seed=args.seed,
        prefix_token_sampler_scheme=args.sjd_sampler_scheme,
        maximal_coupling=args.sjd_maximal_coupling,
    )
    
    return sjd_config


def main(args):
    """Run Jacobi video-to-world generation demo.
    
    This function handles the main SJD-enabled video-to-world generation pipeline:
    - Setting up SJD configuration
    - Initializing the SJD-enabled generation pipeline
    - Processing input images/videos with SJD acceleration
    - Saving generated videos
    
    Args:
        args: Parsed command line arguments containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (temperature, top_p)
            - SJD parameters (max_tokens, jacobi_interval, etc.)
            - Input/output settings (images/videos, save paths)
            - Performance options (model offloading settings)
    """
    inference_type = "base"
    sampling_config = validate_args(args, inference_type)
    
    # Create SJD configuration
    sjd_config = create_sjd_config(args)
    
    if sjd_config.enable_sjd:
        log.info("🚀 Speculative Jacobi Decoding ENABLED")
        log.info(f"   Max tokens per iteration: {sjd_config.max_num_new_tokens}")
        log.info(f"   Initialization scheme: {sjd_config.multi_token_init_scheme}")
        log.info(f"   Sampler scheme: {sjd_config.prefix_token_sampler_scheme}")
    else:
        log.info("📊 Using standard generation (SJD disabled)")
    
    # Initialize SJD-enabled generation pipeline
    pipeline = ARSJDGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.ar_model_dir,
        sjd_config=sjd_config,
        disable_diffusion_decoder=args.disable_diffusion_decoder,
        offload_guardrail_models=args.offload_guardrail_models,
        offload_diffusion_decoder=args.offload_diffusion_decoder,
        offload_network=args.offload_ar_model,
        offload_tokenizer=args.offload_tokenizer,
    )
    
    # Load input image(s) or video(s)
    input_videos = load_vision_input(
        input_type=args.input_type,
        batch_input_path=args.batch_input_path,
        input_image_or_video_path=args.input_image_or_video_path,
        data_resolution=args.data_resolution,
        num_input_frames=args.num_input_frames,
    )
    
    # Process each input
    for idx, input_filename in enumerate(input_videos):
        inp_vid = input_videos[input_filename]
        
        log.info(f"🎬 Processing: {input_filename}")
        
        if sjd_config.enable_sjd:
            log.info(f"⚡ Using SJD acceleration")
        
        # if it's already inside the video_save_folder, skip
        # if os.path.exists(os.path.join(args.video_save_folder, input_filename)):
        #     log.info(f"💾 Video already exists in {args.video_save_folder}, skipping")
        #     continue
        
        # try catch the generation
        try:
            # Generate video
            out_vid = pipeline.generate(
                inp_vid=inp_vid,
                num_input_frames=args.num_input_frames,
                seed=args.seed,
                sampling_config=sampling_config,
            )
        except Exception as e:
            log.critical(f"❌ Error generating video: {e}")
            continue
        
        if out_vid is None:
            log.critical("❌ Guardrail blocked generation.")
            continue
        
        # Save video
        if args.input_image_or_video_path:
            # Single input case
            if sjd_config.enable_sjd:
                output_name = f"{args.video_save_name}_sjd.mp4"
            else:
                output_name = f"{args.video_save_name}_standard.mp4"
            out_vid_path = os.path.join(args.video_save_folder, output_name)
        else:
            # Batch input case
            if sjd_config.enable_sjd:
                output_name = f"{idx}_sjd.mp4"
            else:
                output_name = f"{idx}_standard.mp4"
            out_vid_path = os.path.join(args.video_save_folder, output_name)
        
        # Ensure video is in correct format for imageio
        if torch.is_tensor(out_vid):
            # Convert tensor to numpy if needed
            out_vid_np = out_vid.cpu().numpy()
        else:
            out_vid_np = out_vid
            
        # Ensure video is uint8 format and in range [0, 255]
        if out_vid_np.dtype != 'uint8':
            # Assume values are in range [0, 1] and scale to [0, 255]
            out_vid_np = (out_vid_np.clip(0, 1) * 255).astype('uint8')
        
        # Convert to list of frames for imageio if needed
        if len(out_vid_np.shape) == 4:  # [T, H, W, C]
            video_frames = [out_vid_np[i] for i in range(out_vid_np.shape[0])]
        else:
            video_frames = out_vid_np
        
        # save video with its own original name
        out_vid_path = os.path.join(args.video_save_folder, input_filename)
        imageio.mimsave(out_vid_path, video_frames, fps=25)  # type: ignore
        log.info(f"💾 Saved video to {out_vid_path}")


if __name__ == "__main__":
    torch._C._jit_set_texpr_fuser_enabled(False)
    args = parse_args()
    
    print("=" * 60)
    print("🌌 Cosmos-1 Jacobi Video Generation")
    print("=" * 60)
    
    main(args) 
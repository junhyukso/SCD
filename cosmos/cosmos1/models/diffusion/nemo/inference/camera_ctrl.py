# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from einops import rearrange
from huggingface_hub import snapshot_download
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from nemo import lightning as nl
from nemo.lightning.megatron_parallel import MegatronParallel

MegatronParallel.init_ddp = lambda self: None
from nemo.collections.diffusion.data.camera_ctrl_utils import (
    estimate_pose_list_to_plucker_embedding,
    normalize_camera_trajectory_to_unit_sphere,
)
from nemo.collections.diffusion.mcore_parallel_utils import Utils
from nemo.collections.diffusion.sampler.conditioner import VideoExtendConditioner
from nemo.collections.diffusion.sampler.conditioner_configs import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
    VideoCondBoolConfig,
)
from nemo.collections.diffusion.sampler.cosmos.cosmos_extended_diffusion_pipeline import ExtendedDiffusionPipeline
from transformers import T5EncoderModel, T5TokenizerFast

from cosmos1.models.diffusion.conditioner import DataType
from cosmos1.models.diffusion.nemo.inference.inference_utils import process_prompt, save_video
from cosmos1.models.diffusion.nemo.inference.video2world import (
    compute_num_latent_frames,
    create_condition_latent_from_input_frames,
)
from cosmos1.utils import log
from cosmos1.utils.io import load_from_fileobj

EXAMPLE_PROMPT = (
    "The teal robot is cooking food in a kitchen. Steam rises from a simmering pot "
    "as the robot chops vegetables on a worn wooden cutting board. Copper pans hang "
    "from an overhead rack, catching glints of afternoon light, while a well-loved "
    "cast iron skillet sits on the stovetop next to scattered measuring spoons and "
    "a half-empty bottle of olive oil."
)

DEFAULT_AUGMENT_SIGMA_LIST = 0.001


def parse_args():
    parser = argparse.ArgumentParser(description="Video foundation model inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Cosmos-1.0-Diffusion-7B-Video2World",
        choices=["Cosmos-1.0-Diffusion-7B-Video2World", "Cosmos-1.0-Diffusion-14B-Video2World"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=EXAMPLE_PROMPT,
        help="Prompt which the sampled video condition on",
    )
    # We turn on negative prompt by default. set to "" to turn it off.
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
            "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
            "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
            "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
            "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
            "Overall, the video is of poor quality."
        ),
        help="Negative prompt which the sampled video condition on",
    )
    parser.add_argument("--subject_name", type=str, default="", help="Name of fine-tuned subject")
    parser.add_argument("--guidance", type=float, default=7, help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default="RES", help="Currently only supports RES sampler.")
    parser.add_argument("--video_save_path", type=str, default="outputs", help="Path to save the video")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the sampled video")
    parser.add_argument("--height", type=int, default=704, help="Height of image to sample")
    parser.add_argument("--width", type=int, default=1280, help="Width of image to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices for inference")
    parser.add_argument("--cp_size", type=int, default=1, help="Number of cp ranks for multi-gpu inference.")
    parser.add_argument("--num_steps", type=float, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--num_video_frames", type=int, default=121, help="Number of video frames to sample")
    parser.add_argument(
        "--num_input_frames", type=int, default=1, help="Number of video frames for conditioning, [1-9]"
    )
    parser.add_argument(
        "--conditioned_image_or_video_path", type=str, default="", help="Path to conditioning video file."
    )
    parser.add_argument(
        "--camera_extrinsics_path",
        type=str,
        default="",
        help="Path to camera intrinsics file (the videos corresponding camera.json file within the DL3DV dataset)",
    )
    parser.add_argument("--tokenizer_dir", type=str, default="", help="Directory for video tokenizer")
    parser.add_argument("--cosmos_assets_dir", type=str, default="", help="Directory containing cosmos assets")
    parser.add_argument("--prompt_upsampler_dir", type=str, default="", help="Prompt upsampler weights directory")
    parser.add_argument("--guardrail_dir", type=str, default="", help="Guardrails weights directory")
    parser.add_argument("--nemo_checkpoint", type=str, default="", help="Video diffusion model nemo weights")
    parser.add_argument("--t5_cache_dir", type=str, default=None, help="Path to T5 model")
    parser.add_argument(
        "--enable_prompt_upsampler", action="store_true", help="Whether to use prompt upsampling before generation"
    )

    args = parser.parse_args()
    return args


def print_rank_0(string: str):
    rank = torch.distributed.get_rank()
    if rank == 0:
        log.info(string)


@torch.no_grad()
def encode_for_batch(tokenizer: T5TokenizerFast, encoder: T5EncoderModel, prompts: list[str], max_length: int = 512):
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    return encoded_text


def init_video_tokenizer(args):
    """
    Initializes video tokenizer based on specified video tokenizer config / path.
    """
    from nemo.collections.diffusion.models.model import DiT7BCameraCtrlConfig

    vae_path = os.path.join(args.cosmos_assets_dir, args.tokenizer_dir)
    dit_config = DiT7BCameraCtrlConfig(vae_path=vae_path)
    vae = dit_config.configure_vae()
    return vae


def check_prompt(args):
    prompt = args.prompt
    subject_string = None
    if args.subject_name:
        subject_string = f"A video of sks {args.subject_name}"

    prompt = process_prompt(
        prompt=prompt,
        checkpoint_dir=args.cosmos_assets_dir,
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        guardrails_dir=args.guardrail_dir,
        enable_prompt_upsampler=args.enable_prompt_upsampler,
    )

    if subject_string:
        prompt = f"{subject_string}. {prompt}"
    return prompt


def read_video_or_image_into_frames_for_camera_ctrl_BCTHW(
    input_path: str,
    input_path_format: str = "mp4",
    H: int = None,
    W: int = None,
    normalize: bool = True,
    max_frames: int = -1,
    also_return_fps: bool = False,
) -> torch.Tensor:
    log.debug(f"Reading video from {input_path}")

    loaded_data = load_from_fileobj(input_path, format=input_path_format)
    frames, meta_data = loaded_data
    if input_path.endswith(".png") or input_path.endswith(".jpg") or input_path.endswith(".jpeg"):
        frames = np.array(frames[0])  # HWC, [0,255]
        if frames.shape[-1] > 3:  # RGBA, set the transparent to white
            # Separate the RGB and Alpha channels
            rgb_channels = frames[..., :3]
            alpha_channel = frames[..., 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Create a white background
            white_bg = np.ones_like(rgb_channels) * 255  # White background in RGB

            # Blend the RGB channels with the white background based on the alpha channel
            frames = (rgb_channels * alpha_channel[..., None] + white_bg * (1 - alpha_channel[..., None])).astype(
                np.uint8
            )
        frames = [frames]
        fps = 0
    else:
        fps = int(meta_data.get("fps"))

    if max_frames != -1:
        frames = frames[:max_frames]

    input_tensor = np.stack(frames, axis=0)
    input_tensor = rearrange(input_tensor, "t h w c -> t c h w")
    _, _, H_input, W_input = input_tensor.shape  # [720, 1280]

    # Aspect-ratio-aware resizing
    scaling_ratio = H / H_input
    H_target = int(scaling_ratio * H_input + 0.5)
    W_target = int(scaling_ratio * W_input + 0.5)
    if normalize:
        input_tensor = input_tensor / 128.0 - 1.0
        input_tensor = torch.from_numpy(input_tensor).bfloat16()  # TCHW
        log.debug(f"Raw data shape: {input_tensor.shape}")
        if H is not None and W is not None:
            input_tensor = transforms_F.resize(
                input_tensor,
                size=(H_target, W_target),  # type: ignore
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )

    # Reflection padding
    # [704, 1252] -> [704, 1280]
    padding_left = int((W - W_target) / 2)
    padding_right = W - W_target - padding_left
    padding_top = int((H - H_target) / 2)
    padding_bottom = H - H_target - padding_top
    padding_vals = [padding_left, padding_top, padding_right, padding_bottom]
    input_tensor = transforms_F.pad(input_tensor, padding_vals, padding_mode="reflect")

    padding_mask = torch.ones((1, H, W), dtype=torch.bfloat16)
    padding_mask[:, padding_top : (padding_top + H), padding_left : (padding_left + W)] = 0
    image_size = torch.tensor([[H, W, H_target, W_target]] * 1, dtype=torch.float16)

    input_tensor = rearrange(input_tensor, "(b t) c h w -> b c t h w", b=1)
    if normalize:
        input_tensor = input_tensor.to("cuda")
    log.debug(f"Load shape {input_tensor.shape} value {input_tensor.min()}, {input_tensor.max()}")
    if also_return_fps:
        return input_tensor, fps
    return input_tensor, padding_mask, image_size


def create_plucker_embeddings_from_pose_list(
    camera_data: dict,
    num_frames: int,
    image_size: torch.tensor,
    latent_compression_ratio_h: int = 8,
    latent_compression_ratio_w: int = 8,
):
    """
    Computes the Plücker embeddings from an input list of poses
    """
    # Prepare the instrinsics and extrinsics
    intrinsic = [
        [camera_data["fl_x"], 0, camera_data["cx"]],
        [0, camera_data["fl_y"], camera_data["cy"]],
        [0, 0, 1],
    ]

    pose_list = []
    for i, iframe in enumerate(camera_data["frames"]):
        if i < num_frames:
            rotation, translation = [], []
            # Construct the 3 X 3 rotation matrix and the 3 X 1 translation vector
            for j, row in enumerate(iframe["transform_matrix"]):
                if j < 3:
                    rotation.append(row[:3])
                    translation.append(row[-1])
            pose_list.append(
                {
                    "intrinsics": intrinsic,
                    "rotation": rotation,
                    "translation": translation,
                }
            )

    normalize_camera_trajectory_to_unit_sphere(pose_list)
    plucker_coords, plucker_embeddings_h, plucker_embeddings_w = estimate_pose_list_to_plucker_embedding(
        pose_list,
        latent_compression_ratio_h,
        latent_compression_ratio_w,
        image_size,
    )
    plucker_embeddings = rearrange(
        plucker_coords,
        "b (h w) c -> b c h w",
        h=plucker_embeddings_h,
        w=plucker_embeddings_w,
    )

    return plucker_embeddings.unsqueeze(0), plucker_embeddings_h, plucker_embeddings_w


def prepare_data_batch(args, vae, t5_embeding_max_length=512):
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b", cache_dir=args.t5_cache_dir)
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-11b", cache_dir=args.t5_cache_dir)
    text_encoder.to("cuda")
    text_encoder.eval()

    # Encode text to T5 embedding
    out = encode_for_batch(tokenizer, text_encoder, [args.prompt])[0]
    encoded_text = torch.tensor(out, dtype=torch.bfloat16)

    # Padding T5 embedding to t5_embeding_max_length
    L, C = encoded_text.shape
    t5_embed = torch.zeros(1, t5_embeding_max_length, C, dtype=torch.bfloat16)
    t5_embed[0, :L] = encoded_text

    if args.negative_prompt:
        out = encode_for_batch(tokenizer, text_encoder, [args.negative_prompt])[0]

        encoded_text = torch.tensor(out, dtype=torch.bfloat16)
        # Padding T5 embedding to t5_embeding_max_length
        L, C = encoded_text.shape
        neg_t5_embed = torch.zeros(1, t5_embeding_max_length, C, dtype=torch.bfloat16)
        neg_t5_embed[0, :L] = encoded_text
    else:
        neg_t5_embed = None

    # Prepare data sample
    t, h, w = args.num_video_frames, args.height, args.width
    state_shape = [
        vae.channel,
        vae.get_latent_num_frames(t),
        h // vae.spatial_compression_factor,
        w // vae.spatial_compression_factor,
    ]

    data_batch = {
        "video": torch.zeros((1, 3, t, h, w), dtype=torch.uint8).cuda(),
        "t5_text_embeddings": t5_embed,
        "t5_text_mask": torch.ones(1, t5_embeding_max_length, dtype=torch.bfloat16).cuda(),
        "fps": torch.tensor([args.fps] * 1, dtype=torch.bfloat16).cuda(),
        "num_frames": torch.tensor([args.num_video_frames] * 1, dtype=torch.bfloat16).cuda(),
    }
    if args.negative_prompt:
        data_batch["neg_t5_text_embeddings"] = neg_t5_embed
        data_batch["neg_t5_text_mask"] = torch.ones(1, t5_embeding_max_length, dtype=torch.bfloat16)

    input_path_format = args.conditioned_image_or_video_path.split(".")[-1]
    input_frames, padding_mask, image_size = read_video_or_image_into_frames_for_camera_ctrl_BCTHW(
        args.conditioned_image_or_video_path,
        input_path_format=input_path_format,
        H=args.height,
        W=args.width,
    )
    data_batch["image_size"] = image_size

    condition_latent, _ = create_condition_latent_from_input_frames(vae, input_frames, args.num_input_frames)
    data_batch["condition_latent"] = condition_latent

    with open(args.camera_extrinsics_path, "r") as fp:
        camera_data = json.load(fp)

    plucker_embeddings, plucker_embeddings_h, plucker_embeddings_w = create_plucker_embeddings_from_pose_list(
        camera_data,
        args.num_video_frames,
        image_size[0],
    )

    padding_mask = padding_mask.unsqueeze(0)  # 1,1,H_input,W_input
    padding_mask = torch.nn.functional.interpolate(
        padding_mask, size=(plucker_embeddings_h, plucker_embeddings_w), mode="nearest"
    )
    data_batch["padding_mask"] = padding_mask

    plucker_embeddings = plucker_embeddings * (1 - padding_mask)

    data_batch["plucker_embeddings"] = plucker_embeddings

    return data_batch, state_shape


def setup_diffusion_pipeline(args):
    """
    Initialize DiT model, parallel strategy, and diffusion pipeline for inference.
    """
    # Initialize DiT model
    from nemo.collections.diffusion.models.model import DiT7BCameraCtrlConfig, DiTModel

    dit_config = DiT7BCameraCtrlConfig()
    dit_model = DiTModel(dit_config)

    # Initialize model parallel strategy. Here, we only use context parallel.
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_strictness="log_all",
    )

    # Initialize ptl trainer
    trainer = nl.Trainer(
        devices=args.num_devices,  # you can change the numebr of devices to suit your setup
        max_steps=1,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    # Convert trainer to fabric for inference
    fabric = trainer.to_fabric()
    fabric.strategy.checkpoint_io.save_ckpt_format = "zarr"
    fabric.strategy.checkpoint_io.validate_access_integrity = False
    print(f"Loading weights from {args.nemo_checkpoint}")
    model = fabric.load_model(args.nemo_checkpoint, dit_model).to(device="cuda", dtype=torch.bfloat16)

    # Set up diffusion pipeline
    conditioner = VideoExtendConditioner(
        text=TextConfig(),
        fps=FPSConfig(),
        num_frames=NumFramesConfig(),
        image_size=ImageSizeConfig(),
        padding_mask=PaddingMaskConfig(),
        video_cond_bool=VideoCondBoolConfig(add_pose_condition=True),
    )

    diffusion_pipeline = ExtendedDiffusionPipeline(
        net=model.module, conditioner=conditioner, sampler_type=args.sampler, seed=args.seed
    )
    diffusion_pipeline.conditioner.data_type = DataType.VIDEO

    return diffusion_pipeline


def run_diffusion_inference(args, data_batch, state_shape, vae, diffusion_pipeline):
    # prepare data
    data_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data_batch.items()}
    data_batch["inference_fwd"] = True
    augment_sigma = DEFAULT_AUGMENT_SIGMA_LIST
    num_of_latent_condition = compute_num_latent_frames(vae, args.num_input_frames)

    sample = diffusion_pipeline.generate_samples_from_batch(
        data_batch,
        guidance=args.guidance,
        state_shape=state_shape,
        num_steps=args.num_steps,
        is_negative_prompt=True if "neg_t5_text_embeddings" in data_batch else False,
        num_condition_t=num_of_latent_condition,
        condition_latent=data_batch["condition_latent"],
        condition_video_augment_sigma_in_inference=augment_sigma,
    )

    rank = torch.distributed.get_rank()
    if rank == 0:
        # Post-processing and save video
        sigma_data = 0.5
        grid = (1.0 + vae.decode(sample / sigma_data)).clamp(0, 2) / 2
        grid = (grid[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)
        save_video(
            grid=grid,
            fps=args.fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=args.video_save_path,
            checkpoint_dir=args.cosmos_assets_dir,
            guardrails_dir=args.guardrail_dir,
        )
        print_rank_0(f"saved video to {args.video_save_path}!")


def main(args):
    if args.guardrail_dir == "":
        args.guardrail_dir = snapshot_download("nvidia/Cosmos-1.0-Guardrail")
    if args.tokenizer_dir == "":
        args.tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    if args.prompt_upsampler_dir == "" and args.enable_prompt_upsampler:
        args.prompt_upsampler_dir = snapshot_download("nvidia/Cosmos-1.0-Prompt-Upsampler-12B-Text2World")
    if args.nemo_checkpoint == "":
        args.nemo_checkpoint = snapshot_download(f"nvidia/{args.model}", allow_patterns=["nemo/*"])
        args.nemo_checkpoint = os.path.join(args.nemo_checkpoint, "nemo")

    # Initialize megatron model parallel environment
    Utils.initialize_distributed(1, 1, context_parallel_size=args.cp_size)
    model_parallel_cuda_manual_seed(args.seed)

    # args.prompt = check_prompt(args)

    # Load video tokenizer
    print_rank_0("initializing video tokenizer...")
    vae = init_video_tokenizer(args)

    # Prepare data batch
    print_rank_0("preparing data batch...")
    data_batch, state_shape = prepare_data_batch(args, vae)

    # Setup model / diffusion pipeline
    print_rank_0("setting up diffusion pipeline...")
    diffusion_pipeline = setup_diffusion_pipeline(args)

    # Generate video from prompt
    print_rank_0("generating video...")
    run_diffusion_inference(args, data_batch, state_shape, vae, diffusion_pipeline)


if __name__ == "__main__":
    args = parse_args()
    main(args)

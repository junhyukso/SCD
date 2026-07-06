## Cosmos-1 Speculative Jacobi Decoding (SJD) Demo

This repository contains a minimal, runnable demo of Speculative Jacobi Decoding (SJD) on top of Cosmos‑1 autoregressive generation. It includes a shell script to launch video reasoning/generation, and the core SJD implementation embedded inside the autoregressive model.

The demo is optimized to be run from the repository root. If you run from elsewhere, make sure your `PYTHONPATH` includes the repo root so Python can import `cosmos1/...` modules.

---

### What’s inside
- **Demo launcher**: `run_jacobi_demo.sh`
- **Core AR model**: `cosmos1/models/autoregressive/model.py`
- **SJD implementation**: `cosmos1/models/autoregressive/sjd/`
  - `speculative_sampler.py`: Speculative acceptance/rejection for advanced tokens
  - `multi_token_utils.py`: Multi‑token initialization, KV‑cache rollback utilities
  - `sjd_config.py`: SJD configuration dataclass and defaults
  - `speculative_sampler_gsd.py`: Alternative sampler variants and utilities

---

## Installation

### Option A: Docker (recommended)
The repo includes a `Dockerfile` based on NVIDIA PyTorch. You need an NVIDIA GPU and the NVIDIA Container Toolkit.

```bash
docker build -t cosmos .
docker run -d --name cosmos_container --gpus all --ipc=host -it -v $(pwd):/workspace cosmos
docker attach cosmos_container
```

### Option B: Native (system Python)
Prerequisites:
- Linux (tested on Ubuntu 20.04/22.04/24.04)
- NVIDIA GPU with recent drivers and CUDA-supported PyTorch
- ffmpeg

Then install Python deps:
```bash
pip install --no-cache-dir -r requirements.txt
```

---

## Quick start

Always run from the repo root and export `PYTHONPATH` so the `cosmos1/...` imports resolve:

```bash
cd /workspace  # repo root
export PYTHONPATH="/workspace:${PYTHONPATH}"
```

Now launch the demo:

```bash
bash run_jacobi_demo.sh
```

This will run the Jacobi demo over a single input video and save the output video to the configured output folder.

---

## Configuring `run_jacobi_demo.sh`

The script constructs and runs:

```bash
python cosmos1/models/autoregressive/inference/jacobi_base.py ...
```

It exposes several environment variables you can override at runtime:

- **MODEL_DIR**: AR model directory name or path (default: `Cosmos-1.0-Autoregressive-4B`)
- **INPUT_VIDEO**: Absolute path to the input video file
- **OUTPUT_NAME**: Output basename (default: `demo_output`)
- **OUTPUT_FOLDER**: Output directory (default: `outputs_tmp`)
- **ENABLE_SJD**: `true` to enable SJD; `false` for standard decoding (default: `true`)
- **MAX_TOKENS**: Max speculative tokens per SJD iteration (default: `64`)
- **MAXIMAL_COUPLING**: Enable maximal coupling in sampler (default: `true`)

Important: The script includes placeholder paths you must update before running:

- `INPUT_VIDEO` default uses `/[PATH]/cosmos/real-state-10k/845fcf8e2d6efde7.mp4` — replace with your real video path.
- `--checkpoint_dir /[PATH]/cosmos/checkpoints` — replace with your local checkpoints directory.

You can either edit the script or export variables to override defaults. Example:

```bash
export PYTHONPATH="/workspace:${PYTHONPATH}"
export MODEL_DIR="/data/models/Cosmos-1.0-Autoregressive-4B"
export INPUT_VIDEO="/data/videos/sample.mp4"
export OUTPUT_NAME="my_demo"
export OUTPUT_FOLDER="outputs_tmp"
export ENABLE_SJD=true
export MAX_TOKENS=64
export MAXIMAL_COUPLING=true

bash run_jacobi_demo.sh
```

Outputs are written to the folder passed as `--video_save_folder` (default `outputs_tmp`) with the `--video_save_name` prefix.

---

## SJD implementation overview

The SJD path is integrated directly into the autoregressive decoding loop:

- `cosmos1/models/autoregressive/model.py`
  - Method `AutoRegressiveModel.generate(...)`: orchestrates both standard decoding and the SJD branch.
  - When SJD is enabled, it prepares draft tokens, runs a single forward pass to get advanced logits, performs vectorized sampling, invokes the speculative sampler for acceptance, rolls back unused KV‑cache, and updates the sequence.

- `cosmos1/models/autoregressive/sjd/`
  - `sjd_config.py`: Defines `SJDConfig` (e.g., `enable_sjd`, `max_num_new_tokens`, `multi_token_init_scheme`, `maximal_coupling`).
  - `multi_token_utils.py`: Provides `get_multi_token_initialization(...)`, `find_first_mismatch(...)`, and `rollback_kv_cache(...)` for efficient multi‑token drafting and cache management.
  - `speculative_sampler.py`: Implements `CosmosSpeculativeSampler`, the acceptance/rejection step that couples draft and advanced tokens; supports maximal coupling.
  - `speculative_sampler_gsd.py`: Houses additional sampler utilities/variants used in experimentation.

At a high level per SJD iteration:
1. Reuse leftover advanced tokens from the previous step and fill the rest with new draft tokens.
2. Run a single forward pass to obtain logits for all speculative positions.
3. Sample advanced tokens (vectorized top‑p or multinomial).
4. Use the speculative sampler to accept a prefix of those tokens.
5. Roll back the model KV‑cache for tokens that were not accepted.
6. Commit accepted tokens to the sequence; carry over leftovers for the next iteration.

---

## Troubleshooting

- Module import errors like `ModuleNotFoundError: No module named 'cosmos1'`
  - Ensure you are at the repo root and set `export PYTHONPATH="/workspace:${PYTHONPATH}"`.

- Missing or wrong checkpoint path
  - Edit `run_jacobi_demo.sh` to replace `--checkpoint_dir /[PATH]/cosmos/checkpoints` with your local directory.

- Video I/O errors
  - Install `ffmpeg` (Dockerfile does this automatically). Verify the video path and file permissions.

- GPU / CUDA issues
  - Confirm you’re running inside an NVIDIA‑enabled container (or native env) with a compatible PyTorch + CUDA stack.

---

## License and attribution

This repository contains code under the Apache 2.0 License with NVIDIA copyright notices where applicable. See headers in individual files for details.

# [ICML2026] Speculative Coupled Decoding

[![arXiv](https://img.shields.io/badge/arXiv-2510.24211-b31b1b.svg)](https://arxiv.org/abs/2510.24211)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)

Official implementation for **Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation**, accepted to **ICML 2026**.

Speculative Coupled Decoding (SCD) is a training-free decoding framework for accelerating autoregressive visual generation. It extends Speculative Jacobi Decoding (SJD) with coupling-based draft token sampling, improving token acceptance while preserving the target autoregressive distribution. The method is designed to reduce the number of forward evaluations (NFE) needed for image and video generation without training an auxiliary draft model.

## News

- **ICML 2026**: SCD was accepted to ICML 2026.
- **Paper**: [arXiv:2510.24211](https://arxiv.org/abs/2510.24211)

## Repository Layout

```text
.
|-- lumina1/        # SCD and SJD decoding for Lumina-mGPT 1.0 image generation
|-- janus_pro/      # Janus-Pro code
|-- cosmos/         # Cosmos code
```

This README currently documents the **Lumina-mGPT 1.0** workflow for image AR generation. Check other folders for usage instructions for the other models.

## Lumina-mGPT 1.0 Usage

### 1. Prepare Chameleon Tokenizer Files


Our model uses the image tokenizer from Meta's Chameleon. Please download the necessary files from [Meta's Chameleon homepage](https://ai.meta.com/resources/models-and-libraries/chameleon-downloads/).

Place the downloaded files into the `ckpts/chameleon/tokenizer/` directory. The final folder structure should look like this:

```text
lumina1/
`-- ckpts/
    `-- chameleon/
        `-- tokenizer/
            |-- checklist.chk
            |-- text_tokenizer.json
            |-- vqgan.ckpt
            `-- vqgan.yaml
```

The image decoder needs both `vqgan.ckpt` and `vqgan.yaml`; 

### 2. Install Dependencies

This  scripts were written for PyTorch 2.3+ and Hugging Face Transformers 4.48.1:

```bash
pip install "torch>=2.3.0" transformers==4.48.1 "accelerate>=0.26.0" sentencepiece absl-py pillow numpy einops torchvision tqdm
```

The scripts load Lumina-mGPT from Hugging Face, for example `Alpha-VLLM/Lumina-mGPT-7B-768`. The first run may download model weights unless they are already cached.

### 3. Run Image Generation

From the repository root:

```bash
cd lumina1
```

Run SCD with Gumbel coupling:

```bash
python test_SCD_GS.py
```

Run SCD with maximal coupling:

```bash
python test_SCD_MC.py
```

Run the SJD baseline:

```bash
python test_SJD.py
```

The default scripts generate a list of qualitative prompts at `768x768`. To run a smaller smoke test or change the prompt, edit `q_image_content_conditions` inside the corresponding script.

### 4. Outputs 

Generated images are saved under timestamped folders:

```text
lumina1/SCD_GS/<timestamp>/
lumina1/SCD_MC/<timestamp>/
lumina1/SJD/<timestamp>/
```

The scheduler prints runtime statistics, including the total number of forward evaluations:


## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{so2026speculative,
  title     = {Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation},
  author    = {So, Junhyuk and Kook, Hyunho and Jang, Chaeyeon and Park, Eunhyeok},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}
```


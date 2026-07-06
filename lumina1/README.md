# Implementation of SCD for Lumina-mGPT 1.0

## Getting Started

### 1. Download  Tokenizer

This model relies on the image tokenizer from Meta's Chameleon. You'll need to obtain the required files by visiting [Meta's Chameleon homepage](https://ai.meta.com/resources/models-and-libraries/chameleon-downloads/).

After downloading, arrange the files in the following directory structure. Your folder layout should match this exactly:

    ckpts/
    └── chameleon/
        └── tokenizer/
            ├── checklist.chk
            ├── text_tokenizer.json
            ├── vqgan.ckpt
            └── vqgan.yaml

### 2. Install Dependencies

For best results, we advise using **PyTorch 2.3.0 or newer**. You can install all necessary packages with the following pip command:

    pip install transformers==4.48.1 sentencepiece accelerate>=0.26.0 absl-py

### 3. Running Inference

#### Test the GSD Model

To generate images using our **SCD* model, execute the appropriate script below.

1.  For Maximal Coupling:
    ```bash
    python test_SCD_MC.py
    ```

2.  For Gumbel Coupling:
    ```bash
    python test_SCD_GS.py
    ```

*Note: You can customize the image generation prompt by editing the value directly inside the Python file.*


#### Test the SJD Baseline

To benchmark performance against the **SJD (ICLR 2025) baseline** model, run this script:

    python test_SJD.py

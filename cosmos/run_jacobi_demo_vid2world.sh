#!/bin/bash

# Cosmos-1 Jacobi (SJD) Demo Script for Video-to-World
# This script demonstrates how to use Speculative Jacobi Decoding with Cosmos-1 for prompted video generation.

echo "======================================================"
echo "🌌 Cosmos-1 Jacobi Video-to-World Demo"
echo "======================================================"

# Set default values
MODEL_DIR=${MODEL_DIR:-"Cosmos-1.0-Autoregressive-5B-Video2World"}
INPUT_VIDEO=${INPUT_VIDEO:-"cosmos1/models/autoregressive/assets/v1p0/input.mp4"}
PROMPT=${PROMPT:-"A video recorded from a moving vehicle's perspective, capturing roads, buildings, landscapes, and changing weather and lighting conditions."}
OUTPUT_NAME=${OUTPUT_NAME:-"demo_output-5b-video2world"}
ENABLE_SJD=${ENABLE_SJD:-true}
MAX_NUM_NEW_TOKENS=${MAX_NUM_NEW_TOKENS:-64}
MAXIMAL_COUPLING=${MAXIMAL_COUPLING:-true}


echo "📋 Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Input Video: $INPUT_VIDEO"
echo "  Prompt: $PROMPT"
echo "  Output Name: $OUTPUT_NAME"
echo "  SJD Enabled: $ENABLE_SJD"
echo ""

# Build the command
CMD="python cosmos1/models/autoregressive/inference/jacobi_video2world.py"
CMD="$CMD --input_type=text_and_video"
CMD="$CMD --input_image_or_video_path=$INPUT_VIDEO"
CMD="$CMD --prompt=\"$PROMPT\""
CMD="$CMD --video_save_name=$OUTPUT_NAME"
CMD="$CMD --ar_model_dir=$MODEL_DIR"
CMD="$CMD --top_p=0.7"
CMD="$CMD --temperature=1.0"
CMD="$CMD --checkpoint_dir /[PATH]/cosmos/checkpoints"
CMD="$CMD --offload_guardrail_models"
# CMD="$CMD --offload_diffusion_decoder"
# CMD="$CMD --offload_network"
# CMD="$CMD --offload_tokenizer"
CMD="$CMD --offload_text_encoder_model"

# Add SJD parameters
if [ "$ENABLE_SJD" = true ]; then
    echo "⚡ SJD parameters:"
    echo "  Max new tokens per iteration: $MAX_NUM_NEW_TOKENS"
    echo ""
    
    CMD="$CMD --enable_sjd"
    CMD="$CMD --max_num_new_tokens=$MAX_NUM_NEW_TOKENS"
    if [ "$MAXIMAL_COUPLING" = true ]; then
        CMD="$CMD --sjd_maximal_coupling"
    fi
else
    echo "📊 Using standard generation (SJD disabled)"
    echo ""
fi

echo "🚀 Running command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Generation completed successfully!"
    echo "📁 Output saved to: outputs/${OUTPUT_NAME}.mp4"
else
    echo ""
    echo "❌ Generation failed!"
    exit 1
fi
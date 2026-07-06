#!/bin/bash

# Cosmos-1 Jacobi (SJD) Demo Script
# This script demonstrates how to use Speculative Jacobi Decoding with Cosmos-1

echo "=================================="
echo "🌌 Cosmos-1 Jacobi Demo"
echo "=================================="

# Set default values
MODEL_DIR=${MODEL_DIR:-"Cosmos-1.0-Autoregressive-4B"}
# INPUT_VIDEO=${INPUT_VIDEO:-"/[PATH]/cosmos/real-state-10k/base.jsonl"}
INPUT_VIDEO=${INPUT_VIDEO:-"/[PATH]/cosmos/real-state-10k/845fcf8e2d6efde7.mp4"}
OUTPUT_NAME=${OUTPUT_NAME:-"demo_output"}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-"outputs_tmp"}
ENABLE_SJD=${ENABLE_SJD:-true}
MAX_TOKENS=${MAX_TOKENS:-64}
MAXIMAL_COUPLING=${MAXIMAL_COUPLING:-true}

echo "📋 Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Input: $INPUT_VIDEO"
echo "  Output: $OUTPUT_NAME"
echo "  Output Folder: $OUTPUT_FOLDER"
echo "  SJD Enabled: $ENABLE_SJD"
echo ""

# Build the command
CMD="python cosmos1/models/autoregressive/inference/jacobi_base.py"
CMD="$CMD --input_type=video"
# CMD="$CMD --batch_input_path=$INPUT_VIDEO"
CMD="$CMD --input_image_or_video_path=$INPUT_VIDEO"
CMD="$CMD --video_save_name=$OUTPUT_NAME"
CMD="$CMD --video_save_folder=$OUTPUT_FOLDER"
CMD="$CMD --ar_model_dir=$MODEL_DIR"
CMD="$CMD --top_p=0.8"
CMD="$CMD --temperature=1.0"
CMD="$CMD --checkpoint_dir /[PATH]/cosmos/checkpoints"

# Add SJD parameters
if [ "$ENABLE_SJD" = true ]; then
    echo "⚡ SJD parameters:"
    echo "  Max tokens per iteration: $MAX_TOKENS"
    echo "  Initialization: random"
    echo "  Sampler: speculative_jacobi"
    echo ""
    
    CMD="$CMD --enable_sjd"
    CMD="$CMD --sjd_max_tokens=$MAX_TOKENS"
    CMD="$CMD --multi_token_init_scheme=random"
    CMD="$CMD --sjd_sampler_scheme=speculative_jacobi"
    if [ "$MAXIMAL_COUPLING" = true ]; then
        CMD="$CMD --sjd_maximal_coupling"
    fi
else
    echo "📊 Using standard generation (SJD disabled)"
    echo ""
    CMD="$CMD --disable_sjd"
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
    echo "📁 Output saved to: outputs/${OUTPUT_NAME}*.mp4"
else
    echo ""
    echo "❌ Generation failed!"
    exit 1
fi 
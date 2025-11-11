#!/bin/bash

# Zero-shot evaluation script for external validation (Image-only model)
# This script runs test_external.py for all folds using image-bins76 model

echo "Starting zero-shot evaluation for image-bins76 model..."

# Get current working directory
CURRENT_DIR="$(pwd)"
echo "Current working directory: $CURRENT_DIR"

# Configuration
CONFIG_FILE="configs/ext_val_image.yaml"
OUTPUT_DIR="external_validation_proceed"
BINS=76
MODEL_TYPE="image-bins${BINS}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $CURRENT_DIR/$OUTPUT_DIR"

# Function to run evaluation for a dataset type
run_evaluation() {
    local EXTERNAL_TYPE="eyedye" # 'eyedye' only
    local RESULTS_FILE="${OUTPUT_DIR}/results_zero_shot_${EXTERNAL_TYPE}_${MODEL_TYPE}_combined.xlsx"

    echo ""
    echo "=== Evaluating ${EXTERNAL_TYPE} dataset (all folds) using ${MODEL_TYPE} model ==="

    # Run evaluation for all folds at once (test_external.py handles all 5 folds automatically)
    echo "Running zero-shot evaluation for ${EXTERNAL_TYPE} with ${MODEL_TYPE} model..."

    # Run test_external.py (it will process all 5 folds automatically)
    if python test_external.py \
        --config "$CONFIG_FILE" \
        --bins "$BINS" \
        --external_type "$EXTERNAL_TYPE"; then
        echo "  Successfully evaluated all folds for ${EXTERNAL_TYPE}"
    else
        echo "  Error evaluating ${EXTERNAL_TYPE}"
        return 1
    fi

    echo "Combining results for ${EXTERNAL_TYPE}..."
    echo "Combined results will be saved to: $CURRENT_DIR/$RESULTS_FILE"

    # Combine results using combine_results.py directly
    python combine_results.py logs/test/${MODEL_TYPE}-ext-${EXTERNAL_TYPE}-zero-fold*/results.xlsx -o "$RESULTS_FILE"

    if [ $? -eq 0 ]; then
        echo "Successfully combined ${EXTERNAL_TYPE} results into $RESULTS_FILE"
    else
        echo "Error combining ${EXTERNAL_TYPE} results"
        return 1
    fi
}

# Run evaluation for eyedye only (as specified in requirements)
run_evaluation

echo ""
echo "Zero-shot evaluation for ${MODEL_TYPE} model completed for eyedye dataset!"
echo "Results saved to:"
echo "  - Eyedye: $CURRENT_DIR/${OUTPUT_DIR}/results_zero_shot_eyedye_${MODEL_TYPE}_combined.xlsx"
echo ""
echo "Model checkpoints used:"
echo "  - logs/train/image-bins${BINS}-fold0/best.ckpt"
echo "  - logs/train/image-bins${BINS}-fold1/best.ckpt"  
echo "  - logs/train/image-bins${BINS}-fold2/best.ckpt"
echo "  - logs/train/image-bins${BINS}-fold3/best.ckpt"
echo "  - logs/train/image-bins${BINS}-fold4/best.ckpt"
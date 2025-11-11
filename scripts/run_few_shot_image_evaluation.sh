#!/bin/bash

# Few-shot evaluation script for external validation (Image-only model)
# This script performs few-shot training and evaluation on eyedye dataset
# using image-bins76 model with backbone frozen

echo "Starting few-shot evaluation for image-bins76 model..."
echo "Cwd: $(pwd)"

# Model configuration
BINS=76 # image-bins76 모델 사용
MODEL_TYPE="image-bins${BINS}"
EPOCHS=60
start_ib_epoch=31
CONFIG="configs/ext_val_image.yaml"
OUTPUT_DIR="external_validation_proceed"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Model: $MODEL_TYPE, Bins: $BINS, Epochs: $EPOCHS"

# Function to run few-shot training for a specific dataset (all folds)
run_few_shot_train() {
    local dataset_type=$1
    
    echo "Running few-shot training for $dataset_type dataset (all folds) using $MODEL_TYPE model..."
    
    python train_external.py \
        --config $CONFIG \
        --bins $BINS \
        --external_type $dataset_type \
        --device 0
    
    if [ $? -ne 0 ]; then
        echo "Error: Few-shot training failed for $dataset_type"
        return 1
    fi
    
    echo "Successfully completed few-shot training for $dataset_type (all folds)"
    return 0
}

# Function to run evaluation for a specific dataset (all folds)
run_few_shot_test() {
    local dataset_type=$1
    
    echo "Running few-shot evaluation for $dataset_type dataset (all folds) using $MODEL_TYPE model..."
    
    python test_external.py \
        --config $CONFIG \
        --bins $BINS \
        --is_few_shot \
        --external_type $dataset_type \
        --device 0
    
    if [ $? -ne 0 ]; then
        echo "Error: Few-shot evaluation failed for $dataset_type"
        return 1
    fi
    
    echo "Successfully completed few-shot evaluation for $dataset_type (all folds)"
    return 0
}

# Main execution
echo "=========================================="
echo "Starting Few-shot Training and Evaluation"
echo "Model: $MODEL_TYPE"
echo "=========================================="

# Process eyedye dataset only (as specified in requirements)
DATASET="eyedye"

echo ""
echo "Processing $DATASET dataset..."
echo "=========================================="

# Few-shot training (all folds)
if ! run_few_shot_train $DATASET; then
    echo "Skipping evaluation for $DATASET due to training failure"
    exit 1
fi

# Few-shot evaluation (all folds)
if ! run_few_shot_test $DATASET; then
    echo "Evaluation failed for $DATASET"
    exit 1
fi

echo "Successfully completed $DATASET dataset"

echo ""
echo "Combining results for $DATASET..."

# Combine results using combine_results.py directly
RESULTS_FILE="${OUTPUT_DIR}/results_few_shot_${DATASET}_${MODEL_TYPE}_combined.xlsx"
python combine_results.py logs/test/${MODEL_TYPE}-ext-${DATASET}-few-fold*/results.xlsx -o "$RESULTS_FILE"

if [ $? -eq 0 ]; then
    echo "Combined results saved to: $(pwd)/$RESULTS_FILE"
else
    echo "Error combining $DATASET results"
    exit 1
fi

echo ""
echo "=========================================="
echo "Few-shot evaluation completed!"
echo "=========================================="
echo "Results saved to:"
echo "  - Eyedye: $(pwd)/$RESULTS_FILE"
echo ""
echo "Training logs saved to:"
echo "  - logs/train/${MODEL_TYPE}-ext-${DATASET}-fold*/"
echo ""
echo "Evaluation logs saved to:"
echo "  - logs/test/${MODEL_TYPE}-ext-${DATASET}-few-fold*/"
echo ""
echo "Model checkpoints used:"
echo "  - logs/train/image-bins${BINS}-fold0/best.ckpt"
echo "  - logs/train/image-bins${BINS}-fold1/best.ckpt"  
echo "  - logs/train/image-bins${BINS}-fold2/best.ckpt"
echo "  - logs/train/image-bins${BINS}-fold3/best.ckpt"
echo "  - logs/train/image-bins${BINS}-fold4/best.ckpt"
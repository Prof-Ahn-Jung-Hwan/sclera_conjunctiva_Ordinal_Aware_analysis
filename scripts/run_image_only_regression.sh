#!/bin/bash

# This script trains and evaluates the purest form of 'image-only regression' model.
# - Uses the same train/test data split as the baseline model in the ablation study.
# Stops immediately if an error occurs during script execution.
set -e

# Set paths based on the directory where the script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}" # Change working directory to script directory

# --- (1) List of models to be baselines for Ablation Study ---
# Performs image-only regression using train/test split from image-bins76 model.
BASELINE_MODELS=(
    "image-bins76"
)

# GPU configuration (using 2 GPUs)
GPU_POOL=(0 1)
MAX_CONCURRENT_JOBS=${#GPU_POOL[@]}

# --- (3) Single fold execution function ---
run_single_regression_fold() {
    local fold=$1
    local gpu_id=$2
    local exp_name=$3
    local baseline_exp_name=$4

    echo "--- Starting: ${EXP_NAME} Fold ${fold} on GPU ${gpu_id} ---"

    local baseline_log_dir="logs/train/${baseline_exp_name}-fold${fold}"
    local train_file="${baseline_log_dir}/train.txt"
    local test_file="${baseline_log_dir}/test.txt"

    # Training (using train_regression.py)
    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n torch271 --no-capture-output python3 train_regression.py \
        --config configs/ajoumc_rxt50_image_regression.yaml \
        --fold ${fold} \
        --exp-name "${exp_name}" \
        --train-file "${train_file}" \
        --test-file "${test_file}" \
        --device 0

    # Testing (using test_regression.py)
    local train_log_dir="logs/train/${exp_name}-fold${fold}"
    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n torch271 --no-capture-output python3 test_regression.py \
        --config configs/ajoumc_rxt50_image_regression.yaml \
        --exp-name "${exp_name}" \
        --fold ${fold} \
        --ckpt "${train_log_dir}/best.ckpt" \
        --test-file "${test_file}"

    echo "--- Finished: ${EXP_NAME} Fold ${fold} ---"
}

# --- (4) Main execution loop ---
for baseline_exp_name in "${BASELINE_MODELS[@]}"; do
    # Generate unique experiment name derived from image-bins76
    # Example: image-bins76 -> image-regression-from-bins76
    baseline_suffix=$(echo "${baseline_exp_name}" | sed 's/image-//')
    EXP_NAME="image-regression-from-${baseline_suffix}"

    echo "======================================================================"
    echo "===== Starting Pure Image-Only Regression Study"
    echo "===== Using data split from: ${baseline_exp_name}"
    echo "===== Saving results as: ${EXP_NAME}"
    echo "======================================================================"
    
    # --- 5-Fold parallel execution ---
    job_count=0
    gpu_idx=0

    for fold in {0..4}; do
        GPU_ID=${GPU_POOL[${gpu_idx}]}
        # Pass both generated unique experiment name (EXP_NAME) and baseline name (baseline_exp_name)
        run_single_regression_fold ${fold} ${GPU_ID} "${EXP_NAME}" "${baseline_exp_name}" &
        
        job_count=$((job_count + 1))
        gpu_idx=$(( (gpu_idx + 1) % MAX_CONCURRENT_JOBS ))
        if [ ${job_count} -ge ${MAX_CONCURRENT_JOBS} ]; then
            wait -n
            job_count=$((job_count - 1))
        fi
    done
    wait # Wait until all fold jobs are completed

    # --- Combine results ---
    echo "--- Combining Results for ${EXP_NAME} ---"
    COMBINED_FILENAME="results_${EXP_NAME}_combined.xlsx"
    
    # Find result files for all folds using find command.
    file_list=$(find logs/test -path "*/${EXP_NAME}-fold*/results.xlsx")
    if [ -z "${file_list}" ]; then
        echo "Warning: No result files found for ${EXP_NAME}. Skipping combination."
        continue # Move to next baseline model
    fi
    
    conda run -n torch271 --no-capture-output python3 combine_results.py ${file_list} -o "${COMBINED_FILENAME}"
    echo "Combined results saved to: ${COMBINED_FILENAME}"
done

echo -e "\n\n🎉 All regression-only studies are complete."

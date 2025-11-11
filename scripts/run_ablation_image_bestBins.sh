#!/bin/bash

# Image-only model (image + or + ve + ibfc) Ablation Study script
# Based on best bins 76, but can test other values by adjusting bins

set -e

# --- (1) List of Bins values to analyze ---
# Best bins is 76, but configured as array to allow testing other values
BINS_TO_ANALYZE=(
    "76"  # Best bins from Bayesian optimization
    # "83"  # Uncomment to test additional value
    # "64"  # Uncomment to test additional value
)

# --- (2) Common settings ---
# Define ablation scenarios (using only bins value, excluding demo, n)
declare -a scenarios
scenarios[0]="--no_enc"           # 1. No Variant Encoder (VE=0, OR=1)
scenarios[1]="--no_ord"           # 2. No Ordinal Regression (VE=1, OR=0)
scenarios[2]="--no_enc --no_ord"  # 3. Regression Only (VE=0, OR=0)

# GPU configuration
GPU_POOL=(0 1)
MAX_CONCURRENT_JOBS=${#GPU_POOL[@]}

echo "############################################################"
echo "##### 🚀 Image-only Model Ablation Study"
echo "##### Model: image + or + ve + ibfc (no tab data)"
echo "##### Config: configs/ajoumc_rxt50_image.yaml"
echo "##### Bins to analyze: ${BINS_TO_ANALYZE[*]}"
echo "############################################################"

# --- (3) Single experiment execution function ---
run_single_ablation() {
    local fold=$1
    local flags=$2
    local gpu_id=$3
    # Use variables from outside function
    local base_exp_name="${CURRENT_BASE_EXP_NAME}"
    local config_file="${CURRENT_CONFIG_FILE}"
    local bins_value="${CURRENT_BINS_VALUE}"

    local flag_suffix=$(echo "${flags}" | sed 's/--//g' | sed 's/ /-/g')
    local exp_name_with_flags="${base_exp_name}-${flag_suffix}"

    echo "--- Starting: ${exp_name_with_flags} Fold ${fold} on GPU ${gpu_id} ---"

    # Training (using train_ablation.py)
    # train_ablation.py reuses baseline model's train/test split.
    # Uses log_bins/train/image-bins76-fold0~4/train.txt, test.txt
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 train_ablation.py \
        --config "${config_file}" \
        --bins ${bins_value} \
        --fold ${fold} \
        --exp-name "${exp_name_with_flags}" \
        --baseline-exp-name "${base_exp_name}" \
        --device 0 ${flags}

    # Testing (using test_ablation.py)
    local train_log_dir="logs/train/${exp_name_with_flags}-fold${fold}"
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 test_ablation.py \
        --config "${train_log_dir}/config.yaml" \
        --ckpt "${train_log_dir}/best.ckpt" \
        --exp-name "${base_exp_name}" \
        --fold ${fold} \
        --device 0 ${flags}

    echo "--- Finished: ${exp_name_with_flags} Fold ${fold} ---"
}

# --- (4) Main execution loop ---
for bins_value in "${BINS_TO_ANALYZE[@]}"; do
    # Set global variables for current model to execute
    # Changed from hybrid to image
    export CURRENT_BASE_EXP_NAME="image-bins${bins_value}"
    export CURRENT_CONFIG_FILE="configs/ajoumc_rxt50_image.yaml"
    export CURRENT_BINS_VALUE=${bins_value}
    
    # Set result filename (distinguishable as no enc, no ord are included in filename)
    SUMMARY_CSV="ablation_summary_${CURRENT_BASE_EXP_NAME}.csv"
    SUMMARY_XLSX="ablation_summary_${CURRENT_BASE_EXP_NAME}.xlsx"

    echo "======================================================================"
    echo "===== Starting Ablation Study for: ${CURRENT_BASE_EXP_NAME}"
    echo "===== Config: ${CURRENT_CONFIG_FILE}"
    echo "===== Bins: ${CURRENT_BINS_VALUE}"
    echo "======================================================================"

    # --- Create result summary file header ---
    if [ -f "${SUMMARY_CSV}" ]; then
        echo "Summary file ${SUMMARY_CSV} already exists. Removing it to start fresh."
        rm "${SUMMARY_CSV}"
    fi
    echo "Creating new summary file: ${SUMMARY_CSV}"
    echo "Experiment,1st Fold,2nd Fold,3rd Fold,4th Fold,5th Fold,Mean_MAE,Std_Dev,95%_CI_Range,P_Value_vs_Baseline,Cohens_d_vs_Baseline,Delta_MAE_vs_Baseline" > "${SUMMARY_CSV}"

    # --- Important: Full model (image + or + ve+ ibfc) will NOT be re-executed ---
    # Extract baseline MAE using previously executed results
    echo "=== Extracting baseline results (Full model will NOT be re-executed) ==="
    
    # Check baseline result file
    BASELINE_RESULT_FILE="results_${CURRENT_BASE_EXP_NAME}_combined.xlsx"
    if [ ! -f "${BASELINE_RESULT_FILE}" ]; then
        echo "Warning: Baseline result file ${BASELINE_RESULT_FILE} not found."
        echo "Attempting to combine existing baseline results..."
        
        # Attempt to merge baseline results
        baseline_files=$(find logs/test -path "*/${CURRENT_BASE_EXP_NAME}-fold*/results.xlsx" 2>/dev/null || true)
        if [ -n "${baseline_files}" ]; then
            echo "Found baseline result files. Combining..."
            python3 combine_results.py ${baseline_files} -o "${BASELINE_RESULT_FILE}"
        else
            echo "Error: No baseline result files found for ${CURRENT_BASE_EXP_NAME}."
            echo "Please ensure the full model has been trained first."
            continue
        fi
    fi

    # Extract 5-fold MAE values from baseline model
    echo "Extracting baseline MAEs from ${BASELINE_RESULT_FILE}..."
    BASE_MAES_STR=$(python3 -c "
import pandas as pd, numpy as np, sys
file_path = '${BASELINE_RESULT_FILE}'
try:
    df = pd.read_excel(file_path)
    fold_maes = df.groupby('fold').apply(lambda g: np.mean(np.abs(g['ground truth'] - g['prediction']))).values
    maes = [f'{m:.4f}' for m in fold_maes]
    if len(maes) == 5: print(','.join(maes))
    else: print(f'Error: Expected 5 folds, got {len(maes)}', file=sys.stderr); sys.exit(1)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr); sys.exit(1)
"
    )

    if [ -z "$BASE_MAES_STR" ]; then 
        echo "Error: Failed to extract baseline MAEs from ${BASELINE_RESULT_FILE}. Skipping this bins value."; 
        continue; 
    fi
    echo "Baseline MAEs: ${BASE_MAES_STR}"

    # Add baseline result summary
    python3 -c "
import sys, numpy as np
exp_name, summary_csv, maes_str = sys.argv[1], sys.argv[2], sys.argv[3]
maes = np.array([float(m) for m in maes_str.split(',')])
mean_mae, std_mae, n = np.mean(maes), np.std(maes, ddof=1), len(maes)
ci = 1.96 * (std_mae / np.sqrt(n))
ci_range = f'[{mean_mae - ci:.4f}, {mean_mae + ci:.4f}]'
with open(summary_csv, 'a') as f:
    f.write(f'{exp_name},{maes_str},{mean_mae:.4f},{std_mae:.4f},\"{ci_range}\",N/A,N/A,0.0000\n')
" "${CURRENT_BASE_EXP_NAME}" "${SUMMARY_CSV}" "${BASE_MAES_STR}"

    # --- Parallel execution (execute only ablation scenarios) ---
    echo "=== Starting ablation scenarios (no enc, no ord combinations) ==="
    job_count=0
    gpu_idx=0

    for fold in {0..4}; do
        for flags in "${scenarios[@]}"; do
            GPU_ID=${GPU_POOL[${gpu_idx}]}
            run_single_ablation ${fold} "${flags}" ${GPU_ID} &
            job_count=$((job_count + 1))
            gpu_idx=$(( (gpu_idx + 1) % MAX_CONCURRENT_JOBS ))
            if [ ${job_count} -ge ${MAX_CONCURRENT_JOBS} ]; then
                wait -n
                job_count=$((job_count - 1))
            fi
        done
    done
    wait

    # --- Merge and summarize results ---
    echo "=== Combining and Summarizing Ablation Results for ${CURRENT_BASE_EXP_NAME} ==="

    # Merge results and calculate statistics for each scenario
    for flags in "${scenarios[@]}"; do
        flag_suffix=$(echo "${flags}" | sed 's/--//g' | sed 's/ /-/g')
        exp_name="${CURRENT_BASE_EXP_NAME}-${flag_suffix}"
        output_file="results_${exp_name}_combined.xlsx"  # no enc, no ord are included in filename
        
        # Find result files
        file_list=$(find logs/test -path "*/${exp_name}-fold*/results.xlsx" 2>/dev/null || true)
        if [ -z "${file_list}" ]; then 
            echo "Warning: No result files found for ${exp_name}. Skipping."; 
            continue; 
        fi
        
        echo "Combining results for ${exp_name}..."
        echo "Output file: ${output_file}"
        python3 combine_results.py ${file_list} -o "${output_file}"

        # Calculate statistics and add to summary file
        python3 -c "
import sys, numpy as np, pandas as pd
from scipy import stats
exp_name, summary_file, base_maes_str = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    df = pd.read_excel(f'results_{exp_name}_combined.xlsx')
    maes = df.groupby('fold').apply(lambda g: np.mean(np.abs(g[\"ground truth\"] - g[\"prediction\"]))).values
    base_maes = np.array([float(m) for m in base_maes_str.split(',')])
    mean_mae, std_mae, n = np.mean(maes), np.std(maes, ddof=1), len(maes)
    ci_range = f'[{mean_mae - 1.96 * (std_mae / np.sqrt(n)):.4f}, {mean_mae + 1.96 * (std_mae / np.sqrt(n)):.4f}]'
    maes_str_out = ','.join([f'{m:.4f}' for m in maes])
    delta_mae = np.mean(maes) - np.mean(base_maes)
    t_stat, p_value = stats.ttest_rel(base_maes, maes)
    cohens_d = delta_mae / np.sqrt((np.var(base_maes, ddof=1) + np.var(maes, ddof=1)) / 2) if (np.var(base_maes, ddof=1) + np.var(maes, ddof=1)) > 0 else 0
    with open(summary_file, 'a') as f:
        f.write(f'{exp_name},{maes_str_out},{mean_mae:.4f},{std_mae:.4f},\"{ci_range}\",{p_value:.4f},{cohens_d:.4f},{delta_mae:.4f}\\n')
    print(f'Added {exp_name} to summary: MAE={mean_mae:.4f}, Delta={delta_mae:.4f}, p={p_value:.4f}')
except Exception as e:
    print(f'Error processing {exp_name}: {e}', file=sys.stderr)
" "${exp_name}" "${SUMMARY_CSV}" "${BASE_MAES_STR}"
    done

    # Convert CSV to XLSX
    echo "Converting final summary CSV to XLSX format..."
    python3 -c "
try:
    import pandas as pd
    df = pd.read_csv('${SUMMARY_CSV}')
    df.to_excel('${SUMMARY_XLSX}', index=False)
    print('Successfully converted to XLSX format.')
except Exception as e:
    print(f'Error converting to XLSX: {e}')
"
    
    echo "================================================================"
    echo "✅ Ablation study for ${CURRENT_BASE_EXP_NAME} completed."
    echo "📊 Results saved to:"
    echo "   - Summary CSV: ${SUMMARY_CSV}"
    echo "   - Summary XLSX: ${SUMMARY_XLSX}"
    echo "   - Combined results: results_${CURRENT_BASE_EXP_NAME}_combined.xlsx"
    echo "   - Ablation results: results_${CURRENT_BASE_EXP_NAME}-no-enc_combined.xlsx"
    echo "   -                   results_${CURRENT_BASE_EXP_NAME}-no-ord_combined.xlsx"
    echo "   -                   results_${CURRENT_BASE_EXP_NAME}-no-enc-no-ord_combined.xlsx"
    echo "================================================================"
done

echo ""
echo "🎉 All Image-only Model Ablation Studies Completed!"
echo ""
echo "📋 Generated Files Summary:"
for bins_value in "${BINS_TO_ANALYZE[@]}"; do
    echo "  Bins ${bins_value}:"
    echo "    - ablation_summary_image-bins${bins_value}.csv"
    echo "    - ablation_summary_image-bins${bins_value}.xlsx"
    echo "    - results_image-bins${bins_value}_combined.xlsx"
    echo "    - results_image-bins${bins_value}-no-enc_combined.xlsx"
    echo "    - results_image-bins${bins_value}-no-ord_combined.xlsx"
    echo "    - results_image-bins${bins_value}-no-enc-no-ord_combined.xlsx"
    echo ""
done
echo "✨ Use these results to analyze the effect of encoder (enc) and ordinal regression (ord) components."
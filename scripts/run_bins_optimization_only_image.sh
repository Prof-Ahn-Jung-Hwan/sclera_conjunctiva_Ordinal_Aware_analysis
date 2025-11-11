#!/bin/bash
# Filename: run_bins_optimization_only_image.sh
# Description: Runs 'bins' hyperparameter optimization using only images.
#       Performs optimization with pure image regression model without using anno data.

set -e

# --- Optimization settings ---
# 1. Create result folder
OUTPUT_DIR="bins_optimization_only_image"
mkdir -p "${OUTPUT_DIR}"

# --- Database initialization (prevent Race Condition) ---
DB_FILE="${OUTPUT_DIR}/optimization.db"
echo "Initializing Optuna database at ${DB_FILE} for image-only optimization..."
python3 -c "
import optuna
# This code creates DB if it doesn't exist, and doesn't raise an error if it exists.
# Call create_study directly to pre-create all necessary tables.
optuna.create_study(storage='sqlite:///${DB_FILE}', study_name='_initialization_dummy_', load_if_exists=True)
"
echo "Database initialized."

# Config file using only images (restored to original)
CONFIG_TO_OPTIMIZE="configs/ajoumc_rxt50_image.yaml"
NUM_TRIALS=30 # Original

# GPU configuration (using only GPU 1)
GPU_ID=1
NUM_TRIALS=30

echo "####################################################################"
echo "##### Starting Bayesian Optimization for Image-Only Model #####"
echo "##### Config: ${CONFIG_TO_OPTIMIZE}"
echo "##### Number of Trials: ${NUM_TRIALS}"
echo "##### GPU: ${GPU_ID}"
echo "####################################################################"

# Run optimization (only on GPU 1)
echo "Starting optimization for image-only model..."
echo "Config: ${CONFIG_TO_OPTIMIZE}"
echo "Output directory: ${OUTPUT_DIR}"

# Run optimization on GPU 1
echo "Starting optimization on GPU 1"
python optimize_bins.py \
    --config "${CONFIG_TO_OPTIMIZE}" \
    --n-trials ${NUM_TRIALS} \
    --gpu-id ${GPU_ID} \
    --output-dir "${OUTPUT_DIR}" \
    --tqdm-pos 0

echo "Optimization on GPU 1 completed."

echo -e "\n\n"
echo "🎉 Image-only optimization task is complete. Displaying final summary..."
echo "========================================================================"

# Output final result summary
echo -e "\n--- Summary of Best Parameters for Image-Only Model ---"
python3 -c "
import optuna
import pandas as pd
from pathlib import Path

storage_name = 'sqlite:///${OUTPUT_DIR}/optimization.db'
try:
    summaries = optuna.study.get_all_study_summaries(storage=storage_name)
    if not summaries: 
        raise ValueError('No studies found in DB.')

    results = []
    for s in sorted(summaries, key=lambda x: x.study_name):
        if s.best_trial:
            results.append({
                'Model': s.study_name.replace('bins-optimization-', ''),
                'Best MAE': s.best_trial.value,
                'Best Bins': s.best_trial.params['bins'],
                'Number of Trials': s.n_trials
            })

    if not results: 
        raise ValueError('No completed trials found.')

    df = pd.DataFrame(results)
    print('📊 Optimization Results:')
    print(df.to_string(index=False))

    best_row = df.iloc[0]
    print(f'\n🏆 Best Image-Only Model Results:')
    print(f'   - Model:     \033[1;32m{best_row[\"Model\"]}\033[0m')
    print(f'   - Best MAE:  \033[1;33m{best_row[\"Best MAE\"]:.4f}\033[0m')
    print(f'   - Best Bins: \033[1;33m{best_row[\"Best Bins\"]}\033[0m')
    print(f'   - Trials:    \033[1;36m{best_row[\"Number of Trials\"]}\033[0m')

    # Save optimization history
    study_names = optuna.study.get_all_study_names(storage=storage_name)
    if study_names:
        study_name = study_names[0]
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        # Save results to CSV
        trial_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data.append({
                    'trial_number': trial.number,
                    'bins': trial.params['bins'],
                    'mae': trial.value
                })
        
        if trial_data:
            trial_df = pd.DataFrame(trial_data)
            csv_file = f'${OUTPUT_DIR}/optimization_history.csv'
            trial_df.to_csv(csv_file, index=False)
            print(f'\n💾 Optimization history saved to: {csv_file}')
    else:
        print('No studies found for saving history.')

except Exception as e:
    print(f'Could not summarize results from DB: {e}')
    print('This might indicate that the optimization did not complete successfully.')
    
    # Try to check if summary CSV files exist instead
    import glob
    csv_files = glob.glob('${OUTPUT_DIR}/summary_*.csv')
    if csv_files:
        print(f'Found {len(csv_files)} summary CSV files:')
        for csv_file in csv_files:
            print(f'  - {csv_file}')
            try:
                df = pd.read_csv(csv_file, comment='#')
                if not df.empty:
                    print(f'    Contains {len(df)} experiments')
                    if 'Mean_MAE' in df.columns:
                        best_idx = df['Mean_MAE'].idxmin()
                        best_row = df.iloc[best_idx]
                        print(f'    Best MAE: {best_row[\"Mean_MAE\"]:.4f} (Experiment: {best_row[\"Experiment\"]})')
            except Exception as csv_e:
                print(f'    Error reading CSV: {csv_e}')
"

echo "========================================================================"
echo "✅ Image-only bins optimization completed!"
echo "📁 Results saved in: ${OUTPUT_DIR}/"
echo "🔍 Check the following files:"
echo "   - ${OUTPUT_DIR}/optimization.db (Optuna database)"
echo "   - ${OUTPUT_DIR}/optimization_history.csv (Trial history)"
echo "========================================================================"
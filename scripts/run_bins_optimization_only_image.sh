#!/bin/bash
# 파일명: run_bins_optimization_only_image.sh
# 설명: 이미지만 활용하여 'bins' 하이퍼파라미터 최적화를 실행합니다.
#       anno data를 사용하지 않고 pure image regression 모델로 최적화를 수행합니다.

set -e

# --- 최적화 설정 ---
# 1. 결과 폴더 생성
OUTPUT_DIR="bins_optimization_only_image"
mkdir -p "${OUTPUT_DIR}"

# --- 데이터베이스 초기화 (Race Condition 방지) ---
DB_FILE="${OUTPUT_DIR}/optimization.db"
echo "Initializing Optuna database at ${DB_FILE} for image-only optimization..."
python3 -c "
import optuna
# 이 코드는 DB가 없으면 생성하고, 있어도 에러를 발생시키지 않습니다.
# create_study를 바로 호출하여 필요한 모든 테이블을 미리 생성합니다.
optuna.create_study(storage='sqlite:///${DB_FILE}', study_name='_initialization_dummy_', load_if_exists=True)
"
echo "Database initialized."

# 이미지만 사용하는 config 파일 (원본으로 복원)
CONFIG_TO_OPTIMIZE="configs/ajoumc_rxt50_image.yaml"
NUM_TRIALS=30 # Original

# GPU 설정 (GPU 1만 사용)
GPU_ID=1
NUM_TRIALS=30

echo "####################################################################"
echo "##### Starting Bayesian Optimization for Image-Only Model #####"
echo "##### Config: ${CONFIG_TO_OPTIMIZE}"
echo "##### Number of Trials: ${NUM_TRIALS}"
echo "##### GPU: ${GPU_ID}"
echo "####################################################################"

# 최적화 실행 (GPU 1에서만 실행)
echo "Starting optimization for image-only model..."
echo "Config: ${CONFIG_TO_OPTIMIZE}"
echo "Output directory: ${OUTPUT_DIR}"

# GPU 1에서 최적화 실행
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

# 최종 결과 요약 출력
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

    # 최적화 히스토리 저장
    study_names = optuna.study.get_all_study_names(storage=storage_name)
    if study_names:
        study_name = study_names[0]
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        # 결과를 CSV로 저장
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
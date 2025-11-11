#!/bin/bash

# 이 스크립트는 가장 순수한 형태의 '이미지 단독 회귀' 모델을 학습하고 평가합니다.
# - Ablation study의 베이스라인 모델과 동일한 train/test 데이터 분할을 사용합니다.
# 스크립트 실행 중 오류가 발생하면 즉시 중단합니다.
set -e

# 스크립트가 위치한 디렉토리를 기준으로 경로를 설정합니다.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}" # 스크립트 디렉토리로 작업 위치 변경

# --- (1) Ablation Study의 베이스라인이 될 모델 목록 ---
# image-bins76 모델의 train/test split을 사용하여 image-only regression을 수행합니다.
BASELINE_MODELS=(
    "image-bins76"
)

# GPU 설정 (2대 사용)
GPU_POOL=(0 1)
MAX_CONCURRENT_JOBS=${#GPU_POOL[@]}

# --- (3) 단일 Fold 실행 함수 ---
run_single_regression_fold() {
    local fold=$1
    local gpu_id=$2
    local exp_name=$3
    local baseline_exp_name=$4

    echo "--- Starting: ${EXP_NAME} Fold ${fold} on GPU ${gpu_id} ---"

    local baseline_log_dir="logs/train/${baseline_exp_name}-fold${fold}"
    local train_file="${baseline_log_dir}/train.txt"
    local test_file="${baseline_log_dir}/test.txt"

    # 학습 (train_regression.py 사용)
    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n torch271 --no-capture-output python3 train_regression.py \
        --config configs/ajoumc_rxt50_image_regression.yaml \
        --fold ${fold} \
        --exp-name "${exp_name}" \
        --train-file "${train_file}" \
        --test-file "${test_file}" \
        --device 0

    # 테스트 (test_regression.py 사용)
    local train_log_dir="logs/train/${exp_name}-fold${fold}"
    CUDA_VISIBLE_DEVICES=${gpu_id} conda run -n torch271 --no-capture-output python3 test_regression.py \
        --config configs/ajoumc_rxt50_image_regression.yaml \
        --exp-name "${exp_name}" \
        --fold ${fold} \
        --ckpt "${train_log_dir}/best.ckpt" \
        --test-file "${test_file}"

    echo "--- Finished: ${EXP_NAME} Fold ${fold} ---"
}

# --- (4) 메인 실행 루프 ---
for baseline_exp_name in "${BASELINE_MODELS[@]}"; do
    # image-bins76에서 파생된 고유한 실험 이름 생성
    # 예: image-bins76 -> image-regression-from-bins76
    baseline_suffix=$(echo "${baseline_exp_name}" | sed 's/image-//')
    EXP_NAME="image-regression-from-${baseline_suffix}"

    echo "======================================================================"
    echo "===== Starting Pure Image-Only Regression Study"
    echo "===== Using data split from: ${baseline_exp_name}"
    echo "===== Saving results as: ${EXP_NAME}"
    echo "======================================================================"
    
    # --- 5-Fold 병렬 실행 ---
    job_count=0
    gpu_idx=0

    for fold in {0..4}; do
        GPU_ID=${GPU_POOL[${gpu_idx}]}
        # 생성된 고유 실험 이름(EXP_NAME)과 베이스라인 이름(baseline_exp_name)을 함께 전달
        run_single_regression_fold ${fold} ${GPU_ID} "${EXP_NAME}" "${baseline_exp_name}" &
        
        job_count=$((job_count + 1))
        gpu_idx=$(( (gpu_idx + 1) % MAX_CONCURRENT_JOBS ))
        if [ ${job_count} -ge ${MAX_CONCURRENT_JOBS} ]; then
            wait -n
            job_count=$((job_count - 1))
        fi
    done
    wait # 모든 fold 작업이 끝날 때까지 대기

    # --- 결과 병합 ---
    echo "--- Combining Results for ${EXP_NAME} ---"
    COMBINED_FILENAME="results_${EXP_NAME}_combined.xlsx"
    
    # find 명령어로 모든 fold의 결과 파일을 찾습니다.
    file_list=$(find logs/test -path "*/${EXP_NAME}-fold*/results.xlsx")
    if [ -z "${file_list}" ]; then
        echo "Warning: No result files found for ${EXP_NAME}. Skipping combination."
        continue # 다음 베이스라인 모델로 넘어감
    fi
    
    conda run -n torch271 --no-capture-output python3 combine_results.py ${file_list} -o "${COMBINED_FILENAME}"
    echo "Combined results saved to: ${COMBINED_FILENAME}"
done

echo -e "\n\n🎉 All regression-only studies are complete."

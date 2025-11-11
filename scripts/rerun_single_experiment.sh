#!/bin/bash
# 파일명: rerun_single_experiment.sh
# 설명: 단일 실험 구성에 대해 5-fold CV를 수동으로 재실행합니다.
#       결과는 지정된 요약 파일에 추가(또는 덮어쓰기)됩니다.
# 사용법: ./rerun_single_experiment.sh <config_file> <bins> <summary_csv_file> [extra_flags] [gpu_id]
# 예시:   ./rerun_single_experiment.sh configs/ajoumc_rxt50_hybrid_n.yaml 20 ablation_summary_with_folds.csv "" 0

set -e

# --- 입력값 확인 ---
CONFIG_FILE=$1
BINS_VALUE=$2
SUMMARY_CSV_FILE=$3
EXTRA_FLAGS=$4 # Optional, use "" if not needed
GPU_ID=${5:-0} # Optional, defaults to 0 if not provided

if [ -z "$CONFIG_FILE" ] || [ -z "$BINS_VALUE" ] || [ -z "$SUMMARY_CSV_FILE" ]; then
    echo "사용법: $0 <config_file> <bins_value> <summary_csv_file> [extra_flags]"
    echo "예시: $0 configs/ajoumc_rxt50_hybrid_n.yaml 20 ablation_summary_with_folds.csv"
    exit 1
fi

# --- 실험 이름 생성 (run_ablation_studies.sh와 동일한 로직) ---
BASE_NAME=$(basename "${CONFIG_FILE}" .yaml | sed 's/ajoumc_rxt50_//')
EXP_NAME="${BASE_NAME}-bins${BINS_VALUE}"
if [[ -n "${EXTRA_FLAGS}" ]]; then
    flag_suffix=$(echo "${EXTRA_FLAGS}" | sed 's/--//g' | sed 's/ /-/g')
    EXP_NAME="${EXP_NAME}-${flag_suffix}"
fi

echo "############################################################"
echo "##### Rerunning single experiment: ${EXP_NAME}"
echo "##### Config: ${CONFIG_FILE}, Bins: ${BINS_VALUE}, Flags: '${EXTRA_FLAGS}', GPU: ${GPU_ID}"
echo "##### Results will be updated in: ${SUMMARY_CSV_FILE}"
echo "############################################################"

# --- 5-Fold CV 실행 ---
BASE_TRAIN_LOG_DIR="log_bins/train"
BASE_TEST_LOG_DIR="log_bins/test"
declare -a MAE_VALUES
declare -a EPOCH_VALUES
declare -a RESULT_FILES

for FOLD in {0..4}; do
    TRAIN_LOG_DIR="${BASE_TRAIN_LOG_DIR}/${EXP_NAME}-fold${FOLD}"
    TEST_LOG_DIR="${BASE_TEST_LOG_DIR}/${EXP_NAME}-fold${FOLD}"

    # --- (개선) 이어서 실행하기 기능 ---
    # 해당 fold의 최종 결과 파일(best.ckpt)이 이미 존재하면, 학습을 건너뛰고 테스트만 재실행하여 결과를 가져옵니다.
    if [ -f "${TRAIN_LOG_DIR}/best.ckpt" ]; then
        echo "Fold ${FOLD} for ${EXP_NAME} seems to be already trained. Re-running test to get MAE."
        TEST_OUTPUT=$(CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_bins.py --config "${TRAIN_LOG_DIR}/config.yaml" --ckpt "${TRAIN_LOG_DIR}/best.ckpt" --exp-name "${EXP_NAME}" --fold ${FOLD} --device 0)
        MAE=$(echo "${TEST_OUTPUT}" | grep 'Overall Test MAE:' | awk '{print $4}')
        BEST_EPOCH=$(echo "${TEST_OUTPUT}" | grep 'Best MAE was achieved at epoch:' | awk '{print $6}')
        MAE_VALUES+=(${MAE})
        EPOCH_VALUES+=(${BEST_EPOCH:-0})
        RESULT_FILES+=("${TEST_LOG_DIR}/results.xlsx")
        echo "Fold ${FOLD} MAE: ${MAE}, Best Epoch: ${BEST_EPOCH}"
        continue # 다음 fold로 넘어갑니다.
    fi

    echo "=================================================="
    echo ">>>>> Starting Fold ${FOLD} for ${EXP_NAME} on GPU ${GPU_ID}"
    echo "=================================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train_bins.py --config ${CONFIG_FILE} --bins ${BINS_VALUE} --fold ${FOLD} --exp-name "${EXP_NAME}" --device 0 ${EXTRA_FLAGS}

    TRAIN_LOG_DIR="${BASE_TRAIN_LOG_DIR}/${EXP_NAME}-fold${FOLD}"
    TEST_OUTPUT=$(CUDA_VISIBLE_DEVICES=${GPU_ID} python src/test_bins.py --config "${TRAIN_LOG_DIR}/config.yaml" --ckpt "${TRAIN_LOG_DIR}/best.ckpt" --exp-name "${EXP_NAME}" --fold ${FOLD} --device 0)
    MAE=$(echo "${TEST_OUTPUT}" | grep 'Overall Test MAE:' | awk '{print $4}')
    BEST_EPOCH=$(echo "${TEST_OUTPUT}" | grep 'Best MAE was achieved at epoch:' | awk '{print $6}')
    MAE_VALUES+=(${MAE})
    EPOCH_VALUES+=(${BEST_EPOCH:-0}) # Default to 0 if not found
    echo "Fold ${FOLD} MAE: ${MAE}, Best Epoch: ${BEST_EPOCH}"
    RESULT_FILES+=("${TEST_LOG_DIR}/results.xlsx")
done

# --- 결과 요약 및 저장 ---
echo "--- Summarizing results for ${EXP_NAME} ---"

# --- 파일 잠금을 사용하여 안전하게 CSV 파일 수정 ---
# flock을 사용하여 동시에 여러 프로세스가 이 스크립트를 실행해도 파일이 깨지지 않도록 보호합니다.
(
    flock 200
    
    # CSV 파일에 헤더가 없으면 추가
    if [ ! -f "${SUMMARY_CSV_FILE}" ] || [ ! -s "${SUMMARY_CSV_FILE}" ] || ! head -1 "${SUMMARY_CSV_FILE}" | grep -q "^Experiment,"; then
        echo "Experiment,1st Fold,2nd Fold,3rd Fold,4th Fold,5th Fold,Mean_MAE,Std_Dev,95%_CI_Range" > "${SUMMARY_CSV_FILE}"
    fi
    
    # 요약 파일에서 기존 항목을 삭제하여 결과를 덮어쓸 준비를 합니다.
    sed -i "/^${EXP_NAME},/d" "${SUMMARY_CSV_FILE}"

    # Python으로 통계 계산 및 파일에 추가
    python3 -c '
import sys, numpy as np
exp_name = sys.argv[1]
summary_csv_file = sys.argv[2]
maes = np.array([float(m) for m in sys.argv[3:]])
mean_mae = np.mean(maes)
std_mae = np.std(maes, ddof=1)
n = len(maes)
confidence_interval = 1.96 * (std_mae / np.sqrt(n))
ci_range = f"[{mean_mae - confidence_interval:.4f}, {mean_mae + confidence_interval:.4f}]"
maes_str = ",".join([f"{m:.4f}" for m in maes])

with open(summary_csv_file, "a") as f:
    # 헤더: Experiment,1st Fold,2nd Fold,3rd Fold,4th Fold,5th Fold,Mean_MAE,Std_Dev,95%_CI_Range
    f.write(f"{exp_name},{maes_str},{mean_mae:.4f},{std_mae:.4f},\"{ci_range}\"\n")
' "${EXP_NAME}" "${SUMMARY_CSV_FILE}" "${MAE_VALUES[@]}"

) 200>"${SUMMARY_CSV_FILE}.lock" # 잠금 파일 지정

# 5-fold 결과 파일 병합
COMBINED_FILENAME="results_${EXP_NAME}_combined.xlsx"
python combine_results.py --output ${COMBINED_FILENAME} "${RESULT_FILES[@]}"

# 최종 요약 라인 출력 (optimize_bins.py가 이 라인을 캡처하여 표시)
# MAE_VALUES와 EPOCH_VALUES를 임시 파일에 저장하여 안전하게 Python에 전달
TEMP_MAE_FILE=$(mktemp)
TEMP_EPOCH_FILE=$(mktemp)
printf '%s\n' "${MAE_VALUES[@]}" > "${TEMP_MAE_FILE}"
printf '%s\n' "${EPOCH_VALUES[@]}" > "${TEMP_EPOCH_FILE}"

MEAN_MAE=$(python3 -c "
import numpy as np
with open('${TEMP_MAE_FILE}', 'r') as f:
    maes = [float(line.strip()) for line in f if line.strip()]
print(f'{np.mean(maes):.4f}')
")

AVG_EPOCH=$(python3 -c "
import numpy as np
with open('${TEMP_EPOCH_FILE}', 'r') as f:
    epochs = [float(line.strip()) for line in f if line.strip()]
print(f'{np.mean(epochs):.0f}')
")

# 임시 파일 정리
rm -f "${TEMP_MAE_FILE}" "${TEMP_EPOCH_FILE}"

echo "TRIAL_SUMMARY: Bins: ${BINS_VALUE}, Mean MAE: ${MEAN_MAE}, Avg Best Epoch: ${AVG_EPOCH}"

echo -e "\n##### Rerun for ${EXP_NAME} is complete!"

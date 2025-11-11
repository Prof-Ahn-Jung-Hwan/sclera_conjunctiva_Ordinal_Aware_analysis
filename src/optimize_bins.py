# optimize_bins.py
import argparse
import subprocess
import pandas as pd
import optuna
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Optuna의 기본 로그 출력이 tqdm 진행률 표시줄을 방해하지 않도록 비활성화합니다.
optuna.logging.set_verbosity(optuna.logging.WARNING)

def bootstrap_ci(data, n_bootstraps=1000, ci=0.95):
    """Calculates the confidence interval using bootstrapping."""
    bootstrapped_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    sorted_means = np.array(bootstrapped_means)
    sorted_means.sort()
    lower_bound = sorted_means[int((1.0-ci)/2.0 * len(sorted_means))]
    upper_bound = sorted_means[int((1.0-(1.0-ci)/2.0) * len(sorted_means))]
    
    return lower_bound, upper_bound

def run_experiment_and_get_mae(config_file, bins_value, summary_csv, gpu_id):
    """
    rerun_single_experiment.sh를 호출하여 5-fold CV를 실행하고,
    결과 CSV 파일에서 Mean_MAE 값을 읽어 반환합니다.
    """
    base_name = Path(config_file).stem.replace('ajoumc_rxt50_', '')
    exp_name = f"{base_name}-bins{bins_value}"

    # 이미 실행된 결과가 있는지 확인
    if summary_csv.exists():
        try:
            # CSV 파일을 안전하게 읽기 위해 여러 방법 시도
            try:
                df = pd.read_csv(summary_csv, comment='#')
            except pd.errors.ParserError:
                # 파싱 에러가 발생하면 주석 줄을 제외하고 다시 시도
                with open(summary_csv, 'r') as f:
                    lines = [line for line in f if not line.strip().startswith('#')]
                
                from io import StringIO
                df = pd.read_csv(StringIO(''.join(lines)))
            
            if exp_name in df['Experiment'].values:
                mae = df[df['Experiment'] == exp_name]['Mean_MAE'].iloc[0]
                # print(f"Found existing result for {exp_name}. MAE: {mae:.4f}") # tqdm과 겹치므로 주석 처리
                return mae
        except Exception as e:
            print(f"Warning: Could not read existing results from {summary_csv}: {e}")
            # 파일이 손상되었을 수 있으므로 다시 생성

    # 스크립트 실행
    # print(f"Running new experiment for {exp_name} on GPU {gpu_id}...") # tqdm이 진행률을 보여주므로 주석 처리
    # 지정된 GPU로 실행
    command = [
        "./rerun_single_experiment.sh",
        config_file,
        str(bins_value),
        str(summary_csv),
        "", # extra_flags 없음
        str(gpu_id) # GPU ID
    ]
    
    try:
        # capture_output=True로 되돌려 자식 프로세스의 출력을 숨기고 메인 진행 바를 깔끔하게 유지합니다.
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding='utf-8'
        )
        # 자식 스크립트(rerun_single_experiment.sh)가 출력하는 최종 요약 라인을 찾아 화면에 표시합니다.
        summary_line = [line for line in result.stdout.split('\n') if 'TRIAL_SUMMARY:' in line]
        if summary_line:
            print(summary_line[0])

    except subprocess.CalledProcessError as e:
        # 오류 발생 시에만 상세 정보를 출력합니다.
        print(f"\n\nError running experiment for {exp_name} (bins={bins_value}) on GPU {gpu_id}")
        print("--- STDERR ---")
        print(e.stderr)
        print("--- STDOUT ---")
        print(e.stdout)
        return float('inf')

    # 실행 후 결과 파일에서 MAE 읽기
    try:
        try:
            df = pd.read_csv(summary_csv, comment='#')
        except pd.errors.ParserError:
            # 파싱 에러가 발생하면 주석 줄을 제외하고 다시 시도
            with open(summary_csv, 'r') as f:
                lines = [line for line in f if not line.strip().startswith('#')]
            
            from io import StringIO
            df = pd.read_csv(StringIO(''.join(lines)))
        
        mae = df[df['Experiment'] == exp_name]['Mean_MAE'].iloc[0]
        return mae
    except Exception as e:
        print(f"Error reading results from {summary_csv}: {e}")
        return float('inf')

def objective(trial, config_file, summary_csv, gpu_id):
    """Optuna가 최적화할 목적 함수"""
    # 탐색할 bins 값의 범위. 예: 10에서 100 사이의 정수
    bins = trial.suggest_int("bins", 10, 100)
    
    # 실험을 실행하고 MAE를 반환 (MAE는 낮을수록 좋으므로 최소화 대상)
    mae = run_experiment_and_get_mae(config_file, bins, summary_csv, gpu_id)
    return mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize 'bins' hyperparameter using Bayesian Optimization.")
    parser.add_argument("--config", default="configs/ajoumc_rxt50_hybrid_all.yaml", help="Config file for the experiment.")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of optimization trials to run.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for the experiments.")
    parser.add_argument("--tqdm-pos", type=int, default=0, help="Position for the tqdm progress bar in the terminal.")
    parser.add_argument("--output-dir", default="bins_optimization", help="Output directory for optimization results.")
    args = parser.parse_args()
    
    # 1. 결과를 지정된 output directory에 저장
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    config_name = Path(args.config).stem
    summary_file = output_dir / f"summary_{config_name}.csv"
    
    # --- Optuna Study 설정 (SQLite DB 사용) ---
    # DB를 사용하여 탐색 기록을 저장하고, 중단 시 이어할 수 있도록 합니다.
    study_name = f"bins-optimization-{config_name}"
    storage_name = f"sqlite:///{output_dir}/optimization.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, # 동일한 이름의 study가 있으면 불러옵니다.
        direction="minimize"
    )
    
    # 3. 수동으로 tqdm 프로그레스 바를 생성하여 구버전 Optuna와 호환되도록 합니다.
    tqdm_desc = f"Model: {config_name.replace('ajoumc_rxt50_', '')}"
    
    # 이미 완료된 trial 수를 계산하여 이어서 실행합니다.
    n_completed = len(study.trials)
    n_to_run = args.n_trials - n_completed

    if n_to_run > 0:
        with tqdm(total=args.n_trials, initial=n_completed, desc=tqdm_desc, position=args.tqdm_pos, leave=False) as pbar:
            # objective 함수가 실행될 때마다 프로그레스 바를 업데이트하는 래퍼 함수
            def objective_with_pbar(trial):
                mae = objective(trial, args.config, summary_file, args.gpu_id)
                pbar.update(1)
                pbar.set_postfix(mae=f'{mae:.4f}')
                return mae
            
            study.optimize(objective_with_pbar, n_trials=n_to_run, n_jobs=1)

    print("\n==================================================")
    print("Optimization Finished!")
    print(f"Number of trials: {len(study.trials)}")
    
    if n_to_run > 0:
        print(f"Best trial for {args.config}:")
        best_bins = study.best_params['bins']
        best_mae = study.best_value
        print(f"  MAE: {best_mae:.4f}")
        print(f"  Best Bins: {best_bins}")
        print(f"Summary saved in: {summary_file}")
        print("==================================================")

        # 3. CSV 파일 마지막에 최종 선택된 bins 값 추가
        with open(summary_file, 'a') as f:
            f.write("\n# --- Optimization Result ---\n")
            f.write(f"# Best Bins,{best_bins}\n")
            f.write(f"# Best MAE,{best_mae:.4f}\n")
        print(f"Appended best parameters to {summary_file}")
    else:
        print("No new trials run. Loading existing data for visualization.")
        print(f"Summary file: {summary_file}")

    # 4. Hyperparameter Optimization 산점도 그래프 생성
    # BUG FIX: Use the summary file generated by this optimization run, not a hardcoded old file.
    try:
        try:
            df = pd.read_csv(summary_file, comment='#') # Use comment='#' to ignore summary lines at the end of the file.
        except pd.errors.ParserError:
            # 파싱 에러가 발생하면 주석 줄을 제외하고 다시 시도
            with open(summary_file, 'r') as f:
                lines = [line for line in f if not line.strip().startswith('#')]
            
            from io import StringIO
            df = pd.read_csv(StringIO(''.join(lines)))
        
        if not df.empty:
            df['bins'] = df['Experiment'].str.extract(r'bins(\d+)').astype(int)
            
            # MAE_95%CI 계산 (fold별 MAE로 CI 계산)
            fold_names = ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold']
            df['MAE_CI_lower'] = df.apply(lambda row: bootstrap_ci([row[name] for name in fold_names])[0], axis=1)
            df['MAE_CI_upper'] = df.apply(lambda row: bootstrap_ci([row[name] for name in fold_names])[1], axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(df['bins'], df['Mean_MAE'], yerr=df['Std_Dev'], fmt='o', capsize=5, label='Mean MAE ± Std Dev')
            plt.fill_between(df['bins'], df['MAE_CI_lower'], df['MAE_CI_upper'], alpha=0.3, label='95% CI')
            plt.xlabel('Bins')
            plt.ylabel('Mean MAE')
            plt.title('Hyperparameter Optimization: Bins vs Mean MAE')
            plt.legend()
            plt.grid(True)
            
            output_dir = Path("report_250924")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / 'spp_table1_full_hyperparameter_search_baysian_opti.png')
            plt.close()
            
            # Excel로 저장
            df.to_excel(output_dir / 'spp_table1_full_hyperparameter_search_baysian_opti.xlsx', index=False)
            print(f"Graph and Excel saved to {output_dir}")
        else:
            print("No data available for visualization")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("Summary file might be empty or corrupted")

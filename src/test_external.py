"""
Usage: External validation evaluation script
- Evaluates trained models on external datasets (India, Italy, Ghana)
- Supports both zero-shot and few-shot evaluation
- Compares performance across different external populations

Examples:
    # Joint dataset (India + Italy) evaluation
    python test_external.py --config configs/ajoumc_rxt50_image.yaml --ckpt logs/train/hybrid_demo-bins68-fold0/best.ckpt --external_type joint --fold 0

    # Ghana dataset evaluation
    python test_external.py --config configs/ajoumc_rxt50_image.yaml --ckpt logs/train/hybrid_demo-bins68-external-ghana-fold0/best.ckpt --external_type ghana --fold 0

    # Custom external CSV file
    python test_external.py --config configs/ajoumc_rxt50_image.yaml --ckpt path/to/checkpoint.ckpt --external_csv path/to/external_data.csv
"""

import argparse
import math
import os
# Add project root to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import pandas as pd
import matplotlib.pylab as plt
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from dataset import load_dataset
from dataset.ajoumc_anemia import AjouMC_AnemiaDataset
from misc import MAE_error, setup_seed
from model import HbNet
from ordinal_regression import create_ordinal_lookup, decode_ordinal, encode_ordinal


def main(args):
    # 5-fold test setup
    folds = list(range(5))  # 0, 1, 2, 3, 4
    
    # Shot type 결정 (few-shot 또는 zero-shot)
    shot_type = "few" if args.is_few_shot else "zero"
    print(f"Running {shot_type}-shot evaluation for {args.external_type} dataset")
    
    for fold in folds:
        print(f"\n{'='*80}")
        print(f"Starting {shot_type}-shot test for {args.external_type.upper()} dataset - Fold {fold}")
        print(f"{'='*80}")
        
        # Fold별 설정 업데이트
        args.fold = fold
        
        # 테스트 데이터 경로 설정 (동일)
        # if args.external_type == 'ghana':
        #     test_csv = 'external_validation_proceed/test_ext_ghana.csv'
        # else:  # eyedye
        test_csv = 'external_validation_proceed/test_ext_eyedye.csv'
        
        # 체크포인트 경로 설정 (few-shot vs zero-shot)
        if args.is_few_shot:
            model_type = f'image-bins{args.ordinal_bins}'
            ckpt_path = f'logs/train/{model_type}-ext-{args.external_type}-fold{fold}/best.ckpt'
        else:
            # Zero-shot: 원본 모델 사용
            model_type = f'image-bins{args.ordinal_bins}'
            base_dir = 'logs/train'
            ckpt_path = f'{base_dir}/{model_type}-fold{fold}/best.ckpt'
        
        # 테스트 결과 저장 경로 설정
        model_type_for_log = f'image-bins{args.ordinal_bins}'
        log_dir = f'logs/test/{model_type_for_log}-ext-{args.external_type}-{shot_type}-fold{fold}'
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"{shot_type.capitalize()}-shot test setup for fold {fold}:")
        print(f"  External type: {args.external_type}")
        print(f"  Test data: {test_csv}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Output directory: {log_dir}")

        # 체크포인트 파일 존재 확인
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint file not found: {ckpt_path}")
            print(f"Skipping fold {fold}")
            continue
            
        # 테스트 데이터 파일 존재 확인
        if not os.path.exists(test_csv):
            print(f"ERROR: Test data file not found: {test_csv}")
            print(f"Skipping fold {fold}")
            continue

        # 한글 폰트 설정 (플롯에서 한글 깨짐 방지)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
        except:
            print("Warning: Matplotlib font configuration failed. Korean characters in the plot may not display correctly.")

        args.lookup = create_ordinal_lookup(args)
        print(f"Ordinal lookup table: {args.lookup}")

        # External data 로드 (CSV)
        df = pd.read_csv(test_csv)
        
        test_data = []
        print(f"Processing test data: {test_csv}")
        print(f"DataFrame shape: {df.shape}")
        
        for _, row in df.iterrows():
            # CSV 파일의 full_path 사용 (이미 올바른 경로로 생성됨)
            img_path = row['full_path']
            
            # 이미지 파일 존재 확인
            if not os.path.exists(img_path):
                print(f"WARNING: Image file not found: {img_path}")
                continue
                
            Hb = float(row['Hb'])
            
            test_data.append((Path(img_path), {'Hb': Hb, 'Hybrid': []}))

        print(f"Loaded {len(test_data)} test samples")
        
        if len(test_data) == 0:
            print(f"ERROR: No test data loaded for fold {fold}!")
            continue

        test_dataset = AjouMC_AnemiaDataset(args, test_data, mode='test', w_filename=True)

        # Fold별로 다른 시드 사용
        args.seed += fold
        setup_seed(args.seed)

        model = HbNet(args).to(args.device)

        # 모델 로드
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print("✓ Checkpoint loaded successfully")

        model.eval()
        
        # 엑셀 저장을 위해 모든 결과를 담을 리스트
        all_results = []
        # 이미지 시각화를 위해 n_samples 만큼의 결과를 담을 리스트
        plot_results = []

        # 1. 전체 테스트 데이터셋에 대해 루프를 실행합니다.
        pbar = tqdm(range(len(test_dataset)), desc=f"Evaluating {shot_type}-shot fold {fold}")
        for index in pbar:
            with torch.no_grad():
                img, target, anno, f_name = test_dataset.__getitem__(index)
                img, target, anno = (
                    img.to(args.device).unsqueeze(0),
                    target.to(args.device).unsqueeze(0),
                    anno.to(args.device).unsqueeze(0),
                )
                model_out = model(img)
                prediction = (
                    decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]
                )

                # 2. 모든 결과를 all_results 리스트에 추가합니다. (for Excel)
                all_results.append(
                    {
                        "w_filename": f_name,
                        "ground truth": target.item(),
                        "prediction": prediction.item(),
                    }
                )

                # 3. '--n_samples' 개수만큼만 plot_results 리스트에 추가합니다. (for Plotting)
                if index < args.n_samples:
                    plot_results.append(
                        [
                            img.cpu()[0].permute(1, 2, 0),
                            target,
                            prediction,
                            f_name,
                        ]
                    )

        # 4. 'all_results'를 사용하여 전체 결과에 대한 엑셀 파일을 생성하고 MAE를 계산합니다.
        df_results = pd.DataFrame(all_results)
        mae = MAE_error(torch.tensor(df_results["prediction"].values), torch.tensor(df_results["ground truth"].values)).mean().item()
        print(f"Fold {fold} {shot_type.capitalize()}-shot Test MAE: {mae:.4f}")

        excel_path = os.path.join(log_dir, "results.xlsx")
        df_results.to_excel(excel_path, index=False)
        print(f"Test results saved to {excel_path}")

        # 5. 'plot_results'를 사용하여 '--n_samples' 개수만큼의 이미지만 그립니다.
        cols = 5
        rows = math.ceil(len(plot_results) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = [axes]  # 1행인 경우 axes를 리스트로 만듦
        axes = axes.flatten()

        for i, (img, s1, s2, f_name) in enumerate(plot_results):
            ax = axes[i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{f_name}\nGT:{s1.item():.1f} / Pred:{s2.item():.1f}", fontsize=8)
            ax.imshow(img.mul(torch.tensor([0.229, 0.224, 0.225])).add(torch.tensor([0.485, 0.456, 0.406])))

        # 남은(비어 있는) subplot들은 감춥니다.
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(log_dir, "test.png")
        plt.savefig(plot_path)
        plt.close()  # 메모리 절약을 위해 figure 닫기
        print(f"Test visualization saved to {plot_path}")
        
        print(f"Fold {fold} {shot_type}-shot testing completed! MAE: {mae:.4f}")
        print(f"Results saved to: {log_dir}")
        print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print(f"ALL FOLDS {shot_type.upper()}-SHOT TESTING COMPLETED FOR {args.external_type.upper()} DATASET")
    print(f"{'='*80}")


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "-c",
        "--config",
        default="configs/ajoumc_rxt50_image.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument("--bins", type=int, help="Number of bins for ordinal regression")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for cross-validation")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file to evaluate")
    parser.add_argument("--is_few_shot", action="store_true", help="Use few-shot trained model")
    parser.add_argument("--n_samples", default=100, type=int, help="Number of samples to visualize")
    parser.add_argument("--exp-name", type=str, help="A name for the experiment, used for log directory.")
    parser.add_argument("--no_enc", action="store_true", help="Disable encoder usage")
    parser.add_argument("--no_ord", action="store_true", help="Disable ordinal regression")
    # --- Arguments for External Validation ---
    parser.add_argument("--external_type", type=str, default="eyedye", choices=["eyedye"], help="External dataset type (only eyedye supported)")  # ghana 제거됨
    parser.add_argument("--external_csv", type=str, help="Path to external validation CSV/Excel file")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            # ckpt 경로가 config 파일과 커맨드 라인에 모두 있을 경우, 커맨드 라인 인자를 우선합니다.
            # 이를 위해 cfg에서 ckpt를 미리 제거합니다.
            if 'ckpt' in cfg:
                del cfg['ckpt']
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    """
    Overwrite arguments with command line arguments.
    """
    if args.bins is not None:
        args.ordinal_bins = args.bins

    if args.no_enc:
        args.use_encoder = False
    if args.no_ord:
        args.use_ordinal_regression = False

    shot_type = "few" if args.is_few_shot else "zero"
    print(f"Testing {args.external_type} dataset with 5-fold {shot_type}-shot models")

    main(args)

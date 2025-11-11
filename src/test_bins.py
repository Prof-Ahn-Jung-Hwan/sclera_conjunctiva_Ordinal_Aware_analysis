"""
Ajou University Medical Center & ETRI Anemia Analysis AI Project

This script is a modified version of test.py, specifically for
evaluating models from the 'bins' hyperparameter search. It loads
the best checkpoint and reports the final MAE and the epoch at
which the best performance was achieved.
"""

import argparse
import math
import os
# Add project root to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import pandas as pd
import matplotlib.pylab as plt
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from misc import MAE_error
from model import HbNet
from ordinal_regression import create_ordinal_lookup, decode_ordinal


@torch.no_grad()
def main(args):
    # 한글 폰트 설정 (플롯에서 한글 깨짐 방지)
    try:
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("Warning: NanumGothic font not found. Korean characters in the plot may not display correctly.")
        print("Please install a Korean font (e.g., 'sudo apt-get install fonts-nanum*')")

    # Create the ordinal lookup table, which is needed by the dataset loader and decoder
    args.lookup = create_ordinal_lookup(args)
    print(f"Ordinal lookup table: {args.lookup}")


    # Load dataset for testing
    _, test_folds, _, _ = load_dataset(args, cv_fold=5, seed=args.seed, w_filename=True)
    test_dataloader = DataLoader(
        test_folds[args.fold],
        batch_size=args.test_batch_size,
        num_workers=args.test_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Load model
    model = HbNet(args).to(args.device)
    
    # Load checkpoint and extract best epoch
    checkpoint = torch.load(args.ckpt, map_location=f"cuda:{args.device}")
    model.load_state_dict(checkpoint["state_dict"])
    best_epoch = checkpoint.get("epoch", "N/A") # .get() for safety
    model.eval()

    # Run evaluation
    pbar = tqdm(test_dataloader)
    all_results = []
    plot_results = []

    for img, target, anno, fnames in pbar:
        img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
        model_out = model(img)

        preds = decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]
        mae_error = MAE_error(preds, target)

        pbar.set_description(f"[TEST] Batch MAE: {mae_error.mean().item():.4f}")

        # 배치 내의 각 샘플에 대해 결과 수집
        for j in range(len(fnames)):
            # 1. 모든 결과를 all_results 리스트에 추가 (for Excel)
            all_results.append({
                "w_filename": fnames[j],
                "ground truth": target[j].item(),
                "prediction": preds[j].item(),
            })

            # 2. '--n_samples' 개수만큼만 plot_results 리스트에 추가 (for Plotting)
            if len(plot_results) < args.n_samples:
                plot_results.append([
                    img[j].cpu().permute(1, 2, 0),
                    target[j],
                    preds[j],
                    fnames[j],
                ])

    # --- 결과 출력 및 저장 ---
    df = pd.DataFrame(all_results)
    overall_mae = MAE_error(torch.tensor(df["prediction"].values), torch.tensor(df["ground truth"].values)).mean().item()
    print(f"\nOverall Test MAE: {overall_mae:.4f}")
    print(f"Best MAE was achieved at epoch: {best_epoch}") # 최고 성능 에포크 출력

    # Save detailed results to Excel
    log_dir = os.path.join("log_bins", "test", f"{args.exp_name}-fold{args.fold}")
    os.makedirs(log_dir, exist_ok=True)
    excel_path = os.path.join(log_dir, "results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Test results saved to {excel_path}")

    # --- 시각화 결과 저장 ---
    cols = 5
    rows = math.ceil(len(plot_results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (img_tensor, gt, pred, f_name) in enumerate(plot_results):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{f_name}\nGT:{gt.item():.1f} / Pred:{pred.item():.1f}", fontsize=8)
        ax.imshow(img_tensor.mul(torch.tensor([0.229, 0.224, 0.225])).add(torch.tensor([0.485, 0.456, 0.406])))

    for j in range(i + 1, len(axes)): axes[j].axis("off")
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "test.png")
    plt.savefig(plot_path)
    print(f"Test visualization saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml from the training log directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the best.ckpt model checkpoint")
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument("--fold", default=0, type=int, help="Fold number for cross-validation")
    parser.add_argument("--exp-name", type=str, help="A name for the experiment, used for log directory.")
    parser.add_argument("--n_samples", default=20, type=int, help="Number of samples to visualize.")
    
    args_cmd = parser.parse_args()

    # Load config from the specified YAML file
    with open(args_cmd.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create a namespace object and update it with loaded config and command-line args
    args = argparse.Namespace(**cfg)
    args.config = args_cmd.config
    args.ckpt = args_cmd.ckpt
    args.device = args_cmd.device
    args.fold = args_cmd.fold
    args.exp_name = args_cmd.exp_name
    args.n_samples = args_cmd.n_samples

    main(args)

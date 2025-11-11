"""
Pure Image Regression Tester
============================

This script tests a simple image regression model trained by train_regression.py.
"""

import argparse
import math
import os

# Add project root to Python path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Use existing dataset instead of complex logic in dataset/__init__.py.
from dataset.ajoumc_anemia import AjouMC_AnemiaDataset
from misc import MAE_error, setup_seed
from model.backbone import create_backbone


def main(args):
    # --- Additional settings after loading YAML config ---
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.hybrid_attrs = []  # Empty list since hybrid_mode is False

    # Set Korean font
    try:
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("Warning: NanumGothic font not found.")

    # --- Use data loading logic from test_ablation.py ---
    # Create anno_dict from Excel (empty dictionary since hybrid_mode is False)
    anno_dict = {}

    # Read test.txt
    test_data = []
    with open(args.test_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_path = parts[0]
            Hb = float(parts[1])
            test_data.append((Path(img_path), {"Hb": Hb, "Hybrid": []}))
    test_dataset = AjouMC_AnemiaDataset(args, test_data, mode="test", w_filename=True)
    setup_seed(args.seed)

    # --- Model loading (key changes) ---
    backbone, out_dim = create_backbone(args)

    # Use same model structure as train_regression.py
    class SimpleRegressionModel(nn.Module):
        def __init__(self, backbone, out_dim):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.regression_head = nn.Linear(out_dim, 1)

        def forward(self, x):
            features = self.backbone(x)
            pooled = self.pool(features)
            flattened = self.flatten(pooled)
            return self.regression_head(flattened)

    model = SimpleRegressionModel(backbone, out_dim).to(args.device)

    ckpt = torch.load(args.ckpt, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- Evaluation loop (similar to existing) ---
    all_results = []
    plot_results = []

    pbar = tqdm(range(len(test_dataset)), desc="Evaluating test set")
    for index in pbar:
        with torch.no_grad():
            img, target, hybrid_anno, f_name = test_dataset.__getitem__(
                index
            )  # hybrid_anno is ignored but must be received
            img, target = img.to(args.device).unsqueeze(0), target.to(args.device).unsqueeze(0)

            prediction = model(img)

            all_results.append(
                {
                    "w_filename": f_name,
                    "ground truth": target.item(),
                    "prediction": prediction.item(),
                }
            )

            if index < args.n_samples:
                plot_results.append([img.cpu()[0].permute(1, 2, 0), target, prediction, f_name])

    # --- Result saving and visualization (same as existing) ---
    df = pd.DataFrame(all_results)
    mae = MAE_error(torch.tensor(df["prediction"].values), torch.tensor(df["ground truth"].values)).mean().item()
    print(f"\nOverall Test MAE: {mae:.4f}")

    excel_path = os.path.join(args.log_dir, "results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Test results saved to {excel_path}")

    cols = 5
    rows = math.ceil(len(plot_results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (img, s1, s2, f_name) in enumerate(plot_results):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{f_name}\nGT:{s1.item():.1f} / Pred:{s2.item():.1f}", fontsize=8)
        ax.imshow(img.mul(torch.tensor([0.229, 0.224, 0.225])).add(torch.tensor([0.485, 0.456, 0.406])))

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(args.log_dir, "test.png")
    plt.savefig(plot_path)
    print(f"Test visualization saved to {plot_path}")


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "-c",
        "--config",
        default="configs/ajoumc_rxt50_image_regression.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--exp-name", required=True, type=str, help="A name for the experiment.")
    parser.add_argument("--fold", required=True, type=int, help="Fold number for cross-validation.")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the checkpoint file to evaluate.")
    parser.add_argument("--test-file", required=True, type=str, help="Path to the test.txt file.")
    parser.add_argument("--n_samples", default=20, type=int, help="Number of samples to visualize.")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    # Set test log directory
    log_dir_name = f"{args.exp_name}-fold{args.fold}"
    args.log_dir = os.path.join("logs", "test", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)

    main(args)

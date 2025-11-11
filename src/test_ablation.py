"""
Usage: python test.py --config configs/ajoumc_rxt50_image.yaml --ckpt path/to/your/checkpoint.pth
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
import yaml
from tqdm import tqdm

from dataset import load_dataset
from dataset.ajoumc_anemia import AjouMC_AnemiaDataset
from misc import MAE_error, setup_seed
from model import HbNet
from ordinal_regression import create_ordinal_lookup, decode_ordinal, encode_ordinal


def main(args):

    # Set Korean font (prevent Korean character corruption in plot)
    try:
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("Warning: NanumGothic font not found. Korean characters in the plot may not display correctly.")
        print("Please install a Korean font (e.g., 'sudo apt-get install fonts-nanum*')")

    args.lookup = create_ordinal_lookup(args)
    print(f"Ordinal lookup table: {args.lookup}")

    # Read test.txt (in ablation, according to baseline fold)
    # --exp-name is baseline experiment name (e.g., hybrid_demo-bins68)
    baseline_log_dir = os.path.join("logs", "train", f"{args.exp_name}-fold{args.fold}")
    test_file = os.path.join(baseline_log_dir, "test.txt")
    test_data = []
    with open(test_file, "r") as f:
        for line in f:
            parts = line.strip().split("	")
            img_path = parts[0]
            Hb = float(parts[1])
            test_data.append((Path(img_path), {"Hb": Hb}))

    test_dataset = AjouMC_AnemiaDataset(args, test_data, mode="test", w_filename=True)

    args.seed += args.fold
    setup_seed(args.seed)

    model = HbNet(args).to(args.device)

    # `weights_only=True` is a security best practice.
    ckpt = torch.load(args.ckpt, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])

    model.eval()

    # List to store all results for Excel saving
    all_results = []
    # List to store n_samples results for image visualization
    plot_results = []

    # 1. Run loop over entire test dataset.
    pbar = tqdm(range(len(test_dataset)), desc="Evaluating test set")
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

            # 2. Add all results to all_results list. (for Excel)
            all_results.append(
                {
                    "w_filename": f_name,
                    "ground truth": target.item(),
                    "prediction": prediction.item(),
                }
            )

            # 3. Add only '--n_samples' number of results to plot_results list. (for Plotting)
            if index < args.n_samples:
                plot_results.append(
                    [
                        img.cpu()[0].permute(1, 2, 0),
                        target,
                        prediction,
                        f_name,
                    ]
                )

    # 4. Generate Excel file for all results using 'all_results' and calculate MAE.
    df = pd.DataFrame(all_results)
    mae = MAE_error(torch.tensor(df["prediction"].values), torch.tensor(df["ground truth"].values)).mean().item()
    print(f"\nOverall Test MAE: {mae:.4f}")

    excel_path = os.path.join(args.log_dir, "results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Test results saved to {excel_path}")

    # 5. Draw only '--n_samples' number of images using 'plot_results'.
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

    # Hide remaining (empty) subplots.
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
        default="configs/ajoumc_rxt50_image.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument("--bins", type=int, help="Number of bins for ordinal regression")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for cross-validation")
    parser.add_argument("--ckpt", required=True, type=str, help="Checkpoint file to evaluate")
    parser.add_argument("--n_samples", default=100, type=int, help="Number of samples to visualize")
    parser.add_argument("--exp-name", type=str, help="A name for the experiment, used for log directory.")
    parser.add_argument("--no_enc", action="store_true", help="Disable encoder usage")
    parser.add_argument("--no_ord", action="store_true", help="Disable ordinal regression")
    # --- Arguments for External Validation Override ---
    parser.add_argument("--dataset", type=str, help="Override the dataset type specified in the config file.")
    parser.add_argument("--test_file", type=str, help="Override the test file path specified in the config file.")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            # If ckpt path exists in both config file and command line, command line argument takes precedence.
            # To do this, remove ckpt from cfg in advance.
            if "ckpt" in cfg:
                del cfg["ckpt"]
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    """
    Overwrite arguments with command line arguments.
    """
    if args.bins is not None:
        args.ordinal_bins = args.bins

    if args.exp_name:
        # Actual log directory name is created including ablation options
        baseline_exp_name = args.exp_name
        suffix = ""
        if args.no_enc:
            suffix += "-no_enc"
        if args.no_ord:
            suffix += "-no_ord"
        log_dir_name = f"{baseline_exp_name}{suffix}-fold{args.fold}"
    else:
        # Fallback for standalone execution, maintaining the original naming convention
        log_dir_name = f"{time.strftime('%y%m%d%H%M%S')}-{args.dataset}-{args.backbone}-Fold{args.fold}"

    args.log_dir = os.path.join("logs", "test", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    main(args)

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
    # 5-fold test setup
    folds = list(range(5))  # 0, 1, 2, 3, 4

    # Determine shot type (few-shot or zero-shot)
    shot_type = "few" if args.is_few_shot else "zero"
    print(f"Running {shot_type}-shot evaluation for {args.external_type} dataset")

    for fold in folds:
        print(f"\n{'='*80}")
        print(f"Starting {shot_type}-shot test for {args.external_type.upper()} dataset - Fold {fold}")
        print(f"{'='*80}")

        # Update settings per fold
        args.fold = fold

        # Set test data path (same)
        # if args.external_type == 'ghana':
        #     test_csv = 'external_validation_proceed/test_ext_ghana.csv'
        # else:  # eyedye
        test_csv = "external_validation_proceed/test_ext_eyedye.csv"

        # Set checkpoint path (few-shot vs zero-shot)
        if args.is_few_shot:
            model_type = f"image-bins{args.ordinal_bins}"
            ckpt_path = f"logs/train/{model_type}-ext-{args.external_type}-fold{fold}/best.ckpt"
        else:
            # Zero-shot: Use original model
            model_type = f"image-bins{args.ordinal_bins}"
            base_dir = "logs/train"
            ckpt_path = f"{base_dir}/{model_type}-fold{fold}/best.ckpt"

        # Set test result save path
        model_type_for_log = f"image-bins{args.ordinal_bins}"
        log_dir = f"logs/test/{model_type_for_log}-ext-{args.external_type}-{shot_type}-fold{fold}"

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        print(f"{shot_type.capitalize()}-shot test setup for fold {fold}:")
        print(f"  External type: {args.external_type}")
        print(f"  Test data: {test_csv}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Output directory: {log_dir}")

        # Check if checkpoint file exists
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint file not found: {ckpt_path}")
            print(f"Skipping fold {fold}")
            continue

        # Check if test data file exists
        if not os.path.exists(test_csv):
            print(f"ERROR: Test data file not found: {test_csv}")
            print(f"Skipping fold {fold}")
            continue

        # Set Korean font (prevent Korean character corruption in plot)
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
        except:
            print(
                "Warning: Matplotlib font configuration failed. Korean characters in the plot may not display correctly."
            )

        args.lookup = create_ordinal_lookup(args)
        print(f"Ordinal lookup table: {args.lookup}")

        # Load external data (CSV)
        df = pd.read_csv(test_csv)

        test_data = []
        print(f"Processing test data: {test_csv}")
        print(f"DataFrame shape: {df.shape}")

        for _, row in df.iterrows():
            # Use full_path from CSV file (already created with correct path)
            img_path = row["full_path"]

            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"WARNING: Image file not found: {img_path}")
                continue

            Hb = float(row["Hb"])

            test_data.append((Path(img_path), {"Hb": Hb, "Hybrid": []}))

        print(f"Loaded {len(test_data)} test samples")

        if len(test_data) == 0:
            print(f"ERROR: No test data loaded for fold {fold}!")
            continue

        test_dataset = AjouMC_AnemiaDataset(args, test_data, mode="test", w_filename=True)

        # Use different seed per fold
        args.seed += fold
        setup_seed(args.seed)

        model = HbNet(args).to(args.device)

        # Load model
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print("✓ Checkpoint loaded successfully")

        model.eval()

        # List to store all results for Excel saving
        all_results = []
        # List to store n_samples results for image visualization
        plot_results = []

        # 1. Run loop over entire test dataset.
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
        df_results = pd.DataFrame(all_results)
        mae = (
            MAE_error(torch.tensor(df_results["prediction"].values), torch.tensor(df_results["ground truth"].values))
            .mean()
            .item()
        )
        print(f"Fold {fold} {shot_type.capitalize()}-shot Test MAE: {mae:.4f}")

        excel_path = os.path.join(log_dir, "results.xlsx")
        df_results.to_excel(excel_path, index=False)
        print(f"Test results saved to {excel_path}")

        # 5. Draw only '--n_samples' number of images using 'plot_results'.
        cols = 5
        rows = math.ceil(len(plot_results) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = [axes]  # Convert axes to list if single row
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
        plot_path = os.path.join(log_dir, "test.png")
        plt.savefig(plot_path)
        plt.close()  # Close figure to save memory
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
    parser.add_argument(
        "--external_type",
        type=str,
        default="eyedye",
        choices=["eyedye"],
        help="External dataset type (only eyedye supported)",
    )  # ghana removed
    parser.add_argument("--external_csv", type=str, help="Path to external validation CSV/Excel file")

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

    if args.no_enc:
        args.use_encoder = False
    if args.no_ord:
        args.use_ordinal_regression = False

    shot_type = "few" if args.is_few_shot else "zero"
    print(f"Testing {args.external_type} dataset with 5-fold {shot_type}-shot models")

    main(args)

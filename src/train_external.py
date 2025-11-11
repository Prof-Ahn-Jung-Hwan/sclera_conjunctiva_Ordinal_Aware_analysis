"""
Ajou University Medical Center & ETRI Anemia Analysis AI Project

This script trains a deep learning model for anemia analysis.
- Loads hyperparameters and dataset info from a YAML config file.
- Supports cross-validation and ordinal regression options.
- Automatically saves training logs and config files.

Usage:
    ## Image only, ordinal regression with 60 bins, Training fold 0
    python train.py --config configs/ajoumc_rxt50_image.yaml --bins 60 --fold 0

    ## Image only, ordinal regression with 60 bins, Training fold 1
    python train.py --config configs/ajoumc_rxt50_image.yaml --bins 60 --fold 1

    ## Ablation study, Image only, no encoder, Training fold 0
    python train.py --config configs/ajoumc_rxt50_image.yaml --bins 60 --fold 0 --no_enc

    ## Ablation study, Image only, no ordinal regression, Training fold 0
    python train.py --config configs/ajoumc_rxt50_image.yaml --bins 60 --fold 0 --no_ord

    ## Ablation study, Image only, no encoder, no ordinal regression, Training fold 0 (=Regression mode)
    python train.py --config configs/ajoumc_rxt50_image.yaml --bins 60 --fold 0 --no_enc --no_ord

    ## Image + Hybrid positive attributes group 1, with 60 bins, Training fold 0
    python train.py --config configs/ajoumc_rxt50_hybrid_p1.yaml --bins 60 --fold 0

    ## Image + Hybrid positive attributes group 2, with 60 bins, Training fold 0
    python train.py --config configs/ajoumc_rxt50_hybrid_p2.yaml --bins 60 --fold 0

    ## Image + Hybrid negative attributes group, with 60 bins, Training fold 0
    python train.py --config configs/ajoumc_rxt50_hybrid_n.yaml --bins 60 --fold 0
"""

import argparse
import os

# Add project root to Python path
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from dataset.ajoumc_anemia import AjouMC_AnemiaDataset, calc_imbalance
from misc import MAE_error, adjust_learning_rate, log_data_split, plot_logs, setup_seed
from model import HbNet, create_optimizer
from model.loss import AnemiaLoss
from ordinal_regression import create_ordinal_lookup, decode_ordinal, encode_ordinal


def main(args):
    # 5-fold few-shot training setup
    folds = list(range(5))  # 0, 1, 2, 3, 4

    for fold in folds:
        print(f"\n{'='*80}")
        print(f"Starting few-shot training for {args.external_type.upper()} dataset - Fold {fold}")
        print(f"{'='*80}")

        # Update settings per fold
        args.fold = fold

        # Set data path for few-shot training
        # if args.external_type == 'ghana':
        #     args.external_csv = 'external_validation_proceed/train_ext_ghana.csv'
        # else:  # eyedye
        args.external_csv = "external_validation_proceed/train_ext_eyedye.csv"

        # Set original checkpoint path (select from 5 folds)
        # Determine model type (based on hybrid_mode)
        model_type = f"image-bins{args.ordinal_bins}"
        base_dir = "logs/train"
        args.ckpt = f"{base_dir}/{model_type}-fold{fold}/best.ckpt"

        # Clearly set few-shot training result save path
        args.log_dir = f"logs/train/{model_type}-ext-{args.external_type}-fold{fold}"

        # Create log directory
        os.makedirs(args.log_dir, exist_ok=True)

        print(f"Few-shot training setup for fold {fold}:")
        print(f"  External type: {args.external_type}")
        print(f"  Training data: {args.external_csv}")
        print(f"  Original checkpoint: {args.ckpt}")
        print(f"  Output directory: {args.log_dir}")

        # Check if checkpoint file exists
        if not os.path.exists(args.ckpt):
            print(f"ERROR: Checkpoint file not found: {args.ckpt}")
            print(f"Skipping fold {fold}")
            continue

        args.lookup = create_ordinal_lookup(args)
        print(f"Ordinal lookup table: {args.lookup}")

        # Load external data (CSV/Excel)
        # if args.external_type == 'ghana':
        #     df = pd.read_csv(args.external_csv)
        # else:  # eyedye
        df = pd.read_csv(args.external_csv)

        train_data = []
        print(f"Processing CSV file: {args.external_csv}")
        print(f"External type: {args.external_type}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")

        # Data loading section
        for idx, row in df.iterrows():
            print(f"Processing row {idx+1}/{len(df)}")

            # Use full_path from CSV file (already created with correct path)
            img_path = row["full_path"]
            print(f"Image path: {img_path}")

            # Check if image file exists
            if not os.path.exists(img_path):
                print(f"ERROR: Image file not found: {img_path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Path from CSV: {row['full_path']}")
                break  # Stop fold and move to next fold
            else:
                print(f"✓ Image file found: {img_path}")

            # Process Hb value - stop immediately if empty or invalid value exists
            try:
                hb_value = row["Hb"]
                print(f"Hb value: '{hb_value}', type: {type(hb_value)}")
                if pd.isna(hb_value) or hb_value == "" or hb_value == "_" or str(hb_value).strip() == "":
                    print(f"ERROR: Invalid Hb value '{hb_value}' for {img_path}")
                    print(f"Row data: {row.to_dict()}")
                    break  # Stop fold and move to next fold
                Hb = float(hb_value)
                print(f"✓ Hb converted to: {Hb}")
            except (ValueError, TypeError) as e:
                print(f"ERROR: Cannot convert Hb value '{row['Hb']}' to float for {img_path}: {e}")
                print(f"Row data: {row.to_dict()}")
                break  # Stop fold and move to next fold

            train_data.append((Path(img_path), {"Hb": Hb, "Hybrid": []}))
            print(f"✓ Added sample {idx+1}: {img_path} -> Hb={Hb}")
            print("=" * 50)

        print(f"Loaded {len(train_data)} training samples from {args.external_csv}")

        if len(train_data) == 0:
            print(f"ERROR: No training data loaded for fold {fold}!")
            continue

        # Few-shot training: 60 epochs
        args.epochs = 60

        # Few-shot training performs only training without validation
        # Create dataset - train only
        train_dataset = AjouMC_AnemiaDataset(args, train_data, mode="train", w_filename=True)

        # Calculate imbalance
        imbalance, cls_num_list = calc_imbalance(args, train_data)

        print(f"Train dataset size: {len(train_dataset)}")

        # Set batch size for few-shot training
        batch_size_config = {"eyedye": 10, "ghana": 20}  # 30 data samples  # 100 data samples

        # Map external_type to batch_size_config key
        dataset_key = "eyedye" if args.external_type == "joint" else args.external_type
        train_batch_size = batch_size_config.get(dataset_key, args.train_batch_size)  # Fallback to default value
        print(f"Using batch size {train_batch_size} for dataset {dataset_key}")

        # Use different seed per fold
        args.seed += fold
        setup_seed(args.seed)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=args.train_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,  # Set to False to use all samples in small datasets
            persistent_workers=True,
        )
        per_cls_weights = 1.0 / (np.array(cls_num_list) + 1e-8)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.device)

        model = HbNet(args).to(args.device)

        # Few-shot training: Load pre-trained checkpoint
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=f"cuda:{args.device}")

        # Debug: Check checkpoint structure
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if "hybrid_proj.0.weight" in state_dict:
            print(f"Checkpoint hybrid_proj.0.weight shape: {state_dict['hybrid_proj.0.weight'].shape}")

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")

        # Few-shot training: Backbone freeze
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen for few-shot training")
        elif hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen for few-shot learning")

        optimizer, scheduler = create_optimizer(args, model, len(train_dataloader))
        loss_fn = AnemiaLoss(args, per_cls_weights)

        xs, ys1, ys2 = [], [], []
        best_train_mae = float("inf")
        global_step = 0

        print(f"Starting few-shot training for fold {fold} - {args.epochs} epochs...")

        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            train_result, global_step = train_one_epoch(
                args, train_dataloader, model, optimizer, loss_fn, scheduler, epoch, global_step
            )

            xs.append(epoch)
            ys1.append(train_result["train/loss"])
            ys2.append(train_result["train/mae"])

            # In few-shot learning, save best model based on train loss
            if train_result["train/mae"] < best_train_mae:
                best_train_mae = train_result["train/mae"]
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(args.log_dir, f"best.ckpt"),
                )

            print(
                f'FOLD {fold} EPOCH {epoch+1}/{args.epochs} >> loss: {train_result["train/loss"]:.4f}  mae: {train_result["train/mae"]:.4f}   best train mae: {best_train_mae:.4f}'
            )

        # Save final log after training completion
        plot_logs(
            xs,
            xs,  # Use train xs since there's no test in few-shot
            ys1,
            ys2,
            ys1,  # Use train loss since there's no test in few-shot
            ys2,  # Use train mae since there's no test in few-shot
            fig_name=os.path.join(args.log_dir, "log.png"),
        )

        # Save final model
        torch.save(
            {
                "epoch": args.epochs - 1,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
            },
            os.path.join(args.log_dir, "last.ckpt"),
        )

        print(f"Fold {fold} few-shot training completed! Best train MAE: {best_train_mae:.4f}")
        print(f"Models saved to: {args.log_dir}")
        print(f"{'='*80}")

    print(f"\n{'='*80}")
    print(f"ALL FOLDS TRAINING COMPLETED FOR {args.external_type.upper()} DATASET")
    print(f"{'='*80}")


def train_one_epoch(args, dataloader, model, optimizer, loss_fn, scheduler, epoch, global_step):
    model.train()
    pbar = tqdm(dataloader)
    log_losses = []
    log_maes = []
    seen = 0
    for img, target, anno, _ in pbar:
        optimizer.zero_grad()
        img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
        seen += len(target)
        label = encode_ordinal(args, target).long()
        model_out = model(img, label=label)
        loss = loss_fn(model_out, target, epoch >= args.start_ib_epoch)

        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}, skipping step")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step_update(global_step)

        global_step += 1

        mae_error = MAE_error(
            (decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]),
            target,
        )

        pbar.set_description(
            f"[{epoch}/{args.epochs}] lr:  {optimizer.param_groups[0]['lr']:.7f}  loss: {loss:.4f}  MAE: {mae_error.mean().item():.4f}"
        )
        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())

    if len(log_losses) == 0:
        print(f"ERROR: No training samples processed in epoch {epoch}")
        print("This indicates a serious data loading problem. Training cannot continue.")
        exit(1)

    log_losses = sum(log_losses) / len(log_losses)
    log_maes = sum(log_maes) / seen
    return {"train/loss": log_losses, "train/mae": log_maes}, global_step


@torch.no_grad()
def test_one_epoch(args, dataloader, model, loss_fn, epoch):
    model.eval()
    pbar = tqdm(dataloader)
    log_losses = []
    log_maes = []
    seen = 0
    for img, target, anno, _ in pbar:
        img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
        seen += len(target)
        model_out = model(img)

        loss = loss_fn(model_out, target)
        if torch.isnan(loss):
            mae_error = torch.tensor(float("inf"))
        else:
            mae_error = MAE_error(
                (decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]),
                target,
            )

        pbar.set_description(f"[TEST@({epoch})Epoch] loss: {loss:.4f}  MAE: {mae_error.mean().item():.4f}")
        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())

    if len(log_losses) == 0:
        print(f"Warning: No test samples processed in epoch {epoch}")
        return {"test/loss": float("inf"), "test/mae": float("inf")}

    log_losses = sum(log_losses) / len(log_losses)
    log_maes = sum(log_maes) / seen
    return {"test/loss": log_losses, "test/mae": log_maes}


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument(
        "-c",
        "--config",
        default="configs/ajoumc_rxt50_image.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument("--fold", default=0, type=int, help="Fold number for cross-validation")
    parser.add_argument("--bins", type=int, help="Number of bins for ordinal regression")
    parser.add_argument("--no_enc", action="store_true", help="Disable encoder usage")
    parser.add_argument("--no_ord", action="store_true", help="Disable ordinal regression")
    parser.add_argument("--exp-name", type=str, help="A name for the experiment, used for log directory.")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file to load for few-shot training")
    # --- Arguments for External Validation ---
    parser.add_argument(
        "--external_type",
        type=str,
        default="eyedye",
        choices=["eyedye"],
        help="External dataset type (only eyedye supported)",
    )  # ghana removed
    parser.add_argument("--external_csv", type=str, help="Path to external validation CSV/Excel file")

    args_config, remaining_args = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining_args)

    """
    Overwrite arguments with command line arguments.
    """
    if args.bins is not None:
        args.ordinal_bins = args.bins

    # Set default path for external validation
    if not hasattr(args, "external_csv") or not args.external_csv:
        if args.external_type == "ghana":
            args.external_csv = f"external_validation_ghana/fold{args.fold}/train_ext_forFewShot.csv"
        else:
            args.external_csv = f"external_validation_joint_results/fold{args.fold}/train_ext_forFewShot.csv"

    if args.no_enc:
        args.use_encoder = False
    if args.no_ord:
        args.use_ordinal_regression = False

    if not args.use_ordinal_regression:
        # Adjust hyperparameters when in regression mode (if needed)
        # args.lr = 0.0005
        # args.lambda_1 = 0.000001
        # args.lambda_2 = 0.01
        # args.lambda_3 = 1
        pass

    if args.exp_name:
        log_dir_name = f"{args.exp_name}-ext-{args.external_type}-fold{args.fold}"

    args.log_dir = os.path.join("logs", "train", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    main(args)

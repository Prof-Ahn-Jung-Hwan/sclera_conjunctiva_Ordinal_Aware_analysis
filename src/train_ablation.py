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

    args.lookup = create_ordinal_lookup(args)
    print(f"Ordinal lookup table: {args.lookup}")

    # Read train.txt
    # Extract baseline exp_name (remove suffix or use baseline-exp-name)
    if hasattr(args, "baseline_exp_name") and args.baseline_exp_name:
        baseline_exp_name = args.baseline_exp_name
    else:
        baseline_exp_name = args.exp_name.split("-no_enc")[0].split("-no_ord")[0]
    baseline_log_dir = os.path.join("logs", "train", f"{baseline_exp_name}-fold{args.fold}")
    train_file = os.path.join(baseline_log_dir, "train.txt")
    train_data = []
    with open(train_file, "r") as f:
        for line in f:
            # Modified: If no argument is given to split(), both tab and space are treated as delimiters
            parts = line.strip().split()
            img_path = parts[0]
            Hb = float(parts[1])
            train_data.append((Path(img_path), {"Hb": Hb}))

    # Read test.txt
    test_file = os.path.join(baseline_log_dir, "test.txt")
    test_data = []
    with open(test_file, "r") as f:
        for line in f:
            # Modified: If no argument is given to split(), both tab and space are treated as delimiters
            parts = line.strip().split()
            img_path = parts[0]
            Hb = float(parts[1])
            test_data.append((Path(img_path), {"Hb": Hb}))

    # Create dataset
    train_dataset = AjouMC_AnemiaDataset(args, train_data, mode="train", w_filename=True)
    test_dataset = AjouMC_AnemiaDataset(args, test_data, mode="test", w_filename=True)

    # Calculate imbalance
    imbalance, cls_num_list = calc_imbalance(args, train_data)

    log_data_split(args, train_dataset, test_dataset)

    args.seed += args.fold
    setup_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.train_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,  # Set to False to use all samples in small datasets
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.test_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    per_cls_weights = 1.0 / (np.array(cls_num_list) + 1e-8)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(args.device)

    model = HbNet(args).to(args.device)
    optimizer, scheduler = create_optimizer(args, model, len(train_dataloader))
    scaler = torch.amp.GradScaler()  # Mixed precision training
    loss_fn = AnemiaLoss(args, per_cls_weights)

    xs, ys1, ys2 = [], [], []
    txs, tys1, tys2 = [], [], []
    best_test_mae = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_result, global_step = train_one_epoch(
            args, train_dataloader, model, optimizer, scaler, loss_fn, scheduler, epoch, global_step
        )

        xs.append(epoch)
        ys1.append(train_result["train/loss"])
        ys2.append(train_result["train/mae"])
        if (epoch + 1) % 2 == 0:
            test_result = test_one_epoch(args, test_dataloader, model, loss_fn, epoch)
            txs.append(epoch)
            tys1.append(test_result["test/loss"])
            tys2.append(test_result["test/mae"])

            # If MAE is nan, stop training and raise error
            if torch.isnan(torch.tensor(test_result["test/mae"])):
                print(f"\n\nERROR: MAE is NaN at epoch {epoch}. Stopping training for this fold.")
                # Return abnormal exit code so shell script can detect it
                exit(1)

            if not torch.isnan(torch.tensor(test_result["test/mae"])) and test_result["test/mae"] < best_test_mae:
                best_test_mae = test_result["test/mae"]
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(args.log_dir, f"best.ckpt"),
                )

            print(
                f'TEST REPORT >> loss: {test_result["test/loss"]:.4f}  mae: {test_result["test/mae"]:.4f}   best valid mae: {best_test_mae:.4f}'
            )

        plot_logs(
            xs,
            txs,
            ys1,
            ys2,
            tys1,
            tys2,
            fig_name=os.path.join(args.log_dir, "log.png"),
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
            },
            os.path.join(args.log_dir, "last.ckpt"),
        )


def train_one_epoch(args, dataloader, model, optimizer, scaler, loss_fn, scheduler, epoch, global_step):
    model.train()
    pbar = tqdm(dataloader)
    log_losses = []
    log_maes = []
    seen = 0
    for img, target, anno, _ in pbar:
        model.zero_grad()
        img, target = img.to(args.device), target.to(args.device)
        seen += len(target)
        label = encode_ordinal(args, target).long()
        
        # Mixed precision training with autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_out = model(img, label=label)
            loss = loss_fn(model_out, target, epoch >= args.start_ib_epoch)

        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}, skipping step")
            optimizer.zero_grad()
            continue

        # Mixed precision backward and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
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
    parser.add_argument(
        "--baseline-exp-name", type=str, help="Baseline experiment name for test split (ablation studies)"
    )

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
        log_dir_name = f"{args.exp_name}-fold{args.fold}"
    else:
        # Fallback for standalone execution, maintaining the original naming convention
        log_dir_name = (
            f"{time.strftime('%y%m%d%H%M%S')}-{Path(args_config.config).stem}-bins{args.ordinal_bins}-fold{args.fold}"
        )

    args.log_dir = os.path.join("logs", "train", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    main(args)

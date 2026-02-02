"""
Pure Image Regression Trainer
=============================

This script trains a simple image regression model, serving as a pure baseline
for ablation studies. It uses a standard backbone (e.g., ResNeXt-50) with a
simple regression head and a standard loss function (L1 or MSE).

This is different from train.py, which uses the complex HbNet and AnemiaLoss.
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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use existing dataset instead of complex logic in dataset/__init__.py.
from dataset.ajoumc_anemia import AjouMC_AnemiaDataset
from misc import MAE_error, adjust_learning_rate, log_data_split, plot_logs, setup_seed
from model.backbone import create_backbone


def main(args):
    # --- Additional settings after loading YAML config ---
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.hybrid_attrs = []  # Empty list since hybrid_mode is False

    # --- Use data loading logic from train_ablation.py ---
    # Create anno_dict from Excel (empty dictionary since hybrid_mode is False)
    anno_dict = {}

    # Read train.txt
    train_data = []
    with open(args.train_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_path = parts[0]
            Hb = float(parts[1])
            train_data.append((Path(img_path), {"Hb": Hb, "Hybrid": []}))

    # Read test.txt
    test_data = []
    with open(args.test_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_path = parts[0]
            Hb = float(parts[1])
            test_data.append((Path(img_path), {"Hb": Hb, "Hybrid": []}))

    train_dataset = AjouMC_AnemiaDataset(args, train_data, mode="train", w_filename=True)
    test_dataset = AjouMC_AnemiaDataset(args, test_data, mode="test", w_filename=True)

    args.seed += args.fold
    setup_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.train_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,  # Set to False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.test_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # --- Model and loss function definition (key changes) ---
    # 1. Build model with pure Backbone and Regression Head
    backbone, out_dim = create_backbone(args)
    regression_head = nn.Linear(out_dim, 1)

    # Add AdaptiveAvgPool2d and Flatten to resolve dimension issues
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

    # 2. Use standard loss function (L1 Loss)
    loss_fn = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler()  # Mixed precision training

    # --- Training and evaluation loop (similar to existing) ---
    xs, ys1, ys2 = [], [], []
    txs, tys1, tys2 = [], [], []
    best_test_mae = float("inf")

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_result = train_one_epoch(args, train_dataloader, model, optimizer, scaler, loss_fn, epoch)
        xs.append(epoch)
        ys1.append(train_result["train/loss"])
        ys2.append(train_result["train/mae"])

        if (epoch + 1) % 2 == 0:
            test_result = test_one_epoch(args, test_dataloader, model, loss_fn, epoch)
            txs.append(epoch)
            tys1.append(test_result["test/loss"])
            tys2.append(test_result["test/mae"])

            if test_result["test/mae"] < best_test_mae:
                best_test_mae = test_result["test/mae"]
                torch.save({"state_dict": model.state_dict()}, os.path.join(args.log_dir, "best.ckpt"))

            print(
                f'TEST REPORT >> loss: {test_result["test/loss"]:.4f}  mae: {test_result["test/mae"]:.4f}   best valid mae: {best_test_mae:.4f}'
            )

        plot_logs(xs, txs, ys1, ys2, tys1, tys2, fig_name=os.path.join(args.log_dir, "log.png"))
        torch.save({"state_dict": model.state_dict()}, os.path.join(args.log_dir, "last.ckpt"))


def train_one_epoch(args, dataloader, model, optimizer, scaler, loss_fn, epoch):
    model.train()
    pbar = tqdm(dataloader)
    log_losses, log_maes, seen = [], [], 0

    for img, target, hybrid_anno, _ in pbar:  # hybrid_anno is ignored but must be received
        optimizer.zero_grad()
        img, target = img.to(args.device), target.to(args.device)
        seen += len(target)

        # Mixed precision training with autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(img)
            loss = loss_fn(prediction, target)

        # Mixed precision backward and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mae_error = MAE_error(prediction, target)
        pbar.set_description(
            f"[{epoch}/{args.epochs}] lr: {optimizer.param_groups[0]['lr']:.7f} loss: {loss.item():.4f} MAE: {mae_error.mean().item():.4f}"
        )
        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())

    return {"train/loss": np.mean(log_losses), "train/mae": np.sum(log_maes) / seen}


@torch.no_grad()
def test_one_epoch(args, dataloader, model, loss_fn, epoch):
    model.eval()
    pbar = tqdm(dataloader)
    log_losses, log_maes, seen = [], [], 0

    for img, target, hybrid_anno, _ in pbar:  # hybrid_anno is ignored but must be received
        img, target = img.to(args.device), target.to(args.device)
        seen += len(target)

        prediction = model(img)
        loss = loss_fn(prediction, target)
        mae_error = MAE_error(prediction, target)

        pbar.set_description(f"[TEST@({epoch})Epoch] loss: {loss.item():.4f}  MAE: {mae_error.mean().item():.4f}")
        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())

    return {"test/loss": np.mean(log_losses), "test/mae": np.sum(log_maes) / seen}


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
    parser.add_argument("--device", default=0, type=int, help="GPU device ID")
    parser.add_argument("--fold", required=True, type=int, help="Fold number for cross-validation")
    parser.add_argument("--exp-name", required=True, type=str, help="A name for the experiment.")
    parser.add_argument("--train-file", required=True, type=str, help="Path to the train.txt file.")
    parser.add_argument("--test-file", required=True, type=str, help="Path to the test.txt file.")

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    log_dir_name = f"{args.exp_name}-fold{args.fold}"
    args.log_dir = os.path.join("logs", "train", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)

    main(args)

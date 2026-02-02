"""
Ajou University Medical Center & ETRI Anemia Analysis AI Project

This script is a modified version of train.py, specifically for
efficient hyperparameter search of 'bins'. It includes an early
stopping mechanism to reduce unnecessary training time.
"""

import argparse
import os

# Add project root to Python path
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from misc import MAE_error, adjust_learning_rate, log_data_split, plot_logs, setup_seed
from model import HbNet, create_optimizer
from model.loss import AnemiaLoss
from ordinal_regression import create_ordinal_lookup, decode_ordinal, encode_ordinal


def main(args):

    args.lookup = create_ordinal_lookup(args)
    print(f"Ordinal lookup table: {args.lookup}")

    train_folds, test_folds, imbalance_folds, cls_num_list_folds = load_dataset(
        args, cv_fold=5, seed=args.seed, w_filename=True
    )

    log_data_split(args, train_folds[args.fold], test_folds[args.fold])

    args.seed += args.fold
    setup_seed(args.seed)

    train_dataloader = DataLoader(
        train_folds[args.fold],
        batch_size=args.train_batch_size,
        num_workers=args.train_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,  # Set to False to use all samples in small datasets
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_folds[args.fold],
        batch_size=args.test_batch_size,
        num_workers=args.test_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    cls_num_list = cls_num_list_folds[args.fold]
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

    # --- For recording best performance epoch ---
    best_epoch = 0

    # # --- (Disabled) Early stopping logic ---
    # # min_epochs_before_stop = args.min_epochs_before_stop
    # # patience = args.patience
    # # patience_counter = 0
    # # ---

    global_step = 0
    training_start_time = time.time()  # Track total training time
    
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

            if test_result["test/mae"] < best_test_mae:
                best_test_mae = test_result["test/mae"]
                best_epoch = epoch  # Record epoch when best performance achieved
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "opt_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(args.log_dir, f"best.ckpt"),
                )
                # patience_counter = 0 # (Disabled)
            # else:
            #     if epoch >= min_epochs_before_stop:
            #         patience_counter += 1

            print(
                f'TEST REPORT >> loss: {test_result["test/loss"]:.4f}  mae: {test_result["test/mae"]:.4f}   best valid mae: {best_test_mae:.4f} at epoch {best_epoch}'
            )

        # # --- (Disabled) Early stopping logic ---
        # if patience_counter >= patience:
        #     print(f"\nEarly stopping at epoch {epoch} as there was no improvement for {patience} epochs.")
        #     break

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
    
    # Track total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s ({total_training_time:.2f} seconds)")
    print(f"Best validation MAE: {best_test_mae:.4f} at epoch {best_epoch}")
    print("="*60)


def train_one_epoch(args, dataloader, model, optimizer, scaler, loss_fn, scheduler, epoch, global_step):
    model.train()
    pbar = tqdm(dataloader, desc=f"[{epoch}/{args.epochs}]")  # Modified tqdm description
    log_losses = []
    log_maes = []
    seen = 0
    for img, target, anno, _ in pbar:
        optimizer.zero_grad()
        img, target = img.to(args.device), target.to(args.device)
        seen += len(target)
        label = encode_ordinal(args, target).long()
        
        # Mixed precision training with autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_out = model(img, label=label)
            loss = loss_fn(model_out, target, epoch >= args.start_ib_epoch)

        # Mixed precision backward and optimization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Gradient clipping for stability
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step_update(global_step)

        global_step += 1

        mae_error = MAE_error(
            (decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]),
            target,
        )

        pbar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.7f}", loss=f"{loss:.4f}", MAE=f"{mae_error.mean().item():.4f}"
        )
        log_losses.append(loss.item())
        log_maes.append(mae_error.sum().item())
    log_losses = sum(log_losses) / len(log_losses)
    log_maes = sum(log_maes) / seen
    return {"train/loss": log_losses, "train/mae": log_maes}, global_step


@torch.no_grad()
def test_one_epoch(args, dataloader, model, loss_fn, epoch):
    model.eval()
    pbar = tqdm(dataloader, desc=f"[TEST@({epoch})Epoch]")  # Modified tqdm description
    log_losses = []
    log_maes = []
    seen = 0
    for img, target, anno, _ in pbar:
        img, target, anno = img.to(args.device), target.to(args.device), anno.to(args.device)
        seen += len(target)
        model_out = model(img)

        loss = loss_fn(model_out, target)
        mae_error = MAE_error(
            (decode_ordinal(args, model_out["logits"]) if args.use_ordinal_regression else model_out["logits"]),
            target,
        )

        pbar.set_postfix(loss=f"{loss:.4f}", MAE=f"{mae_error.mean().item():.4f}")
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

    args_config, remaining_args = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining_args)

    if args.bins is not None:
        args.ordinal_bins = args.bins

    if args.no_enc:
        args.use_encoder = False
    if args.no_ord:
        args.use_ordinal_regression = False

    if not args.use_ordinal_regression:
        args.lr = 0.0005
        args.lambda_1 = 0.000001
        args.lambda_2 = 0.01
        args.lambda_3 = 1

    if args.exp_name:
        log_dir_name = f"{args.exp_name}-fold{args.fold}"
    else:
        log_dir_name = (
            f"{time.strftime('%y%m%d%H%M%S')}-{Path(args_config.config).stem}-bins{args.ordinal_bins}-fold{args.fold}"
        )

    args.log_dir = os.path.join("log_bins", "train", log_dir_name)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    main(args)

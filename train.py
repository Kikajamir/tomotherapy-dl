"""
Kaggle-ready training entry point.

Thin CLI wrapper around `training.train` -- reuses the existing
patient-wise split, `SlidingWindowDataset`, Attention U-Net, Weighted
Huber Loss, Adam optimizer, and AMP training loop exactly as implemented.
Nothing here changes model, loss, optimizer, or training methodology; it
only makes the PKL path and output paths configurable from the command
line (or the environment), so the same script runs unchanged locally and
on Kaggle -- the dataset path is the only thing that needs to differ.

No preprocessing happens here. `--data-path` must point at an already
generated `processed_data.pkl` (see `run_pipeline.py`).

Usage:
    python train.py --data-path processed_data.pkl
    python train.py --data-path /kaggle/input/<dataset-name>/processed_data.pkl
"""

import argparse
import json

import torch

from configs.config import DATA_PATH, SAVE_PATH
from training.train import (
    load_and_split_patients,
    build_dataloaders,
    build_generator,
    build_optimizer,
    train,
)
from model.losses import WeightedHuberLoss


def save_loss_curve(train_history, val_history, path):
    """Best-effort: saves a PNG of the train/val loss curves. Never raises --
    a missing/broken matplotlib backend should not fail a training run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(train_history, label="Train Loss")
        plt.plot(val_history, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Huber Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Loss curve saved to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not save loss curve ({exc}); continuing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DATA_PATH,
                         help=f"Path to processed_data.pkl (default: {DATA_PATH}).")
    parser.add_argument("--save-path", default=SAVE_PATH,
                         help=f"Where to save the best checkpoint (default: {SAVE_PATH}).")
    parser.add_argument("--history-path", default="training_history.json",
                         help="Where to save per-epoch train/val loss as JSON.")
    parser.add_argument("--loss-curve-path", default="loss_curve.png",
                         help="Where to save the loss-curve PNG.")
    args = parser.parse_args()

    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Data path      : {args.data_path}")
    print(f"Checkpoint out : {args.save_path}")

    train_data, val_data, test_data = load_and_split_patients(data_path=args.data_path)

    train_loader, val_loader, test_loader = build_dataloaders(train_data, val_data, test_data)

    generator = build_generator()
    optimizer = build_optimizer(generator)
    criterion = WeightedHuberLoss()

    train_history, val_history = train(
        generator, optimizer, criterion, train_loader, val_loader,
        save_path=args.save_path,
    )

    with open(args.history_path, "w") as f:
        json.dump({"train_loss": train_history, "val_loss": val_history}, f, indent=2)
    print(f"History saved to {args.history_path}")

    save_loss_curve(train_history, val_history, args.loss_curve_path)


if __name__ == "__main__":
    main()

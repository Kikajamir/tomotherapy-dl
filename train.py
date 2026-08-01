"""
Kaggle-ready training entry point.

Thin CLI wrapper around `training.train` -- reuses the existing
patient-wise split, `SlidingWindowDataset`, Attention U-Net, Adam
optimizer, and AMP training loop exactly as implemented. `--loss-type`
selects between the baseline loss (WeightedHuberLoss, unchanged), the
WeightedL1Loss ablation, and the WeightedHuberGradientLoss ablation (see
`configs.config.LOSS_TYPE`); nothing else about the model, optimizer, or
training methodology changes. Also makes the PKL path and output paths
configurable from the command line (or the environment), so the same
script runs unchanged locally and on Kaggle -- the dataset path is the
only thing that needs to differ there (`configs.config.DATA_PATH`
defaults to the Kaggle input path but honors a `TOMOQA_DATA_PATH`
environment variable, and `--data-path` below overrides both).

The Sigmoid Output (`OUTPUT_ACTIVATION`), Deep Supervision
(`USE_DEEP_SUPERVISION`), and Multi-Scale Loss (`USE_MULTI_SCALE_LOSS`)
ablations are selected purely via `configs/config.py` -- no CLI flag is
needed for them, and each writes its checkpoint to its own
`experiments/<name>/` directory (see `--save-path` default below) so
running `python train.py` with no arguments after only editing
config.py just works. Do not enable more than one ablation at a time;
this script does not attempt to combine them.

No preprocessing happens here. `--data-path` must point at an already
generated `processed_data.pkl` (see `run_pipeline.py`).

Usage:
    python train.py --data-path processed_data.pkl
    python train.py --data-path /kaggle/input/<dataset-name>/processed_data.pkl
    python train.py --loss-type weighted_l1     # ablation run, see LOSS_TYPE
    python train.py --loss-type huber_gradient  # ablation run, see LOSS_TYPE
"""

import argparse
import json
import os

import torch

from configs.config import (
    DATA_PATH,
    SAVE_PATH,
    WEIGHTED_L1_SAVE_PATH,
    HUBER_GRADIENT_SAVE_PATH,
    SIGMOID_OUTPUT_SAVE_PATH,
    DEEP_SUPERVISION_SAVE_PATH,
    MULTI_SCALE_LOSS_SAVE_PATH,
    BEAM_WEIGHTED_SAVE_PATH,
    LOSS_TYPE,
    OUTPUT_ACTIVATION,
    USE_DEEP_SUPERVISION,
    USE_MULTI_SCALE_LOSS,
)
from training.train import (
    load_and_split_patients,
    build_dataloaders,
    build_generator,
    build_optimizer,
    build_criterion,
    train,
)


def save_loss_curve(train_history, val_history, path, loss_type=LOSS_TYPE):
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
        plt.ylabel(f"Loss ({loss_type})")
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
    parser.add_argument("--loss-type", default=LOSS_TYPE,
                         choices=["l1", "weighted_l1", "huber_gradient", "beam_weighted"],
                         help=f"Loss ablation switch (default: {LOSS_TYPE!r}). "
                              "'l1' is the unchanged baseline (WeightedHuberLoss); "
                              "'weighted_l1' selects WeightedL1Loss; "
                              "'huber_gradient' selects WeightedHuberGradientLoss "
                              "(WeightedHuberLoss + lambda_gradient * GradientLoss); "
                              "'beam_weighted' selects BeamAwareWeightedHuberLoss "
                              "(region-reweighted by a planned-derived beam mask, see "
                              "BEAM_THRESHOLD/BEAM_WEIGHT/BACKGROUND_WEIGHT).")
    parser.add_argument("--save-path", default=None,
                         help="Where to save the best checkpoint. Defaults to "
                              f"{SAVE_PATH!r} for --loss-type l1, "
                              f"{WEIGHTED_L1_SAVE_PATH!r} for --loss-type weighted_l1, or "
                              f"{HUBER_GRADIENT_SAVE_PATH!r} for --loss-type huber_gradient "
                              "(unless OUTPUT_ACTIVATION/USE_DEEP_SUPERVISION/"
                              "USE_MULTI_SCALE_LOSS in configs/config.py select one of the "
                              f"other ablations: {SIGMOID_OUTPUT_SAVE_PATH!r}, "
                              f"{DEEP_SUPERVISION_SAVE_PATH!r}, or "
                              f"{MULTI_SCALE_LOSS_SAVE_PATH!r} respectively), so an ablation "
                              "run never overwrites another experiment's checkpoint.")
    parser.add_argument("--history-path", default="training_history.json",
                         help="Where to save per-epoch train/val loss as JSON.")
    parser.add_argument("--loss-curve-path", default="loss_curve.png",
                         help="Where to save the loss-curve PNG.")
    args = parser.parse_args()

    # Sigmoid Output / Deep Supervision / Multi-Scale Loss are selected
    # purely via configs/config.py (not a CLI flag) and each get their own
    # checkpoint directory. Do not combine them -- only one is expected to
    # be active (non-default) at a time.
    if OUTPUT_ACTIVATION == "sigmoid":
        config_ablation_save_path = SIGMOID_OUTPUT_SAVE_PATH
    elif USE_DEEP_SUPERVISION:
        config_ablation_save_path = DEEP_SUPERVISION_SAVE_PATH
    elif USE_MULTI_SCALE_LOSS:
        config_ablation_save_path = MULTI_SCALE_LOSS_SAVE_PATH
    else:
        config_ablation_save_path = None

    default_save_paths = {
        "l1": SAVE_PATH,
        "weighted_l1": WEIGHTED_L1_SAVE_PATH,
        "huber_gradient": HUBER_GRADIENT_SAVE_PATH,
        "beam_weighted": BEAM_WEIGHTED_SAVE_PATH,
    }
    save_path = (
        args.save_path
        or config_ablation_save_path
        or default_save_paths[args.loss_type]
    )
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"CUDA available     : {torch.cuda.is_available()}")
    print(f"Data path          : {args.data_path}")
    print(f"Loss type          : {args.loss_type}")
    print(f"Output activation  : {OUTPUT_ACTIVATION}")
    print(f"Deep supervision   : {USE_DEEP_SUPERVISION}")
    print(f"Multi-scale loss   : {USE_MULTI_SCALE_LOSS}")
    print(f"Checkpoint out     : {save_path}")

    train_data, val_data, test_data = load_and_split_patients(data_path=args.data_path)

    train_loader, val_loader, test_loader = build_dataloaders(train_data, val_data, test_data)

    generator = build_generator()
    optimizer = build_optimizer(generator)
    criterion = build_criterion(args.loss_type)

    train_history, val_history, val_mae_history = train(
        generator, optimizer, criterion, train_loader, val_loader,
        save_path=save_path,
    )

    with open(args.history_path, "w") as f:
        json.dump(
            {
                "loss_type": args.loss_type,
                "train_loss": train_history,
                "val_loss_weighted": val_history,
                "val_mae_unweighted": val_mae_history,
            },
            f,
            indent=2,
        )
    print(f"History saved to {args.history_path}")

    save_loss_curve(train_history, val_history, args.loss_curve_path, loss_type=args.loss_type)


if __name__ == "__main__":
    main()

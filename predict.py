"""
Kaggle-ready prediction / evaluation entry point.

Thin CLI wrapper around `evaluation.evaluate` -- reuses the existing
sliding-window reconstruction and MAE/RMSE/PSNR/SSIM metrics exactly as
implemented. Loads the same `processed_data.pkl` used for training (via
`--data-path`) and regenerates the identical patient-wise split (same
`RANDOM_SEED`, via `training.train.load_and_split_patients`) to recover
the held-out test patients, then reconstructs and scores them with the
checkpoint `train.py` saved (via `--checkpoint`).

`evaluation.evaluate`'s own `plot_*` functions call `plt.show()`, which is
meant for interactive notebook use and saves nothing to disk -- on a
headless Kaggle script that would silently do nothing. Rather than change
that module, this script adds its own disk-saving figure step using the
same `reconstructed_results` data.

Usage:
    python predict.py --data-path processed_data.pkl --checkpoint best_generator_residual.pth
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs.config import DATA_PATH, SAVE_PATH, EVAL_SAVE_DIR
from training.train import load_and_split_patients
from evaluation.evaluate import (
    load_best_generator,
    reconstruct_patients,
    compute_metrics,
    save_results,
)
from evaluation.projection_profiles import generate_projection_profile_analysis


def save_comparison_figures(reconstructed_results, output_dir):
    """
    Saves, per test patient: a 3-panel planned/ground-truth/prediction
    sinogram comparison, and an absolute-error map. Pure output-saving,
    computed from the same arrays `evaluate.reconstruct_patients` already
    produced -- no new computation.
    """
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for patient_id, data in reconstructed_results.items():
        planned = data["planned"]
        ground_truth = data["ground_truth"]
        prediction = data["prediction"]

        vmin = min(planned.min(), ground_truth.min(), prediction.min())
        vmax = max(planned.max(), ground_truth.max(), prediction.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        titles = ["Planned Sinogram", "Ground Truth", "Predicted Sinogram"]
        images = [planned, ground_truth, prediction]

        im = None
        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, cmap="gray", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Detector Channel")
            ax.set_ylabel("Projection")

        fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02, label="Detector Signal")
        fig.suptitle(f"Patient: {patient_id}", fontsize=16, fontweight="bold")
        fig.savefig(os.path.join(figures_dir, f"{patient_id}_comparison.png"))
        plt.close(fig)

        error = np.abs(ground_truth - prediction)
        fig = plt.figure(figsize=(7, 8))
        im = plt.imshow(error, cmap="hot", aspect="auto", origin="lower")
        plt.title(f"Absolute Error Map ({patient_id})")
        plt.xlabel("Detector Channel")
        plt.ylabel("Projection")
        plt.colorbar(im, label="Absolute Error")
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"{patient_id}_error_map.png"))
        plt.close(fig)

    print(f"Comparison/error-map figures saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DATA_PATH,
                         help=f"Path to processed_data.pkl (default: {DATA_PATH}).")
    parser.add_argument("--checkpoint", default=SAVE_PATH,
                         help=f"Path to the trained checkpoint (default: {SAVE_PATH}).")
    parser.add_argument("--output-dir", default=EVAL_SAVE_DIR,
                         help=f"Where to save predictions/metrics/figures (default: {EVAL_SAVE_DIR}).")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip saving comparison/error-map PNGs (metrics + .npy arrays are always saved).")
    parser.add_argument("--no-projection-profiles", action="store_true",
                         help="Skip generating per-patient detector projection profile figures/CSV.")
    args = parser.parse_args()

    _, _, test_data = load_and_split_patients(data_path=args.data_path)

    generator = load_best_generator(checkpoint_path=args.checkpoint)
    reconstructed_results = reconstruct_patients(generator, test_data)

    metrics_df, summary = compute_metrics(reconstructed_results)
    print(metrics_df)
    print(summary)

    save_results(reconstructed_results, metrics_df, summary, save_dir=args.output_dir)

    if not args.no_figures:
        save_comparison_figures(reconstructed_results, args.output_dir)

    if not args.no_projection_profiles:
        generate_projection_profile_analysis(save_dir=args.output_dir)


if __name__ == "__main__":
    main()

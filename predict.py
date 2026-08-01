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

from configs.config import DATA_PATH, SAVE_PATH, EVAL_SAVE_DIR, BEAM_THRESHOLD
from training.train import load_and_split_patients
from evaluation.evaluate import (
    load_best_generator,
    reconstruct_patients,
    compute_metrics,
    save_results,
)
from evaluation.projection_profiles import generate_projection_profile_analysis
from model.losses import build_beam_mask


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


def save_region_diagnostic_figures(reconstructed_results, output_dir, beam_threshold=BEAM_THRESHOLD):
    """
    Saves, per test patient: a beam mask (planned > beam_threshold), a
    beam-only/background-only absolute-error pair, and a projection-wise
    beam/background/global MAE curve. Same disk-saving pattern as
    `save_comparison_figures` above -- computed from the same
    `reconstructed_results` arrays, no new computation beyond masking.
    """
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for patient_id, data in reconstructed_results.items():
        planned = data["planned"]
        ground_truth = data["ground_truth"]
        prediction = data["prediction"]

        beam_mask = build_beam_mask(planned, beam_threshold)
        error = np.abs(ground_truth - prediction)

        fig = plt.figure(figsize=(7, 8))
        plt.imshow(
            beam_mask.astype(np.float32), cmap="gray", aspect="auto",
            origin="lower", vmin=0, vmax=1,
        )
        plt.title(f"Beam Mask (planned > {beam_threshold}) -- {patient_id}")
        plt.xlabel("Detector Channel")
        plt.ylabel("Projection")
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"{patient_id}_beam_mask.png"))
        plt.close(fig)

        beam_error = np.where(beam_mask, error, np.nan)
        background_error = np.where(~beam_mask, error, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
        im0 = axes[0].imshow(beam_error, cmap="hot", aspect="auto", origin="lower")
        axes[0].set_title("Beam-only Absolute Error")
        axes[0].set_xlabel("Detector Channel")
        axes[0].set_ylabel("Projection")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(background_error, cmap="hot", aspect="auto", origin="lower")
        axes[1].set_title("Background-only Absolute Error")
        axes[1].set_xlabel("Detector Channel")
        axes[1].set_ylabel("Projection")
        fig.colorbar(im1, ax=axes[1])

        fig.suptitle(f"Region Error Maps ({patient_id})", fontsize=16, fontweight="bold")
        fig.savefig(os.path.join(figures_dir, f"{patient_id}_region_error.png"))
        plt.close(fig)

        num_proj = error.shape[0]
        global_mae = error.mean(axis=1)
        beam_mae = np.full(num_proj, np.nan)
        background_mae = np.full(num_proj, np.nan)
        for row in range(num_proj):
            row_beam = beam_mask[row]
            if row_beam.any():
                beam_mae[row] = error[row, row_beam].mean()
            if (~row_beam).any():
                background_mae[row] = error[row, ~row_beam].mean()

        fig = plt.figure(figsize=(14, 6))
        projections = np.arange(num_proj)
        plt.plot(projections, global_mae, color="black", linewidth=1.5, label="Global MAE")
        plt.plot(projections, beam_mae, color="red", linewidth=1.5, label="Beam MAE")
        plt.plot(
            projections, background_mae, color="dodgerblue", linewidth=1.5,
            label="Background MAE",
        )
        plt.xlabel("Projection")
        plt.ylabel("MAE")
        plt.title(f"Projection-wise MAE ({patient_id})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"{patient_id}_projection_wise_mae.png"))
        plt.close(fig)

    print(f"Region diagnostic (beam mask / region error / projection-wise MAE) figures saved to {figures_dir}")


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
    parser.add_argument("--no-region-figures", action="store_true",
                         help="Skip saving beam-mask/region-error/projection-wise-MAE PNGs.")
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

    if not args.no_region_figures:
        save_region_diagnostic_figures(reconstructed_results, args.output_dir)

    if not args.no_projection_profiles:
        generate_projection_profile_analysis(save_dir=args.output_dir)


if __name__ == "__main__":
    main()

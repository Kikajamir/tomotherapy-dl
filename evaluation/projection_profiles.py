"""
Detector projection profile analysis -- a purely additive extension of
the evaluation pipeline (see README/CLAUDE.md: preprocessing, model,
training, inference, metrics, and existing comparison/error-map figures
must not change).

Operates entirely on the per-patient `{pid}_planned.npy` /
`{pid}_ground_truth.npy` / `{pid}_prediction.npy` arrays that
`evaluation.evaluate.save_results` already writes to disk -- no
checkpoint is loaded and no inference is run here.
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from configs.config import EVAL_SAVE_DIR

# A projection counts as "active" once its total detector signal exceeds
# this fraction of the patient's peak projection signal -- separates real
# beam-on projections from near-zero background/leakage rows.
ACTIVE_ENERGY_FRACTION = 0.01


def load_patient_arrays(save_dir=EVAL_SAVE_DIR):
    """
    Discovers every patient with saved planned/ground_truth/prediction
    arrays in `save_dir` and returns
    {patient_id: {"planned", "ground_truth", "prediction"}}.
    """
    patients = {}

    for gt_path in sorted(glob.glob(os.path.join(save_dir, "*_ground_truth.npy"))):
        pid = os.path.basename(gt_path)[: -len("_ground_truth.npy")]

        planned_path = os.path.join(save_dir, f"{pid}_planned.npy")
        prediction_path = os.path.join(save_dir, f"{pid}_prediction.npy")

        if not (os.path.exists(planned_path) and os.path.exists(prediction_path)):
            continue

        patients[pid] = {
            "planned": np.load(planned_path),
            "ground_truth": np.load(gt_path),
            "prediction": np.load(prediction_path),
        }

    return patients


def select_projection_indices(ground_truth):
    """
    Identifies the five diagnostic projection indices for one patient's
    ground-truth sinogram: Zero, First Active, Middle (of the active
    region), Maximum Fluence, Last Active.
    """
    projection_energy = ground_truth.sum(axis=1)

    zero_idx = int(np.argmin(projection_energy))
    max_idx = int(np.argmax(projection_energy))

    active_mask = projection_energy > (projection_energy.max() * ACTIVE_ENERGY_FRACTION)
    active_indices = np.flatnonzero(active_mask)

    if active_indices.size == 0:
        first_idx = last_idx = middle_idx = max_idx
    else:
        first_idx = int(active_indices[0])
        last_idx = int(active_indices[-1])
        middle_idx = int((first_idx + last_idx) // 2)

    return {
        "Zero": zero_idx,
        "First Active": first_idx,
        "Middle": middle_idx,
        "Maximum Fluence": max_idx,
        "Last Active": last_idx,
    }


def compute_profile_metrics(gt_row, pred_row):
    """Returns (MAE, RMSE, Pearson) for one projection's detector profile."""
    mae = float(np.mean(np.abs(pred_row - gt_row)))
    rmse = float(np.sqrt(np.mean((pred_row - gt_row) ** 2)))

    if np.std(gt_row) == 0 or np.std(pred_row) == 0:
        pearson = float("nan")
    else:
        pearson, _ = pearsonr(gt_row, pred_row)
        pearson = float(pearson)

    return mae, rmse, pearson


def plot_patient_projection_profiles(patient_id, data, save_path):
    """
    Builds and saves the 5-projection (Profile + Residual, stacked)
    figure for one patient. Returns the per-projection metric rows for
    the CSV.
    """
    gt = data["ground_truth"]
    pred = data["prediction"]

    projections = select_projection_indices(gt)
    channels = np.arange(gt.shape[1])

    fig, axes = plt.subplots(
        2 * len(projections), 1,
        figsize=(14, 3.2 * 2 * len(projections)),
        constrained_layout=True,
    )

    rows = []
    for i, (label, proj_idx) in enumerate(projections.items()):
        gt_row = gt[proj_idx]
        pred_row = pred[proj_idx]
        mae, rmse, pearson = compute_profile_metrics(gt_row, pred_row)

        rows.append({
            "Patient": patient_id,
            "Projection Type": label,
            "Projection Index": proj_idx,
            "MAE": mae,
            "RMSE": rmse,
            "Pearson": pearson,
        })

        profile_ax = axes[2 * i]
        profile_ax.plot(channels, gt_row, color="black", linewidth=2, label="Ground Truth")
        profile_ax.plot(
            channels, pred_row, color="red", linestyle="--", linewidth=2,
            label="Prediction",
        )
        profile_ax.set_title(
            f"Projection {proj_idx} ({label})\n"
            f"MAE = {mae:.4g}   RMSE = {rmse:.4g}   Pearson = {pearson:.4f}",
            fontsize=11,
        )
        profile_ax.set_xlabel("Detector Channel")
        profile_ax.set_ylabel("Detector Intensity")
        profile_ax.set_xlim(0, gt.shape[1] - 1)
        profile_ax.grid(alpha=0.3)
        profile_ax.legend(loc="upper right", fontsize=9)

        residual = pred_row - gt_row

        residual_ax = axes[2 * i + 1]
        residual_ax.plot(channels, residual, color="blue", linewidth=1.5)
        residual_ax.axhline(0, color="black", linestyle="--", linewidth=1)
        residual_ax.set_title("Prediction - Ground Truth", fontsize=10)
        residual_ax.set_xlabel("Detector Channel")
        residual_ax.set_ylabel("Residual")
        residual_ax.set_xlim(0, gt.shape[1] - 1)
        residual_ax.grid(alpha=0.3)

    fig.suptitle(
        f"Patient {patient_id}\nDetector Projection Profile Analysis",
        fontsize=16, fontweight="bold",
    )

    fig.savefig(save_path)
    plt.close(fig)

    return rows


def generate_projection_profile_analysis(save_dir=EVAL_SAVE_DIR):
    """
    Entry point: for every patient with saved planned/ground_truth/
    prediction arrays under `save_dir`, plots the 5-projection profile
    figure to `save_dir/figures/{pid}_projection_profiles.png` and writes
    `save_dir/projection_profile_metrics.csv`. Reuses the already-
    reconstructed arrays only -- runs no inference.
    """
    patients = load_patient_arrays(save_dir)

    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    all_rows = []
    for patient_id, data in patients.items():
        save_path = os.path.join(figures_dir, f"{patient_id}_projection_profiles.png")
        rows = plot_patient_projection_profiles(patient_id, data, save_path)
        all_rows.extend(rows)

    metrics_df = pd.DataFrame(
        all_rows,
        columns=["Patient", "Projection Type", "Projection Index", "MAE", "RMSE", "Pearson"],
    )

    csv_path = os.path.join(save_dir, "projection_profile_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    print(f"Projection profile figures saved to {figures_dir}")
    print(f"Projection profile metrics saved to {csv_path}")

    return metrics_df


if __name__ == "__main__":
    generate_projection_profile_analysis()

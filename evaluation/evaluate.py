"""
Evaluation pipeline extracted verbatim (logic-wise) from
`unetablation.ipynb`: loads the best checkpoint, reconstructs full
sinograms for the test patients via overlapping sliding-window
inference (note: a finer stride than training is used here, matching
the notebook's reconstruction cell), computes per-patient MAE / RMSE /
PSNR / SSIM, saves results, and provides the same diagnostic plots the
notebook produced.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from configs.config import (
    DEVICE,
    RECON_WINDOW_SIZE,
    RECON_STRIDE,
    SAVE_PATH,
    EVAL_SAVE_DIR,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    BASE_CHANNELS,
    NUM_RES_BLOCKS,
)
from model.generator import Generator
from evaluation.projection_profiles import generate_projection_profile_analysis


# =====================================================================
# Load Best Generator
# =====================================================================

def load_best_generator(checkpoint_path=SAVE_PATH, device=DEVICE):
    generator = Generator(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        num_residual_blocks=NUM_RES_BLOCKS,
    ).to(device)

    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))

    generator.eval()

    print("Best residual generator loaded successfully.")

    return generator


# =====================================================================
# Full Sinogram Reconstruction (Residual Learning)
# =====================================================================

def reconstruct_patients(
    generator,
    test_data,
    device=DEVICE,
    window_size=RECON_WINDOW_SIZE,
    stride=RECON_STRIDE,
):
    """
    Reconstructs the full delivered-detector sinogram for every patient
    in `test_data` using overlapping sliding-window inference on the
    residual (detector - planned) target, averaging predictions over
    overlapping windows.

    Returns a dict keyed by patient ID with keys: planned, ground_truth,
    prediction, gantry, projection, position_mm.
    """
    reconstructed_results = {}

    with torch.no_grad():
        for patient_id, patient in tqdm(
            test_data.items(), desc="Reconstructing Test Patients"
        ):
            planned = patient["planned"]
            detector = patient["detector"]

            num_proj, width = planned.shape

            prediction_sum = np.zeros((num_proj, width), dtype=np.float32)
            prediction_count = np.zeros((num_proj, width), dtype=np.float32)

            # Short Patient
            if num_proj < window_size:
                pad_rows = window_size - num_proj

                planned_window = np.pad(
                    planned, ((0, pad_rows), (0, 0)), mode="constant"
                )

                x = (
                    torch.from_numpy(planned_window)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )

                predicted_residual = generator(x)

                predicted_residual = (
                    predicted_residual.squeeze().cpu().numpy()
                )

                predicted_residual = predicted_residual[:num_proj]

                prediction = planned + predicted_residual

                prediction_sum += prediction
                prediction_count += 1

            # Normal Patient
            else:
                starts = list(
                    range(0, num_proj - window_size + 1, stride)
                )

                last_start = num_proj - window_size

                if starts[-1] != last_start:
                    starts.append(last_start)

                for start in starts:
                    planned_window = planned[start:start + window_size]

                    x = (
                        torch.from_numpy(planned_window)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )

                    predicted_residual = generator(x)

                    predicted_residual = (
                        predicted_residual.squeeze().cpu().numpy()
                    )

                    prediction = planned_window + predicted_residual

                    prediction_sum[start:start + window_size] += prediction
                    prediction_count[start:start + window_size] += 1

            # Average overlapping predictions
            prediction = prediction_sum / prediction_count

            reconstructed_results[patient_id] = {
                "planned": planned,
                "ground_truth": detector,
                "prediction": prediction,
                "gantry": patient["gantry"],
                "projection": patient["projection"],
                "position_mm": patient["position_mm"],
            }

    print(f"Patients reconstructed : {len(reconstructed_results)}")

    return reconstructed_results


# =====================================================================
# Evaluation Metrics
# =====================================================================

def compute_metrics(reconstructed_results):
    """
    Computes per-patient MAE, RMSE, PSNR, SSIM, plus a Mean/Std summary
    across patients. Returns (metrics_df, summary_df).
    """
    metrics_list = []

    for pid, data in reconstructed_results.items():
        gt = data["ground_truth"]
        pred = data["prediction"]

        mae = mean_absolute_error(gt.flatten(), pred.flatten())

        rmse = np.sqrt(mean_squared_error(gt.flatten(), pred.flatten()))

        data_range = max(gt.max() - gt.min(), 1e-8)

        psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)

        ssim = structural_similarity(gt, pred, data_range=data_range)

        metrics_list.append({
            "Patient": pid,
            "MAE": mae,
            "RMSE": rmse,
            "PSNR": psnr,
            "SSIM": ssim,
        })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values("Patient").reset_index(drop=True)

    summary = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "PSNR", "SSIM"],
        "Mean": [
            metrics_df["MAE"].mean(),
            metrics_df["RMSE"].mean(),
            metrics_df["PSNR"].mean(),
            metrics_df["SSIM"].mean(),
        ],
        "Std": [
            metrics_df["MAE"].std(),
            metrics_df["RMSE"].std(),
            metrics_df["PSNR"].std(),
            metrics_df["SSIM"].std(),
        ],
    })

    return metrics_df, summary


# =====================================================================
# Save Results
# =====================================================================

def save_results(reconstructed_results, metrics_df, summary, save_dir=EVAL_SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    for pid, data in reconstructed_results.items():
        np.save(
            os.path.join(save_dir, f"{pid}_prediction.npy"), data["prediction"]
        )
        np.save(
            os.path.join(save_dir, f"{pid}_ground_truth.npy"), data["ground_truth"]
        )
        np.save(
            os.path.join(save_dir, f"{pid}_planned.npy"), data["planned"]
        )

    metrics_df.to_csv(os.path.join(save_dir, "patient_metrics.csv"), index=False)
    summary.to_csv(os.path.join(save_dir, "overall_metrics.csv"), index=False)

    print("Evaluation Results Saved")
    print(f"Directory : {save_dir}")


# =====================================================================
# Diagnostic Plots (all optional, matplotlib required)
# =====================================================================

def plot_sinogram_comparison(reconstructed_results, patient_id):
    import matplotlib.pyplot as plt

    data = reconstructed_results[patient_id]

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
        im = ax.imshow(
            img, cmap="gray", aspect="auto", origin="lower", vmin=vmin, vmax=vmax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Detector Channel")
        ax.set_ylabel("Projection")

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Detector Signal")

    plt.suptitle(f"Patient : {patient_id}", fontsize=18, fontweight="bold")
    plt.show()


def plot_absolute_error_map(reconstructed_results, patient_id):
    import matplotlib.pyplot as plt

    data = reconstructed_results[patient_id]
    ground_truth = data["ground_truth"]
    prediction = data["prediction"]

    error = np.abs(ground_truth - prediction)

    plt.figure(figsize=(7, 8))

    im = plt.imshow(error, cmap="hot", aspect="auto", origin="lower")

    plt.title(f"Absolute Error Map ({patient_id})", fontsize=16, fontweight="bold")
    plt.xlabel("Detector Channel")
    plt.ylabel("Projection")

    cbar = plt.colorbar(im)
    cbar.set_label("Absolute Error")

    plt.tight_layout()
    plt.show()


def plot_projection_profile_comparison(reconstructed_results, patient_id):
    import matplotlib.pyplot as plt

    data = reconstructed_results[patient_id]
    gt = data["ground_truth"]
    pred = data["prediction"]

    num_proj = gt.shape[0]

    projection_indices = [
        0,
        num_proj // 4,
        num_proj // 2,
        (3 * num_proj) // 4,
        num_proj - 1,
    ]

    fig, axes = plt.subplots(
        len(projection_indices),
        2,
        figsize=(16, 3.8 * len(projection_indices)),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )

    for row, proj in enumerate(projection_indices):
        detector = np.arange(gt.shape[1])

        # Signal comparison
        axes[row, 0].plot(
            detector, gt[proj], color="black", linewidth=2, label="Ground Truth"
        )
        axes[row, 0].plot(
            detector, pred[proj], color="red", linestyle="--", linewidth=2,
            label="Prediction",
        )
        axes[row, 0].set_title(f"Projection {proj}", fontsize=12)
        axes[row, 0].set_ylabel("Detector Signal")
        axes[row, 0].grid(alpha=0.3)

        if row == 0:
            axes[row, 0].legend()

        # Residual
        residual = pred[proj] - gt[proj]

        axes[row, 1].plot(detector, residual, color="blue", linewidth=1.5)
        axes[row, 1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[row, 1].set_title("Residual")
        axes[row, 1].grid(alpha=0.3)

    axes[-1, 0].set_xlabel("Detector Channel")
    axes[-1, 1].set_xlabel("Detector Channel")

    plt.suptitle(
        f"Projection Profile Comparison ({patient_id})",
        fontsize=18,
        fontweight="bold",
    )
    plt.show()


def plot_peak_fluence_projection_profiles(reconstructed_results, patient_id):
    import matplotlib.pyplot as plt

    data = reconstructed_results[patient_id]
    gt = data["ground_truth"]
    pred = data["prediction"]

    # Find projections with largest detector signal
    projection_energy = gt.sum(axis=1)

    top_idx = np.argsort(projection_energy)[-5:]
    top_idx = np.sort(top_idx)

    print("Selected projections:", top_idx)

    fig, axes = plt.subplots(
        len(top_idx), 1, figsize=(14, 3.8 * len(top_idx)), constrained_layout=True
    )

    if len(top_idx) == 1:
        axes = [axes]

    for ax, proj in zip(axes, top_idx):
        detector = np.arange(gt.shape[1])

        ax.plot(detector, gt[proj], color="black", linewidth=2, label="Ground Truth")
        ax.plot(
            detector, pred[proj], color="red", linestyle="--", linewidth=2,
            label="Prediction",
        )
        ax.fill_between(
            detector, gt[proj], pred[proj], color="dodgerblue", alpha=0.25,
            label="Residual",
        )

        ax.set_title(f"Projection {proj}")
        ax.set_ylabel("Detector Signal")
        ax.grid(alpha=0.3)

    axes[0].legend()
    axes[-1].set_xlabel("Detector Channel")

    plt.suptitle(
        f"Peak Fluence Projection Profiles ({patient_id})",
        fontsize=18,
        fontweight="bold",
    )
    plt.show()


def plot_detector_channel_profiles(reconstructed_results, patient_id):
    import matplotlib.pyplot as plt

    data = reconstructed_results[patient_id]
    gt = data["ground_truth"]
    pred = data["prediction"]

    num_channels = gt.shape[1]

    channels = [
        num_channels // 4,
        num_channels // 2,
        (3 * num_channels) // 4,
    ]

    fig, axes = plt.subplots(
        len(channels), 1, figsize=(14, 4 * len(channels)), constrained_layout=True
    )

    if len(channels) == 1:
        axes = [axes]

    projection = np.arange(gt.shape[0])

    for ax, ch in zip(axes, channels):
        ax.plot(
            projection, gt[:, ch], color="black", linewidth=2, label="Ground Truth"
        )
        ax.plot(
            projection, pred[:, ch], color="red", linestyle="--", linewidth=2,
            label="Prediction",
        )
        ax.fill_between(
            projection, gt[:, ch], pred[:, ch], color="dodgerblue", alpha=0.25
        )

        ax.set_title(f"Detector Channel {ch}")
        ax.set_xlabel("Projection")
        ax.set_ylabel("Signal")
        ax.grid(alpha=0.3)

    axes[0].legend()

    plt.suptitle(
        f"Detector Channel Profiles ({patient_id})", fontsize=18, fontweight="bold"
    )
    plt.show()


def plot_error_histogram(reconstructed_results, patient_id):
    import matplotlib.pyplot as plt

    data = reconstructed_results[patient_id]
    ground_truth = data["ground_truth"]
    prediction = data["prediction"]

    error = prediction - ground_truth

    plt.figure(figsize=(8, 5))

    plt.hist(error.ravel(), bins=200, density=True, color="royalblue", alpha=0.85)

    plt.axvline(0, color="black", linestyle="--", linewidth=2)

    plt.xlabel("Prediction - Ground Truth")
    plt.ylabel("Density")
    plt.title(
        f"Pixel-wise Error Distribution ({patient_id})", fontsize=15,
        fontweight="bold",
    )
    plt.grid(alpha=0.3)
    plt.show()


# =====================================================================
# Entry point
# =====================================================================

def evaluate(test_data, checkpoint_path=SAVE_PATH, save_dir=EVAL_SAVE_DIR):
    """
    Runs the full evaluation pipeline: load checkpoint, reconstruct test
    patients, compute metrics, save results.

    Returns (reconstructed_results, metrics_df, summary).
    """
    generator = load_best_generator(checkpoint_path)

    reconstructed_results = reconstruct_patients(generator, test_data)

    metrics_df, summary = compute_metrics(reconstructed_results)

    print(metrics_df)
    print(summary)

    save_results(reconstructed_results, metrics_df, summary, save_dir)

    generate_projection_profile_analysis(save_dir=save_dir)

    return reconstructed_results, metrics_df, summary


if __name__ == "__main__":
    import argparse

    from training.train import load_and_split_patients

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=SAVE_PATH,
                         help=f"Path to a generator checkpoint to evaluate "
                              f"(default: {SAVE_PATH!r}).")
    args = parser.parse_args()

    _, _, test_data = load_and_split_patients()
    evaluate(test_data, checkpoint_path=args.checkpoint)

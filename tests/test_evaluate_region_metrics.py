"""
Sanity checks for the beam/background region metrics added to
`evaluation.evaluate.compute_metrics` -- uses a small synthetic
`reconstructed_results` dict (no real patient data, no trained
checkpoint/generator needed, since `compute_metrics` only consumes
already-reconstructed arrays).
"""

import numpy as np
import pytest

from evaluation.evaluate import compute_metrics


def _synthetic_result():
    # SSIM's default win_size (7) requires both dimensions >= 7, so this
    # uses an 8x8 image (left half background, right half beam).
    rng = np.random.default_rng(0)

    planned = np.zeros((8, 8), dtype=np.float32)
    planned[:, 4:] = 1.0

    background_gt = rng.uniform(0.0, 0.05, size=(8, 4)).astype(np.float32)
    beam_gt = rng.uniform(0.5, 0.9, size=(8, 4)).astype(np.float32)
    ground_truth = np.hstack([background_gt, beam_gt])

    # Background prediction is close to ground truth; beam prediction is
    # deliberately off by ~0.2 (underestimated peak), matching the
    # "beam error >> background error" pattern this ablation targets.
    background_pred = background_gt + 0.01
    beam_pred = beam_gt - 0.2
    prediction = np.hstack([background_pred, beam_pred]).astype(np.float32)

    return {
        "P1": {
            "planned": planned,
            "ground_truth": ground_truth,
            "prediction": prediction,
        }
    }


def test_compute_metrics_includes_beam_and_background_columns():
    metrics_df, summary = compute_metrics(_synthetic_result())

    for col in ["MAE", "RMSE", "PSNR", "SSIM",
                "Beam_MAE", "Beam_RMSE", "Background_MAE", "Background_RMSE"]:
        assert col in metrics_df.columns
        assert col in summary["Metric"].values


def test_compute_metrics_beam_background_mae_match_manual_split():
    reconstructed_results = _synthetic_result()
    metrics_df, _ = compute_metrics(reconstructed_results)

    data = reconstructed_results["P1"]
    planned = data["planned"]
    gt = data["ground_truth"]
    pred = data["prediction"]

    beam_mask = planned > 0.0
    background_mask = ~beam_mask

    expected_beam_mae = np.abs(gt[beam_mask] - pred[beam_mask]).mean()
    expected_background_mae = np.abs(gt[background_mask] - pred[background_mask]).mean()

    row = metrics_df.loc[metrics_df["Patient"] == "P1"].iloc[0]

    assert row["Beam_MAE"] == pytest.approx(expected_beam_mae, abs=1e-6)
    assert row["Background_MAE"] == pytest.approx(expected_background_mae, abs=1e-6)


def test_compute_metrics_beam_mae_is_worse_than_background_mae_here():
    # By construction the prediction is off by ~0.2 in the beam region
    # and only ~0.01 in the background -- beam error should dominate.
    metrics_df, _ = compute_metrics(_synthetic_result())
    row = metrics_df.iloc[0]
    assert row["Beam_MAE"] > row["Background_MAE"]


def test_compute_metrics_handles_all_background_patient():
    # Degenerate case: planned is all zero (no beam at all) -- Beam_MAE
    # should be NaN, not a crash, and Background_MAE should still equal
    # the (global) MAE.
    rng = np.random.default_rng(1)
    ground_truth = rng.uniform(0.0, 0.1, size=(8, 8)).astype(np.float32)
    reconstructed_results = {
        "P2": {
            "planned": np.zeros((8, 8), dtype=np.float32),
            "ground_truth": ground_truth,
            "prediction": (ground_truth + 0.02).astype(np.float32),
        }
    }

    metrics_df, _ = compute_metrics(reconstructed_results)
    row = metrics_df.iloc[0]

    assert np.isnan(row["Beam_MAE"])
    assert row["Background_MAE"] == pytest.approx(row["MAE"], abs=1e-6)

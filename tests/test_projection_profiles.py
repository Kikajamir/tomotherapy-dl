"""
Sanity checks for `evaluation.projection_profiles` -- the detector
projection profile analysis. Uses small synthetic planned/ground_truth/
prediction arrays under pytest's tmp_path (mimicking the .npy layout
`evaluation.evaluate.save_results` writes); no real patient data, no
checkpoint, no inference.
"""

import os

import numpy as np
import pytest

from evaluation.projection_profiles import (
    compute_profile_metrics,
    generate_projection_profile_analysis,
    load_patient_arrays,
    select_projection_indices,
)


def _synthetic_sinogram(num_proj=20, width=10):
    # Rows 0-2 and 17-19 are zero background; rows 3-16 are the active
    # treatment region, with row 10 carrying the largest signal.
    ground_truth = np.zeros((num_proj, width), dtype=np.float32)
    ground_truth[3:17] = 1.0
    ground_truth[10] = 5.0
    return ground_truth


def test_select_projection_indices_identifies_expected_rows():
    gt = _synthetic_sinogram()

    indices = select_projection_indices(gt)

    assert indices["Zero"] in range(0, 3)
    assert indices["First Active"] == 3
    assert indices["Last Active"] == 16
    assert indices["Maximum Fluence"] == 10
    assert indices["First Active"] <= indices["Middle"] <= indices["Last Active"]


def test_compute_profile_metrics_zero_for_identical_rows():
    row = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    mae, rmse, pearson = compute_profile_metrics(row, row.copy())

    assert mae == 0.0
    assert rmse == 0.0
    assert pearson == pytest.approx(1.0)


def test_compute_profile_metrics_nan_pearson_for_constant_row():
    gt_row = np.zeros(4, dtype=np.float32)
    pred_row = np.array([0.1, 0.0, -0.1, 0.05], dtype=np.float32)

    mae, rmse, pearson = compute_profile_metrics(gt_row, pred_row)

    assert mae > 0.0
    assert np.isnan(pearson)


def _write_patient_arrays(save_dir, patient_id, gt):
    np.save(os.path.join(save_dir, f"{patient_id}_planned.npy"), np.zeros_like(gt))
    np.save(os.path.join(save_dir, f"{patient_id}_ground_truth.npy"), gt)
    np.save(os.path.join(save_dir, f"{patient_id}_prediction.npy"), gt + 0.1)


def test_load_patient_arrays_discovers_only_complete_triplets(tmp_path):
    gt = _synthetic_sinogram()
    _write_patient_arrays(tmp_path, "P1", gt)

    # P2 is missing its prediction array -- must be skipped, not crash.
    np.save(tmp_path / "P2_planned.npy", np.zeros_like(gt))
    np.save(tmp_path / "P2_ground_truth.npy", gt)

    patients = load_patient_arrays(str(tmp_path))

    assert set(patients.keys()) == {"P1"}


def test_generate_projection_profile_analysis_end_to_end(tmp_path):
    gt = _synthetic_sinogram()
    _write_patient_arrays(tmp_path, "P1", gt)

    metrics_df = generate_projection_profile_analysis(save_dir=str(tmp_path))

    assert list(metrics_df.columns) == [
        "Patient", "Projection Type", "Projection Index", "MAE", "RMSE", "Pearson",
    ]
    assert len(metrics_df) == 5  # 5 projection types for the one patient
    assert set(metrics_df["Projection Type"]) == {
        "Zero", "First Active", "Middle", "Maximum Fluence", "Last Active",
    }

    assert os.path.exists(os.path.join(tmp_path, "projection_profile_metrics.csv"))
    assert os.path.exists(os.path.join(tmp_path, "figures", "P1_projection_profiles.png"))

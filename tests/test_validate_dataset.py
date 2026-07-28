"""
Sanity checks for `preprocessing.validate_dataset`, using small synthetic
arrays -- no real patient data.
"""

import pickle

import numpy as np

from preprocessing.validate_dataset import (
    validate_sample,
    validate_processed_data,
    validate_pkl_file,
    summarize_validation,
)


def test_validate_sample_passes_for_clean_data():
    data = {
        "planned": np.random.rand(10, 5).astype(np.float32),
        "detector": np.random.rand(10, 5).astype(np.float32),
    }
    result = validate_sample("P1", data)

    assert result["passed"]
    assert result["issues"] == []
    assert result["shape"] == (10, 5)


def test_validate_sample_flags_shape_mismatch():
    data = {"planned": np.zeros((10, 5)), "detector": np.zeros((10, 6))}
    result = validate_sample("P1", data)

    assert not result["passed"]
    assert any("shape mismatch" in issue for issue in result["issues"])


def test_validate_sample_flags_nan_and_inf():
    planned = np.zeros((4, 4))
    planned[0, 0] = np.nan
    planned[1, 1] = np.inf

    data = {"planned": planned, "detector": np.zeros((4, 4))}
    result = validate_sample("P1", data)

    assert not result["passed"]
    assert result["planned_nan"] == 1
    assert result["planned_inf"] == 1
    assert result["detector_nan"] == 0
    assert result["detector_inf"] == 0


def test_validate_sample_stats_computed_over_finite_values_only():
    planned = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.inf]])
    data = {"planned": planned, "detector": planned.copy()}

    result = validate_sample("P1", data)

    assert result["planned_stats"]["min"] == 1.0
    assert result["planned_stats"]["max"] == 4.0


def test_validate_processed_data_and_summary():
    good = {"planned": np.ones((3, 3)), "detector": np.ones((3, 3))}
    bad = {"planned": np.array([[np.nan]]), "detector": np.array([[np.nan]])}
    processed = {"P1": good, "P2": bad}

    report = validate_processed_data(processed)
    summary = summarize_validation(report)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["failures"][0]["patient"] == "P2"


def test_validate_pkl_file(tmp_path):
    processed = {"P1": {"planned": np.ones((3, 3)), "detector": np.ones((3, 3))}}
    pkl_path = tmp_path / "data.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(processed, f)

    report = validate_pkl_file(str(pkl_path))

    assert len(report) == 1
    assert report[0]["passed"]

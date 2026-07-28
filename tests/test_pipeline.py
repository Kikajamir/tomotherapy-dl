"""
Sanity checks for `preprocessing.pipeline` (auto-discovery + preprocessing
orchestration), using small synthetic planned/detector CSVs under
pytest's tmp_path -- no real patient data, no RTPLAN involved.

Synthetic CSVs use exactly 64 planned-signal columns and 640
detector-signal columns to match `build_coordinate_system`'s fixed
leaf/detector grids (`preprocessing.build_dataset.preprocess_patient`
interpolates from those exact widths).
"""

import os

import numpy as np
import pandas as pd

from preprocessing.pipeline import discover_patient_files, run_preprocessing


def _write_patient_csvs(patient_dir, num_rows=5, planned_name="planned.csv", detector_name="D1.csv"):
    os.makedirs(patient_dir, exist_ok=True)

    meta_cols = ["Gantry", "Proj", "Pos"]
    rng = np.random.default_rng(0)

    planned_df = pd.DataFrame(
        np.hstack([np.zeros((num_rows, 3)), rng.random((num_rows, 64))]),
        columns=meta_cols + [str(i) for i in range(64)],
    )
    detector_df = pd.DataFrame(
        np.hstack([np.zeros((num_rows, 3)), rng.random((num_rows, 640))]),
        columns=meta_cols + [str(i) for i in range(640)],
    )

    planned_df.to_csv(os.path.join(patient_dir, planned_name), index=False)
    detector_df.to_csv(os.path.join(patient_dir, detector_name), index=False)


def test_discover_patient_files_case_insensitive_planned_name(tmp_path):
    _write_patient_csvs(tmp_path / "P1", planned_name="planned.csv")
    _write_patient_csvs(tmp_path / "P2", planned_name="PLANNED.csv")

    patients, skipped = discover_patient_files(str(tmp_path))

    assert {p["patient"] for p in patients} == {"P1", "P2"}
    assert skipped == {}


def test_discover_patient_files_skips_folders_missing_csvs(tmp_path):
    _write_patient_csvs(tmp_path / "P1")
    os.makedirs(tmp_path / "P2")  # no CSVs at all

    patients, skipped = discover_patient_files(str(tmp_path))

    assert {p["patient"] for p in patients} == {"P1"}
    assert "planned.csv" in skipped["P2"]
    assert "D1.csv" in skipped["P2"]


def test_discover_patient_files_recovers_patient_missing_only_rtplan(tmp_path):
    # Mirrors the real P37 case: filed under Incomplete/ purely for a
    # missing RTPLAN, but has valid planned/detector CSVs -- this task
    # doesn't care about RTPLAN, so it must still be discovered.
    _write_patient_csvs(tmp_path / "Incomplete" / "P37")

    patients, skipped = discover_patient_files(str(tmp_path))

    assert {p["patient"] for p in patients} == {"P37"}
    assert skipped == {}


def test_run_preprocessing_end_to_end(tmp_path):
    _write_patient_csvs(tmp_path / "P1")
    _write_patient_csvs(tmp_path / "P2")

    output_pkl = tmp_path / "processed_data.pkl"

    processed_data, skipped, failed = run_preprocessing(
        dataset_root=str(tmp_path), save_path=str(output_pkl)
    )

    assert set(processed_data.keys()) == {"P1", "P2"}
    assert skipped == {}
    assert failed == {}
    assert output_pkl.exists()

    for data in processed_data.values():
        assert data["planned"].shape == data["detector"].shape
        assert data["planned"].shape[1] == 542  # documented common-coordinate output width


def test_run_preprocessing_continues_past_a_missing_patient(tmp_path):
    _write_patient_csvs(tmp_path / "P1")
    os.makedirs(tmp_path / "P2")  # missing CSVs -> skipped, not failed

    processed_data, skipped, failed = run_preprocessing(
        dataset_root=str(tmp_path), save_path=str(tmp_path / "out.pkl")
    )

    assert set(processed_data.keys()) == {"P1"}
    assert "P2" in skipped
    assert failed == {}

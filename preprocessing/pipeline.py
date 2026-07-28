"""
Auto-discovery + preprocessing orchestration for the local baseline
pipeline (raw dataset -> `processed_data.pkl`).

This module does not implement or alter any preprocessing math. It only
replaces `build_dataset.find_patients`'s "P1..P{NUM_PATIENTS}, contiguous
numbering" discovery with genuine directory-listing discovery (via
`audit_dataset.discover_patient_dirs`), then hands the discovered files to
`build_dataset`'s existing, unmodified functions
(`build_coordinate_system`, `load_raw_data`, `preprocess_patient`) exactly
as they already work. RTPLAN is never read here.

A patient only needs `planned.csv`/`PLANNED.csv` and `D1.csv` to be usable
-- the earlier `Complete/`/`Incomplete/` audit split additionally required
an RTPLAN file, which this pipeline does not need, so a patient filed
under `Incomplete/` for a missing RTPLAN (e.g. `P37`) is still included
here if its planned/detector CSVs are present.
"""

import pickle

from preprocessing.audit_dataset import default_dataset_root, discover_patient_dirs
from preprocessing.build_dataset import (
    build_coordinate_system,
    find_file_case_insensitive,
    load_raw_data,
    preprocess_patient,
)
from configs.config import SAVE_FILE

PLANNED_FILENAME = "planned.csv"
DETECTOR_FILENAME = "D1.csv"


def discover_patient_files(dataset_root=None):
    """
    Returns (patients, skipped).

    patients : list of {"patient", "planned", "detector"} dicts, the same
               shape `build_dataset.find_patients` produces -- a drop-in
               input to `build_dataset.load_raw_data`.
    skipped  : {patient_id: reason} for every discovered folder missing
               one or both required files.
    """
    dataset_root = dataset_root or default_dataset_root()
    patient_dirs = discover_patient_dirs(dataset_root)

    patients = []
    skipped = {}

    for patient_id in sorted(patient_dirs):
        patient_dir = patient_dirs[patient_id]

        planned_path = find_file_case_insensitive(patient_dir, PLANNED_FILENAME)
        detector_path = find_file_case_insensitive(patient_dir, DETECTOR_FILENAME)

        if planned_path and detector_path:
            patients.append({
                "patient": patient_id,
                "planned": planned_path,
                "detector": detector_path,
            })
        else:
            missing = []
            if not planned_path:
                missing.append(PLANNED_FILENAME)
            if not detector_path:
                missing.append(DETECTOR_FILENAME)
            skipped[patient_id] = f"missing {', '.join(missing)}"

    return patients, skipped


def run_preprocessing(dataset_root=None, save_path=SAVE_FILE):
    """
    Discovers every usable patient, runs the unmodified preprocessing
    pipeline on each, and writes `save_path` in the same pickle format
    `build_dataset.build_dataset` already uses.

    Returns (processed_data, skipped, failed):
      processed_data : {patient_id: {"planned","detector","gantry",
                        "projection","position_mm"}}, ready for
                        SlidingWindowDataset / training.
      skipped        : patients excluded before preprocessing (missing
                        planned.csv or D1.csv).
      failed         : patients whose preprocessing raised an exception,
                        {patient_id: error message}. Preprocessing keeps
                        going for the remaining patients.
    """
    patients, skipped = discover_patient_files(dataset_root)

    print(f"Discovered {len(patients) + len(skipped)} patient folder(s).")
    print(f"Usable (planned + D1 present): {len(patients)}")

    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for patient_id, reason in sorted(skipped.items()):
            print(f"  {patient_id}: {reason}")

    leaf_mm, detector_iso_mm, common_mm = build_coordinate_system()

    all_data = load_raw_data(patients)

    processed_data = {}
    failed = {}

    for patient_id in sorted(all_data):
        try:
            planned = all_data[patient_id]["planned"]
            detector = all_data[patient_id]["detector"]

            plan_common, det_common = preprocess_patient(
                planned, detector, leaf_mm, detector_iso_mm, common_mm
            )

            processed_data[patient_id] = {
                "planned": plan_common,
                "detector": det_common,
                "gantry": all_data[patient_id]["gantry"],
                "projection": all_data[patient_id]["projection"],
                "position_mm": all_data[patient_id]["position_mm"],
            }
        except Exception as exc:  # noqa: BLE001 -- one bad patient must not abort the run
            failed[patient_id] = str(exc)

    if failed:
        print(f"Preprocessing failed for {len(failed)} patient(s):")
        for patient_id, reason in sorted(failed.items()):
            print(f"  {patient_id}: {reason}")

    print(f"Preprocessed {len(processed_data)} patient(s) successfully.")

    with open(save_path, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"Saved to {save_path}")

    return processed_data, skipped, failed

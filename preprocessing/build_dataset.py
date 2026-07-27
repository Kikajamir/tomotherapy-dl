"""
Preprocessing pipeline extracted from `pix2pix.ipynb`.

Turns the raw per-patient TomoTherapy CSV exports (planned fluence +
detector exit signal) into a single `processed_data.pkl` file containing,
per patient, the planned and detector sinograms resampled onto a shared
spatial coordinate system.

The algorithm is unchanged from the notebook:
  1. Find each patient's `planned.csv` / `D1.csv` (case-insensitive).
  2. Load them, keeping only the signal columns (column index 3 onward).
  3. Flip the detector image horizontally.
  4. Interpolate the planned fluence onto the leaf coordinate grid
     ("nearest") and the detector signal onto the detector coordinate
     grid ("linear"), both resampled onto a shared `common_mm` axis.
  5. Shift the detector image by -DETECTOR_SHIFT to align it with the
     planned fluence.
  6. Normalize the detector image to [0, 1] (planned is left unnormalized,
     matching NORMALIZE_PLANNED = False).
  7. Save the result as a dict keyed by patient ID to `processed_data.pkl`.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from configs.config import (
    BASE_PATH,
    SAVE_FILE,
    NUM_PATIENTS,
    LEAF_PITCH,
    PITCH_ISO,
    DETECTOR_SHIFT,
    INTERP_PLAN,
    INTERP_DETECTOR,
    NORMALIZE_PLANNED,
    NORMALIZE_DETECTOR,
)


# =====================================================================
# Coordinate system
# =====================================================================

def build_coordinate_system():
    """
    Recreates the leaf / detector / common coordinate axes exactly as
    computed in the notebook.
    """
    leaf_mm = (np.arange(64) - 31.5) * LEAF_PITCH

    detector_iso_mm = (np.arange(640) - 319.5) * PITCH_ISO

    common_mm = np.arange(
        max(leaf_mm.min(), detector_iso_mm.min()),
        min(leaf_mm.max(), detector_iso_mm.max()),
        PITCH_ISO,
    )

    return leaf_mm, detector_iso_mm, common_mm


# =====================================================================
# File discovery
# =====================================================================

def find_file_case_insensitive(folder, filename):
    filename = filename.lower()

    for f in os.listdir(folder):
        if f.lower() == filename:
            return os.path.join(folder, f)

    return None


def find_patients(base_path=BASE_PATH, num_patients=NUM_PATIENTS):
    """
    Scans `base_path` for P1..P{num_patients} folders containing both a
    planned.csv and a D1.csv file.

    Returns (patients, missing) where `patients` is a list of dicts with
    keys "patient", "planned", "detector", and `missing` is a list of
    patient IDs that were not found.
    """
    patients = []
    missing = []

    for i in range(1, num_patients + 1):
        patient = f"P{i}"
        patient_dir = os.path.join(base_path, patient)

        planned = find_file_case_insensitive(patient_dir, "planned.csv")
        detector = find_file_case_insensitive(patient_dir, "D1.csv")

        if planned and detector:
            patients.append({
                "patient": patient,
                "planned": planned,
                "detector": detector,
            })
        else:
            missing.append(patient)

    return patients, missing


def load_raw_data(patients):
    """
    Loads the planned/detector CSVs for each patient into memory.

    Returns a dict keyed by patient ID with keys: planned, detector,
    gantry, projection, position_mm (all as in the notebook).
    """
    all_data = {}

    for patient in patients:
        pid = patient["patient"]

        plan_df = pd.read_csv(patient["planned"])
        det_df = pd.read_csv(patient["detector"])

        planned = plan_df.iloc[:, 3:].to_numpy(dtype=np.float32)
        detector = det_df.iloc[:, 3:].to_numpy(dtype=np.float32)

        all_data[pid] = {
            "planned": planned,
            "detector": detector,
            "gantry": plan_df.iloc[:, 0].to_numpy(),
            "projection": plan_df.iloc[:, 1].to_numpy(),
            "position_mm": plan_df.iloc[:, 2].to_numpy(),
        }

    return all_data


# =====================================================================
# Per-patient preprocessing
# =====================================================================

def shift_image(img, shift):
    """
    Shift image horizontally with zero padding.

    shift > 0 : shift right
    shift < 0 : shift left
    """
    shifted = np.zeros_like(img)

    if shift == 0:
        return img.copy()
    elif shift > 0:
        shifted[:, shift:] = img[:, :-shift]
    else:
        shift = abs(shift)
        shifted[:, :-shift] = img[:, shift:]

    return shifted


def normalize(img):
    """Normalize image to [0,1]."""
    img = img.astype(np.float32)
    maximum = img.max()

    if maximum > 0:
        img /= maximum

    return img


def preprocess_patient(planned, detector, leaf_mm, detector_iso_mm, common_mm):
    """
    Preprocess one patient.

    Steps
    -----
    1. Flip detector
    2. Interpolate planned
    3. Interpolate detector
    4. Shift detector
    5. Normalize (detector only, per NORMALIZE_PLANNED/NORMALIZE_DETECTOR)
    """
    # Detector flip
    detector = detector[:, ::-1]

    # Planned interpolation
    planned_interp = interp1d(
        leaf_mm,
        planned,
        axis=1,
        kind=INTERP_PLAN,
        bounds_error=False,
        fill_value=0,
    )
    planned_common = planned_interp(common_mm)

    # Detector interpolation
    detector_interp = interp1d(
        detector_iso_mm,
        detector,
        axis=1,
        kind=INTERP_DETECTOR,
        bounds_error=False,
        fill_value=0,
    )
    detector_common = detector_interp(common_mm)

    # Detector alignment
    detector_common = shift_image(detector_common, -DETECTOR_SHIFT)

    # Normalize
    if NORMALIZE_PLANNED:
        planned_common = normalize(planned_common)

    if NORMALIZE_DETECTOR:
        detector_common = normalize(detector_common)

    return planned_common, detector_common


# =====================================================================
# Full pipeline
# =====================================================================

def build_dataset(base_path=BASE_PATH, save_path=SAVE_FILE, num_patients=NUM_PATIENTS):
    """
    Runs the full preprocessing pipeline and writes `processed_data.pkl`.

    Returns the `processed_data` dict (patient_id -> dict with keys
    "planned", "detector", "gantry", "projection", "position_mm").
    """
    leaf_mm, detector_iso_mm, common_mm = build_coordinate_system()

    patients, missing = find_patients(base_path, num_patients)
    print(f"Patients found : {len(patients)}")
    if missing:
        print("Missing :", missing)

    all_data = load_raw_data(patients)
    print(f"Loaded {len(all_data)} patients.")

    processed_data = {}

    for patient in sorted(all_data):
        planned = all_data[patient]["planned"]
        detector = all_data[patient]["detector"]

        plan_common, det_common = preprocess_patient(
            planned, detector, leaf_mm, detector_iso_mm, common_mm
        )

        processed_data[patient] = {
            "planned": plan_common,
            "detector": det_common,
            "gantry": all_data[patient]["gantry"],
            "projection": all_data[patient]["projection"],
            "position_mm": all_data[patient]["position_mm"],
        }

    print("Finished preprocessing.")

    with open(save_path, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"Saved to {save_path}")

    return processed_data


if __name__ == "__main__":
    build_dataset()

"""
Read-only validation for a `processed_data.pkl` produced by
`preprocessing.pipeline.run_preprocessing` (or the original
`preprocessing.build_dataset.build_dataset`) -- never modifies the pkl or
recomputes anything preprocessing already did.

For every patient sample, checks: matching planned/detector shapes, NaN,
Inf, and basic summary statistics (min/max/mean/std) for both arrays.
"""

import pickle

import numpy as np


def validate_sample(patient_id, data):
    """Validates one patient's {"planned", "detector"} pair. Returns a dict report."""
    planned = data["planned"]
    detector = data["detector"]

    issues = []

    if planned.shape != detector.shape:
        issues.append(
            f"shape mismatch: planned{planned.shape} vs detector{detector.shape}"
        )

    planned_nan = int(np.isnan(planned).sum())
    planned_inf = int(np.isinf(planned).sum())
    detector_nan = int(np.isnan(detector).sum())
    detector_inf = int(np.isinf(detector).sum())

    if planned_nan:
        issues.append(f"planned has {planned_nan} NaN value(s)")
    if planned_inf:
        issues.append(f"planned has {planned_inf} Inf value(s)")
    if detector_nan:
        issues.append(f"detector has {detector_nan} NaN value(s)")
    if detector_inf:
        issues.append(f"detector has {detector_inf} Inf value(s)")

    def stats(array):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return {"min": None, "max": None, "mean": None, "std": None}
        return {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        }

    return {
        "patient": patient_id,
        "shape": tuple(planned.shape),
        "planned_stats": stats(planned),
        "detector_stats": stats(detector),
        "planned_nan": planned_nan,
        "planned_inf": planned_inf,
        "detector_nan": detector_nan,
        "detector_inf": detector_inf,
        "passed": len(issues) == 0,
        "issues": issues,
    }


def validate_processed_data(processed_data):
    """Validates every patient in a {patient_id: {"planned","detector",...}} dict."""
    return [
        validate_sample(patient_id, data)
        for patient_id, data in sorted(processed_data.items())
    ]


def validate_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        processed_data = pickle.load(f)
    return validate_processed_data(processed_data)


def summarize_validation(report):
    passed = [r for r in report if r["passed"]]
    failed = [r for r in report if not r["passed"]]

    return {
        "total": len(report),
        "passed": len(passed),
        "failed": len(failed),
        "failures": failed,
    }

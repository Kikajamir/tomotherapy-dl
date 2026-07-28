"""
Sanity checks for the raw-dataset audit tool (`preprocessing.audit_dataset`),
using small synthetic patient folders under pytest's tmp_path instead of
real TomoTherapy data.
"""

import os

from preprocessing.audit_dataset import (
    run_audit,
    organize_patients,
    locate_patient_dir,
    discover_patient_dirs,
)


def _make_patient(root, patient_id, rtplan=1, detector=1, planned=1):
    """
    Filenames must match exactly (case-insensitively), so a "duplicate"
    can't just be a differently-named file in the same folder -- it has
    to be a second file with the matching name in a different
    subdirectory (which is how real recursive duplicates arise).
    """
    patient_dir = os.path.join(root, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    for i in range(rtplan):
        sub = os.path.join(patient_dir, f"RTP{i + 1}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "plan.cdms"), "w") as f:
            f.write("x")

    for i in range(detector):
        sub = patient_dir if i == 0 else os.path.join(patient_dir, f"extra{i}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "D1.csv"), "w") as f:
            f.write("x")

    for i in range(planned):
        sub = patient_dir if i == 0 else os.path.join(patient_dir, f"extra{i}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "planned.csv"), "w") as f:
            f.write("x")

    return patient_dir


def test_complete_patient_is_classified_complete(tmp_path):
    _make_patient(tmp_path, "P1")

    results = run_audit(dataset_root=str(tmp_path), num_patients=1)

    assert results[0].status == "Complete"
    assert results[0].exists_complete


def test_missing_rtplan_is_detected(tmp_path):
    _make_patient(tmp_path, "P1", rtplan=0)

    results = run_audit(dataset_root=str(tmp_path), num_patients=1)

    assert results[0].status == "Missing RTPLAN"
    assert not results[0].exists_complete


def test_missing_detector_and_planned_are_detected(tmp_path):
    _make_patient(tmp_path, "P1", detector=0)
    _make_patient(tmp_path, "P2", planned=0)

    results = run_audit(dataset_root=str(tmp_path), num_patients=2)

    assert results[0].status == "Missing Detector CSV"
    assert results[1].status == "Missing Planned CSV"


def test_duplicate_rtplan_and_planned_are_flagged(tmp_path):
    _make_patient(tmp_path, "P1", rtplan=2)
    _make_patient(tmp_path, "P2", planned=2)

    results = run_audit(dataset_root=str(tmp_path), num_patients=2)

    assert results[0].status == "Multiple RTPLAN Files"
    # Duplicates still count as "present" for the exists-based folder split.
    assert results[0].exists_complete

    assert results[1].status == "Multiple Planned Files"
    assert results[1].exists_complete


def test_planned_filename_match_is_case_insensitive(tmp_path):
    patient_dir = os.path.join(tmp_path, "P1")
    rtp_dir = os.path.join(patient_dir, "RTP1")
    os.makedirs(rtp_dir, exist_ok=True)
    with open(os.path.join(rtp_dir, "plan.cdms"), "w") as f:
        f.write("x")
    with open(os.path.join(patient_dir, "D1.csv"), "w") as f:
        f.write("x")
    with open(os.path.join(patient_dir, "PLANNED.csv"), "w") as f:
        f.write("x")

    results = run_audit(dataset_root=str(tmp_path), num_patients=1)

    assert results[0].status == "Complete"


def test_organize_moves_into_complete_and_incomplete(tmp_path):
    _make_patient(tmp_path, "P1")
    _make_patient(tmp_path, "P2", rtplan=0)

    results = run_audit(dataset_root=str(tmp_path), num_patients=2)
    organize_patients(results, dataset_root=str(tmp_path))

    assert os.path.isdir(os.path.join(tmp_path, "Complete", "P1"))
    assert os.path.isdir(os.path.join(tmp_path, "Incomplete", "P2"))
    assert not os.path.isdir(os.path.join(tmp_path, "P1"))
    assert not os.path.isdir(os.path.join(tmp_path, "P2"))


def test_rerun_after_organize_finds_relocated_patients(tmp_path):
    _make_patient(tmp_path, "P1")
    results = run_audit(dataset_root=str(tmp_path), num_patients=1)
    organize_patients(results, dataset_root=str(tmp_path))

    located = locate_patient_dir(str(tmp_path), "P1")
    assert located == os.path.join(str(tmp_path), "Complete", "P1")

    results_again = run_audit(dataset_root=str(tmp_path), num_patients=1)
    assert results_again[0].status == "Complete"

    actions = organize_patients(results_again, dataset_root=str(tmp_path))
    assert actions[0][1] == "already-in-place"


def test_discover_patient_dirs_finds_patients_regardless_of_location(tmp_path):
    # One patient still sitting directly under the dataset root, one
    # already filed into Complete/, one filed into Incomplete/ (e.g. for
    # missing RTPLAN only) -- discovery should find all three.
    os.makedirs(tmp_path / "P1")
    os.makedirs(tmp_path / "Complete" / "P2")
    os.makedirs(tmp_path / "Incomplete" / "P37")

    found = discover_patient_dirs(str(tmp_path))

    assert set(found.keys()) == {"P1", "P2", "P37"}
    assert found["P1"] == str(tmp_path / "P1")
    assert found["P2"] == str(tmp_path / "Complete" / "P2")
    assert found["P37"] == str(tmp_path / "Incomplete" / "P37")


def test_discover_patient_dirs_does_not_treat_container_dirs_as_patients(tmp_path):
    os.makedirs(tmp_path / "Complete" / "P1")
    os.makedirs(tmp_path / "Incomplete" / "P2")

    found = discover_patient_dirs(str(tmp_path))

    assert "Complete" not in found
    assert "Incomplete" not in found
    assert set(found.keys()) == {"P1", "P2"}


def test_discover_patient_dirs_works_with_no_complete_incomplete_split(tmp_path):
    # Pre-audit layout: patients directly under the dataset root, no
    # Complete/Incomplete folders at all.
    os.makedirs(tmp_path / "P5")
    os.makedirs(tmp_path / "P9")

    found = discover_patient_dirs(str(tmp_path))

    assert set(found.keys()) == {"P5", "P9"}

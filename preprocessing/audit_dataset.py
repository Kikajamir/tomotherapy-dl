"""
Raw dataset audit for the "DA exit data_new" patient export.

Independent of `build_dataset.py`: that module only looks for a top-level
`planned.csv` / `D1.csv` per patient and silently skips anything else.
This script instead recursively verifies, per patient, that exactly one
RTPLAN DICOM export (`*.cdms`), one delivered detector export (`D1.csv`),
and one planned fluence export (`planned.csv`, case-insensitive) exist --
and flags duplicates rather than silently picking one. It is read-only
except for `organize_patients`, which files each patient folder under
`Complete/` or `Incomplete/` based on plain file-existence (a patient
with a duplicate is still "existing", just flagged in `status` for
manual review).

Rerunnable: `locate_patient_dir` looks in the dataset root and in both
`Complete/` and `Incomplete/`, so running this again after a prior
`organize_patients` call re-audits patients in place and moves them again
only if their status has changed.
"""

import argparse
import os
import shutil
from dataclasses import dataclass, field

NUM_PATIENTS = 40

RTPLAN_EXTENSION = ".cdms"
DETECTOR_FILENAME = "d1.csv"
PLANNED_FILENAME = "planned.csv"

COMPLETE_DIRNAME = "Complete"
INCOMPLETE_DIRNAME = "Incomplete"


def default_dataset_root():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "DA exit data_new")


# =====================================================================
# Data model
# =====================================================================

@dataclass
class PatientAudit:
    patient: str
    current_dir: str = None
    rtplan_files: list = field(default_factory=list)
    detector_files: list = field(default_factory=list)
    planned_files: list = field(default_factory=list)
    status: str = "Other issue"

    @property
    def has_rtplan(self):
        return len(self.rtplan_files) > 0

    @property
    def has_detector(self):
        return len(self.detector_files) > 0

    @property
    def has_planned(self):
        return len(self.planned_files) > 0

    @property
    def exists_complete(self):
        """All three required files exist (duplicates notwithstanding)."""
        return self.has_rtplan and self.has_detector and self.has_planned


# =====================================================================
# Discovery
# =====================================================================

def locate_patient_dir(dataset_root, patient_id):
    """
    Finds a patient's folder whether it still sits directly under
    `dataset_root` or was already filed into Complete/Incomplete by a
    previous `organize_patients` run.
    """
    candidates = [
        os.path.join(dataset_root, patient_id),
        os.path.join(dataset_root, COMPLETE_DIRNAME, patient_id),
        os.path.join(dataset_root, INCOMPLETE_DIRNAME, patient_id),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return None


def discover_patient_dirs(dataset_root):
    """
    Every patient folder found anywhere under `dataset_root`: directly
    inside it, and/or one level inside `Complete/` or `Incomplete/` --
    however the dataset happens to be organized right now. No assumption
    about patient count, numbering, or contiguity; a folder's own name
    is used as its patient ID verbatim.

    Returns {patient_id: absolute_path}. If the same patient_id exists
    in more than one of those locations, the first one found wins (in
    the order: dataset_root, Complete/, Incomplete/).
    """
    search_roots = [dataset_root]

    for sub in (COMPLETE_DIRNAME, INCOMPLETE_DIRNAME):
        candidate = os.path.join(dataset_root, sub)
        if os.path.isdir(candidate):
            search_roots.append(candidate)

    patient_dirs = {}

    for root in search_roots:
        for name in sorted(os.listdir(root)):
            if name in (COMPLETE_DIRNAME, INCOMPLETE_DIRNAME):
                continue

            full_path = os.path.join(root, name)

            if os.path.isdir(full_path):
                patient_dirs.setdefault(name, full_path)

    return patient_dirs


def find_files_recursive(patient_dir, predicate):
    matches = []

    for root, _dirs, files in os.walk(patient_dir):
        for name in files:
            if predicate(name):
                matches.append(os.path.join(root, name))

    return sorted(matches)


# =====================================================================
# Per-patient audit
# =====================================================================

def classify_status(audit):
    if not audit.has_rtplan:
        return "Missing RTPLAN"
    if not audit.has_detector:
        return "Missing Detector CSV"
    if not audit.has_planned:
        return "Missing Planned CSV"
    if len(audit.rtplan_files) > 1:
        return "Multiple RTPLAN Files"
    if len(audit.planned_files) > 1:
        return "Multiple Planned Files"
    if len(audit.detector_files) > 1:
        return "Other issue"
    return "Complete"


def audit_patient(patient_id, patient_dir):
    if patient_dir is None:
        audit = PatientAudit(patient=patient_id, current_dir=None)
        audit.status = "Other issue"
        return audit

    audit = PatientAudit(
        patient=patient_id,
        current_dir=patient_dir,
        rtplan_files=find_files_recursive(
            patient_dir, lambda f: f.lower().endswith(RTPLAN_EXTENSION)
        ),
        detector_files=find_files_recursive(
            patient_dir, lambda f: f.lower() == DETECTOR_FILENAME
        ),
        planned_files=find_files_recursive(
            patient_dir, lambda f: f.lower() == PLANNED_FILENAME
        ),
    )
    audit.status = classify_status(audit)

    return audit


def run_audit(dataset_root=None, num_patients=NUM_PATIENTS):
    """Audits P1..P{num_patients} and returns a list of PatientAudit."""
    dataset_root = dataset_root or default_dataset_root()

    results = []

    for i in range(1, num_patients + 1):
        patient_id = f"P{i}"
        patient_dir = locate_patient_dir(dataset_root, patient_id)
        results.append(audit_patient(patient_id, patient_dir))

    return results


# =====================================================================
# Reporting
# =====================================================================

def format_table(results):
    header = f"{'Patient':<8}{'RTPLAN (.cdms)':<16}{'D1.csv':<10}{'planned.csv':<14}{'Status':<24}"
    lines = [header, "-" * len(header)]

    for audit in results:
        lines.append(
            f"{audit.patient:<8}"
            f"{'Yes' if audit.has_rtplan else 'No':<16}"
            f"{'Yes' if audit.has_detector else 'No':<10}"
            f"{'Yes' if audit.has_planned else 'No':<14}"
            f"{audit.status:<24}"
        )

    return "\n".join(lines)


def build_report(results):
    complete = [a for a in results if a.status == "Complete"]
    incomplete = [a for a in results if a.status != "Complete"]

    lines = []
    lines.append(f"Complete patients   : {len(complete)}")
    lines.append(f"Incomplete patients : {len(incomplete)}")
    lines.append("")

    lines.append("Missing files by incomplete patient:")
    if not incomplete:
        lines.append("  (none)")
    for audit in incomplete:
        missing = []
        if not audit.has_rtplan:
            missing.append("RTPLAN (.cdms)")
        if not audit.has_detector:
            missing.append("D1.csv")
        if not audit.has_planned:
            missing.append("planned.csv")
        note = ", ".join(missing) if missing else f"no missing files ({audit.status})"
        lines.append(f"  {audit.patient}: {note}")

    lines.append("")
    lines.append("Duplicate RTPLAN files:")
    dup_rtplan = [a for a in results if len(a.rtplan_files) > 1]
    if not dup_rtplan:
        lines.append("  (none)")
    for audit in dup_rtplan:
        lines.append(f"  {audit.patient}:")
        for path in audit.rtplan_files:
            lines.append(f"    {path}")

    lines.append("")
    lines.append("Duplicate planned files:")
    dup_planned = [a for a in results if len(a.planned_files) > 1]
    if not dup_planned:
        lines.append("  (none)")
    for audit in dup_planned:
        lines.append(f"  {audit.patient}:")
        for path in audit.planned_files:
            lines.append(f"    {path}")

    lines.append("")
    lines.append("Duplicate detector (D1.csv) files:")
    dup_detector = [a for a in results if len(a.detector_files) > 1]
    if not dup_detector:
        lines.append("  (none)")
    for audit in dup_detector:
        lines.append(f"  {audit.patient}:")
        for path in audit.detector_files:
            lines.append(f"    {path}")

    return "\n".join(lines)


# =====================================================================
# Organize (the only function that touches the filesystem)
# =====================================================================

def organize_patients(results, dataset_root=None, dry_run=False):
    """
    Moves each patient folder into Complete/ or Incomplete/ under
    `dataset_root`, based on plain file existence (`exists_complete`) --
    not on `status`, so a patient flagged e.g. "Multiple Planned Files"
    still counts as complete for filing purposes since all three file
    types are present.

    Returns a list of (patient_id, action, destination) tuples, where
    action is one of "moved", "already-in-place", "skipped-not-found".
    """
    dataset_root = dataset_root or default_dataset_root()
    complete_dir = os.path.join(dataset_root, COMPLETE_DIRNAME)
    incomplete_dir = os.path.join(dataset_root, INCOMPLETE_DIRNAME)

    if not dry_run:
        os.makedirs(complete_dir, exist_ok=True)
        os.makedirs(incomplete_dir, exist_ok=True)

    actions = []

    for audit in results:
        if audit.current_dir is None:
            actions.append((audit.patient, "skipped-not-found", None))
            continue

        target_root = complete_dir if audit.exists_complete else incomplete_dir
        destination = os.path.join(target_root, audit.patient)

        if os.path.normcase(os.path.abspath(audit.current_dir)) == os.path.normcase(
            os.path.abspath(destination)
        ):
            actions.append((audit.patient, "already-in-place", destination))
            continue

        if not dry_run:
            shutil.move(audit.current_dir, destination)

        actions.append((audit.patient, "moved", destination))

    return actions


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--num-patients", type=int, default=NUM_PATIENTS)
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Audit and report only; do not move any patient folders.",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Where to save the text report (defaults to audit_report.txt "
        "inside the dataset root).",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root or default_dataset_root()

    results = run_audit(dataset_root, args.num_patients)

    print(format_table(results))
    print()

    report = build_report(results)
    print(report)

    if not args.no_move:
        print()
        actions = organize_patients(results, dataset_root)
        moved = [a for a in actions if a[1] == "moved"]
        print(f"Moved {len(moved)} patient folder(s) into "
              f"{COMPLETE_DIRNAME}/ or {INCOMPLETE_DIRNAME}/.")

    report_file = args.report_file or os.path.join(dataset_root, "audit_report.txt")
    with open(report_file, "w") as f:
        f.write(format_table(results))
        f.write("\n\n")
        f.write(report)
        f.write("\n")
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()

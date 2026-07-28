"""
RTPLAN metadata extraction for the `Complete/` patient set.

For every patient folder under the dataset's `Complete/` directory (see
`preprocessing.audit_dataset`), this module recursively locates the RTPLAN
DICOM export (`*.cdms`), reads it with pydicom, verifies it is really an
RT Plan, and dumps every tag requested by the audit spec into a plain
dict tree. It does not train, reshape, or interpret the data for the
model -- this is a QA/archival step only. Output:

    metadata/rtplan_raw.json     one JSON object per patient
    metadata/rtplan_summary.csv  one flattened row per patient

Two of the requested "TomoTherapy-specific" values (Pitch, Field Width)
are not exposed as documented DICOM tags in these files -- they only
appear as free text inside `BeamDescription`, e.g.
"Pitch=0.175 and field width=50.44". They are parsed from that string
via regex and labelled as such (`tomotherapy_specific.source`).

Modulation Factor, Couch Speed, per-projection MLC leaf pattern, and any
other TomoTherapy value are not present as named/documented DICOM
attributes at all: TomoTherapy stores them under an undocumented private
block (creator "TOMO_HA_01", group 0x300D, VR "UN"). Per the "extract
without making assumptions" requirement, this module does NOT guess which
private tag is which physical quantity -- it preserves every private tag
it finds verbatim, keyed by its DICOM tag address, under
`private_tags` at whichever level (dataset/beam/control point) it was
found. One notable private tag, present once per control point with
exactly 64 backslash-separated numeric values, lines up with TomoTherapy's
known 64-leaf binary MLC (the same leaf count `configs.config` uses to
build `leaf_mm`) -- it is surfaced as-is, but its meaning is not asserted.
"""

import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.valuerep import PersonName

from preprocessing.audit_dataset import (
    COMPLETE_DIRNAME,
    RTPLAN_EXTENSION,
    default_dataset_root,
    find_files_recursive,
)

RTPLAN_MODALITY = "RTPLAN"
RTPLAN_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.481.5"

METADATA_DIRNAME = "metadata"
RTPLAN_RAW_JSON_FILENAME = "rtplan_raw.json"
RTPLAN_SUMMARY_CSV_FILENAME = "rtplan_summary.csv"

BEAM_DESCRIPTION_PATTERN = re.compile(
    r"pitch\s*=\s*(?P<pitch>[\d.]+).*?field\s*width\s*=\s*(?P<field_width>[\d.]+)",
    re.IGNORECASE,
)

PATIENT_INFO_KEYWORDS = [
    "PatientID",
    "PatientName",
    "StudyDate",
    "Manufacturer",
    "ManufacturerModelName",
]

BEAM_INFO_KEYWORDS = [
    "BeamName",
    "BeamNumber",
    "BeamType",
    "RadiationType",
    "TreatmentMachineName",
    "PrimaryDosimeterUnit",
    "SourceAxisDistance",
    "BeamDescription",
]

CONTROL_POINT_KEYWORDS = [
    "ControlPointIndex",
    "GantryAngle",
    "GantryRotationDirection",
    "BeamLimitingDeviceAngle",
    "PatientSupportAngle",
    "IsocenterPosition",
    "CumulativeMetersetWeight",
    "TableTopVerticalPosition",
    "TableTopLongitudinalPosition",
    "TableTopLateralPosition",
    "TableTopPitchAngle",
    "TableTopRollAngle",
    "TableTopEccentricAngle",
]

FRACTION_GROUP_KEYWORDS = [
    "FractionGroupNumber",
    "NumberOfFractionsPlanned",
    "NumberOfBeams",
]

REFERENCED_BEAM_KEYWORDS = [
    "ReferencedBeamNumber",
    "BeamDose",
    "BeamMeterset",
]


def default_metadata_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, METADATA_DIRNAME)


def default_complete_dir() -> str:
    return os.path.join(default_dataset_root(), COMPLETE_DIRNAME)


# =====================================================================
# Value conversion (pydicom types -> plain JSON-serializable Python)
# =====================================================================

def _maybe_number(token: str) -> Any:
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def convert_private_bytes(raw: bytes) -> Any:
    """
    Private tags with VR "UN" arrive as raw bytes. TomoTherapy encodes
    them as ASCII, sometimes as a single number, sometimes as a
    backslash-separated array (DICOM's usual multi-value separator).
    """
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return raw.hex()

    text = text.strip()

    if "\\" in text:
        return [_maybe_number(part) for part in text.split("\\")]

    return _maybe_number(text)


def convert_value(value: Any) -> Any:
    """Recursively converts a pydicom element value into plain Python."""
    if value is None:
        return None
    if isinstance(value, PersonName):
        return str(value)
    if isinstance(value, bytes):
        return convert_private_bytes(value)
    if isinstance(value, (MultiValue, list, tuple)):
        return [convert_value(v) for v in value]
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def get_value(dataset: Dataset, keyword: str) -> Any:
    return convert_value(dataset.get(keyword))


def extract_private_tags(dataset: Dataset) -> Dict[str, Any]:
    """
    Every private element directly on `dataset` (not descending into
    sequences), keyed by its DICOM tag address, e.g. "(300d, 1040)".
    Values are preserved verbatim -- no attempt is made to name or
    interpret them.
    """
    private_tags: Dict[str, Any] = {}

    for element in dataset:
        if element.tag.is_private:
            private_tags[str(element.tag)] = convert_value(element.value)

    return private_tags


# =====================================================================
# Discovery
# =====================================================================

def find_rtplan_files(patient_dir: str) -> List[str]:
    return find_files_recursive(
        patient_dir, lambda name: name.lower().endswith(RTPLAN_EXTENSION)
    )


def is_rtplan(dataset: Dataset) -> bool:
    modality = dataset.get("Modality")
    sop_class_uid = str(dataset.get("SOPClassUID", ""))
    return modality == RTPLAN_MODALITY or sop_class_uid == RTPLAN_SOP_CLASS_UID


# =====================================================================
# Per-section extraction
# =====================================================================

def extract_patient_info(dataset: Dataset) -> Dict[str, Any]:
    return {keyword: get_value(dataset, keyword) for keyword in PATIENT_INFO_KEYWORDS}


def extract_beam_limiting_devices(beam: Dataset) -> List[Dict[str, Any]]:
    devices = []

    for item in beam.get("BeamLimitingDeviceSequence", []):
        devices.append(
            {
                "device_type": get_value(item, "RTBeamLimitingDeviceType"),
                "number_of_leaf_pairs": get_value(item, "NumberOfLeafJawPairs"),
            }
        )

    return devices


def extract_beam_limiting_device_positions(control_point: Dataset) -> Dict[str, Any]:
    positions = {}

    for item in control_point.get("BeamLimitingDevicePositionSequence", []):
        device_type = get_value(item, "RTBeamLimitingDeviceType")
        positions[device_type] = get_value(item, "LeafJawPositions")

    return positions


def extract_control_points(beam: Dataset) -> List[Dict[str, Any]]:
    control_points = []

    for cp in beam.get("ControlPointSequence", []):
        record = {keyword: get_value(cp, keyword) for keyword in CONTROL_POINT_KEYWORDS}
        record["beam_limiting_device_positions"] = extract_beam_limiting_device_positions(cp)
        record["private_tags"] = extract_private_tags(cp)
        control_points.append(record)

    return control_points


def parse_tomotherapy_specific(beam: Dataset) -> Dict[str, Any]:
    description = get_value(beam, "BeamDescription") or ""
    match = BEAM_DESCRIPTION_PATTERN.search(description)

    pitch = float(match.group("pitch")) if match else None
    field_width_cm = float(match.group("field_width")) if match else None

    return {
        "pitch": pitch,
        "field_width_cm": field_width_cm,
        "source": "parsed from BeamDescription free text" if match else None,
        "modulation_factor": None,
        "couch_speed": None,
        "projection_count": None,
        "beam_description_raw": description or None,
        "private_tags": extract_private_tags(beam),
    }


def extract_beam(beam: Dataset) -> Dict[str, Any]:
    record = {keyword: get_value(beam, keyword) for keyword in BEAM_INFO_KEYWORDS}
    record["beam_limiting_devices"] = extract_beam_limiting_devices(beam)
    record["number_of_control_points_declared"] = get_value(beam, "NumberOfControlPoints")
    control_points = extract_control_points(beam)
    record["control_point_count_actual"] = len(control_points)
    record["control_points"] = control_points
    record["tomotherapy_specific"] = parse_tomotherapy_specific(beam)
    return record


def extract_fraction_groups(dataset: Dataset) -> List[Dict[str, Any]]:
    fraction_groups = []

    for fg in dataset.get("FractionGroupSequence", []):
        record = {keyword: get_value(fg, keyword) for keyword in FRACTION_GROUP_KEYWORDS}
        referenced_beams = []
        for item in fg.get("ReferencedBeamSequence", []):
            referenced_beams.append(
                {keyword: get_value(item, keyword) for keyword in REFERENCED_BEAM_KEYWORDS}
            )
        record["referenced_beam_sequence"] = referenced_beams
        fraction_groups.append(record)

    return fraction_groups


# =====================================================================
# Per-patient orchestration
# =====================================================================

@dataclass
class PatientRTPlanRecord:
    patient: str
    status: str
    rtplan_files_found: List[str]
    source_file: Optional[str] = None
    error: Optional[str] = None
    patient_info: Optional[Dict[str, Any]] = None
    modality: Optional[str] = None
    sop_class_uid: Optional[str] = None
    fraction_groups: Optional[List[Dict[str, Any]]] = None
    beams: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient": self.patient,
            "status": self.status,
            "rtplan_files_found": self.rtplan_files_found,
            "source_file": self.source_file,
            "error": self.error,
            "patient_info": self.patient_info,
            "modality": self.modality,
            "sop_class_uid": self.sop_class_uid,
            "fraction_groups": self.fraction_groups,
            "beams": self.beams,
        }


def parse_patient_rtplan(patient_id: str, patient_dir: str) -> PatientRTPlanRecord:
    rtplan_files = find_rtplan_files(patient_dir)

    if not rtplan_files:
        return PatientRTPlanRecord(
            patient=patient_id,
            status="missing_rtplan",
            rtplan_files_found=[],
            error="No *.cdms file found under patient folder.",
        )

    source_file = rtplan_files[0]

    try:
        dataset = pydicom.dcmread(source_file)
    except (InvalidDicomError, OSError) as exc:
        return PatientRTPlanRecord(
            patient=patient_id,
            status="read_error",
            rtplan_files_found=rtplan_files,
            source_file=source_file,
            error=str(exc),
        )

    modality = dataset.get("Modality")
    sop_class_uid = str(dataset.get("SOPClassUID", "")) or None

    if not is_rtplan(dataset):
        return PatientRTPlanRecord(
            patient=patient_id,
            status="not_rtplan",
            rtplan_files_found=rtplan_files,
            source_file=source_file,
            modality=modality,
            sop_class_uid=sop_class_uid,
            error=f"Modality={modality!r}, SOPClassUID={sop_class_uid!r} "
            f"does not match RTPLAN ({RTPLAN_SOP_CLASS_UID}).",
        )

    return PatientRTPlanRecord(
        patient=patient_id,
        status="ok",
        rtplan_files_found=rtplan_files,
        source_file=source_file,
        modality=modality,
        sop_class_uid=sop_class_uid,
        patient_info=extract_patient_info(dataset),
        fraction_groups=extract_fraction_groups(dataset),
        beams=[extract_beam(beam) for beam in dataset.get("BeamSequence", [])],
    )


def discover_patients(complete_dir: str) -> List[str]:
    def sort_key(name: str):
        match = re.fullmatch(r"P(\d+)", name)
        return (0, int(match.group(1))) if match else (1, name)

    if not os.path.isdir(complete_dir):
        return []

    patients = [
        name
        for name in os.listdir(complete_dir)
        if os.path.isdir(os.path.join(complete_dir, name))
    ]

    return sorted(patients, key=sort_key)


def build_rtplan_dataset(complete_dir: str = None) -> Dict[str, Dict[str, Any]]:
    complete_dir = complete_dir or default_complete_dir()

    dataset: Dict[str, Dict[str, Any]] = {}

    for patient_id in discover_patients(complete_dir):
        patient_dir = os.path.join(complete_dir, patient_id)
        record = parse_patient_rtplan(patient_id, patient_dir)
        dataset[patient_id] = record.to_dict()

    return dataset


# =====================================================================
# Output
# =====================================================================

def save_raw_json(dataset: Dict[str, Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def build_summary_row(patient_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "patient": patient_id,
        "status": record.get("status"),
        "error": record.get("error"),
        "source_file": record.get("source_file"),
    }

    patient_info = record.get("patient_info") or {}
    row["patient_id_dicom"] = patient_info.get("PatientID")
    row["patient_name"] = patient_info.get("PatientName")
    row["study_date"] = patient_info.get("StudyDate")
    row["manufacturer"] = patient_info.get("Manufacturer")
    row["manufacturer_model_name"] = patient_info.get("ManufacturerModelName")

    beams = record.get("beams") or []
    row["num_beams"] = len(beams)
    row["beam_names"] = ";".join(str(b.get("BeamName")) for b in beams)

    first_beam = beams[0] if beams else {}
    row["treatment_machine_name"] = first_beam.get("TreatmentMachineName")
    row["radiation_type"] = first_beam.get("RadiationType")
    row["number_of_control_points"] = first_beam.get("number_of_control_points_declared")

    tomo = first_beam.get("tomotherapy_specific") or {}
    row["pitch"] = tomo.get("pitch")
    row["field_width_cm"] = tomo.get("field_width_cm")

    fraction_groups = record.get("fraction_groups") or []
    first_fg = fraction_groups[0] if fraction_groups else {}
    row["number_of_fractions_planned"] = first_fg.get("NumberOfFractionsPlanned")
    row["total_beam_meterset_mu"] = sum(
        ref.get("BeamMeterset") or 0
        for ref in first_fg.get("referenced_beam_sequence", [])
    ) if fraction_groups else None

    return row


def build_summary_rows(dataset: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [build_summary_row(patient_id, record) for patient_id, record in dataset.items()]


def save_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "patient",
        "status",
        "patient_id_dicom",
        "patient_name",
        "study_date",
        "manufacturer",
        "manufacturer_model_name",
        "num_beams",
        "beam_names",
        "treatment_machine_name",
        "radiation_type",
        "number_of_control_points",
        "pitch",
        "field_width_cm",
        "number_of_fractions_planned",
        "total_beam_meterset_mu",
        "source_file",
        "error",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =====================================================================
# CLI
# =====================================================================

def run(complete_dir: str = None, metadata_dir: str = None) -> Dict[str, Dict[str, Any]]:
    complete_dir = complete_dir or default_complete_dir()
    metadata_dir = metadata_dir or default_metadata_dir()

    dataset = build_rtplan_dataset(complete_dir)

    save_raw_json(dataset, os.path.join(metadata_dir, RTPLAN_RAW_JSON_FILENAME))
    save_summary_csv(
        build_summary_rows(dataset), os.path.join(metadata_dir, RTPLAN_SUMMARY_CSV_FILENAME)
    )

    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    args = parser.parse_args()

    dataset = run(args.complete_dir, args.metadata_dir)

    statuses: Dict[str, int] = {}
    for record in dataset.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1

    print(f"Parsed {len(dataset)} patients.")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()

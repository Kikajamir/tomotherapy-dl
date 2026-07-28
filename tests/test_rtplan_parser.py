"""
Sanity checks for `preprocessing.rtplan_parser`, using small synthetic
pydicom datasets (no real patient data, no mocking).
"""

import csv
import json
import os

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.multival import MultiValue
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.valuerep import PersonName

from preprocessing.rtplan_parser import (
    build_rtplan_dataset,
    build_summary_row,
    build_summary_rows,
    convert_private_bytes,
    convert_value,
    discover_patients,
    extract_beam,
    extract_beam_limiting_device_positions,
    extract_beam_limiting_devices,
    extract_control_points,
    extract_fraction_groups,
    extract_patient_info,
    extract_private_tags,
    find_rtplan_files,
    is_rtplan,
    parse_patient_rtplan,
    parse_tomotherapy_specific,
    run,
    save_raw_json,
    save_summary_csv,
)


# =====================================================================
# Synthetic dataset builders
# =====================================================================

def _make_control_point(index, gantry_angle, jaw_x=(-200.0, 200.0), leaf_pattern=None):
    cp = Dataset()
    cp.ControlPointIndex = index
    cp.GantryAngle = gantry_angle
    cp.CumulativeMetersetWeight = index * 0.01
    cp.IsocenterPosition = [1.0, 2.0, float(index)]

    jaw = Dataset()
    jaw.RTBeamLimitingDeviceType = "X"
    jaw.LeafJawPositions = list(jaw_x)
    cp.BeamLimitingDevicePositionSequence = [jaw]

    if leaf_pattern is not None:
        cp.add_new(0x300D0010, "LO", "TOMO_HA_01")
        packed = "\\".join(str(v) for v in leaf_pattern)
        cp.add_new(0x300D10A7, "UN", packed.encode("ascii"))

    return cp


def _make_beam(beam_number=1, name="H01 HELICAL", description="Pitch=0.200 and field width=25.08",
               num_control_points=3):
    beam = Dataset()
    beam.BeamName = name
    beam.BeamNumber = beam_number
    beam.BeamType = "DYNAMIC"
    beam.RadiationType = "PHOTON"
    beam.TreatmentMachineName = "TEST_MACHINE"
    beam.PrimaryDosimeterUnit = "MINUTE"
    beam.SourceAxisDistance = "850.0"
    beam.BeamDescription = description
    beam.NumberOfControlPoints = num_control_points

    device = Dataset()
    device.RTBeamLimitingDeviceType = "X"
    device.NumberOfLeafJawPairs = 1
    beam.BeamLimitingDeviceSequence = [device]

    beam.ControlPointSequence = [
        _make_control_point(i, gantry_angle=10.0 * i, leaf_pattern=[0, 1, 0] if i == 1 else None)
        for i in range(num_control_points)
    ]

    beam.add_new(0x300D0010, "LO", "TOMO_HA_01")
    beam.add_new(0x300D1060, "UN", b"0.20000 ")

    return beam


def _make_rtplan_dataset(patient_id="TESTID", num_beams=1):
    ds = Dataset()
    ds.PatientID = patient_id
    ds.PatientName = "Test^Patient"
    ds.StudyDate = "20240101"
    ds.Manufacturer = "TomoTherapy Incorporated"
    ds.ManufacturerModelName = "Hi-Art"
    ds.Modality = "RTPLAN"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.5"
    ds.SOPInstanceUID = generate_uid()

    beams = [_make_beam(beam_number=i + 1) for i in range(num_beams)]
    ds.BeamSequence = beams

    fg = Dataset()
    fg.FractionGroupNumber = 1
    fg.NumberOfFractionsPlanned = 15
    fg.NumberOfBeams = num_beams

    referenced = []
    for beam in beams:
        ref = Dataset()
        ref.ReferencedBeamNumber = beam.BeamNumber
        ref.BeamDose = 3.2
        ref.BeamMeterset = 7.95
        referenced.append(ref)
    fg.ReferencedBeamSequence = referenced

    ds.FractionGroupSequence = [fg]

    return ds


def _write_dicom_file(ds, path):
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.save_as(path, enforce_file_format=True, little_endian=True, implicit_vr=False)


# =====================================================================
# Value conversion
# =====================================================================

def test_convert_value_handles_person_name():
    assert convert_value(PersonName("Doe^Jane")) == "Doe^Jane"


def test_convert_value_handles_multivalue_and_none():
    assert convert_value(MultiValue(float, [1.0, 2.0])) == [1.0, 2.0]
    assert convert_value(None) is None
    assert convert_value(3.5) == 3.5
    assert convert_value("X") == "X"


def test_convert_private_bytes_single_number():
    assert convert_private_bytes(b"13.2") == 13.2


def test_convert_private_bytes_backslash_separated_array():
    result = convert_private_bytes(b"0\\1\\0\\2")
    assert result == [0, 1, 0, 2]


def test_convert_private_bytes_non_ascii_falls_back_to_hex():
    raw = bytes([0xFF, 0xFE, 0x00])
    result = convert_private_bytes(raw)
    assert result == raw.hex()


# =====================================================================
# Section extraction
# =====================================================================

def test_is_rtplan_true_and_false():
    ds = _make_rtplan_dataset()
    assert is_rtplan(ds) is True

    ds.Modality = "RTDOSE"
    del ds.SOPClassUID
    assert is_rtplan(ds) is False


def test_extract_patient_info_missing_fields_are_none():
    ds = Dataset()
    ds.PatientID = "ABC"
    info = extract_patient_info(ds)
    assert info["PatientID"] == "ABC"
    assert info["StudyDate"] is None


def test_extract_beam_limiting_devices():
    beam = _make_beam()
    devices = extract_beam_limiting_devices(beam)
    assert devices == [{"device_type": "X", "number_of_leaf_pairs": 1}]


def test_extract_beam_limiting_device_positions():
    cp = _make_control_point(0, gantry_angle=38.82, jaw_x=(-25.0, 25.0))
    positions = extract_beam_limiting_device_positions(cp)
    assert positions == {"X": [-25.0, 25.0]}


def test_extract_control_points_includes_jaw_and_private_leaf_pattern():
    beam = _make_beam(num_control_points=3)
    control_points = extract_control_points(beam)

    assert len(control_points) == 3
    assert control_points[0]["GantryAngle"] == 0.0
    assert control_points[1]["GantryAngle"] == 10.0
    assert control_points[0]["beam_limiting_device_positions"] == {"X": [-200.0, 200.0]}

    # Locate the leaf-pattern private tag generically instead of assuming its exact
    # tag string casing, matching the "no assumptions" extraction style.
    leaf_values = [
        v for k, v in control_points[1]["private_tags"].items() if k != "(300D,0010)"
    ]
    assert leaf_values == [[0, 1, 0]]


def test_extract_private_tags_ignores_public_elements():
    beam = _make_beam()
    private = extract_private_tags(beam)
    assert "(300D,0010)" in private
    assert private["(300D,0010)"] == "TOMO_HA_01"
    assert "(300A,00C2)" not in private  # BeamName is public, not private


def test_parse_tomotherapy_specific_parses_pitch_and_field_width():
    beam = _make_beam(description="Pitch=0.175 and field width=50.44")
    result = parse_tomotherapy_specific(beam)
    assert result["pitch"] == 0.175
    assert result["field_width_cm"] == 50.44
    assert result["source"] is not None
    assert result["modulation_factor"] is None
    assert result["couch_speed"] is None
    assert result["projection_count"] is None


def test_parse_tomotherapy_specific_returns_none_when_not_present():
    beam = _make_beam(description="TomoTherapy fixed-angle beam:1")
    result = parse_tomotherapy_specific(beam)
    assert result["pitch"] is None
    assert result["field_width_cm"] is None
    assert result["source"] is None


def test_extract_fraction_groups():
    ds = _make_rtplan_dataset(num_beams=2)
    groups = extract_fraction_groups(ds)

    assert len(groups) == 1
    assert groups[0]["NumberOfFractionsPlanned"] == 15
    assert len(groups[0]["referenced_beam_sequence"]) == 2
    assert groups[0]["referenced_beam_sequence"][0]["ReferencedBeamNumber"] == 1


def test_extract_beam_reports_declared_vs_actual_control_point_count():
    beam = _make_beam(num_control_points=3)
    beam.NumberOfControlPoints = 999  # deliberately mismatched from the file's actual data

    record = extract_beam(beam)

    assert record["number_of_control_points_declared"] == 999
    assert record["control_point_count_actual"] == 3


# =====================================================================
# Discovery
# =====================================================================

def test_discover_patients_sorts_numerically(tmp_path):
    for name in ["P10", "P2", "P1", "NotAPatient"]:
        os.makedirs(tmp_path / name)

    patients = discover_patients(str(tmp_path))

    assert patients == ["P1", "P2", "P10", "NotAPatient"]


def test_find_rtplan_files_recursive(tmp_path):
    nested = tmp_path / "RTP1"
    os.makedirs(nested)
    (nested / "plan.cdms").write_bytes(b"x")
    (tmp_path / "unrelated.csv").write_text("x")

    found = find_rtplan_files(str(tmp_path))

    assert len(found) == 1
    assert found[0].endswith("plan.cdms")


# =====================================================================
# End-to-end per-patient parsing
# =====================================================================

def test_parse_patient_rtplan_missing_folder_reports_missing_rtplan(tmp_path):
    record = parse_patient_rtplan("P1", str(tmp_path / "P1"))
    assert record.status == "missing_rtplan"


def test_parse_patient_rtplan_non_rtplan_modality(tmp_path):
    patient_dir = tmp_path / "P1"
    os.makedirs(patient_dir)

    ds = _make_rtplan_dataset()
    ds.Modality = "RTDOSE"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage, not RTPLAN
    _write_dicom_file(ds, str(patient_dir / "plan.cdms"))

    record = parse_patient_rtplan("P1", str(patient_dir))
    assert record.status == "not_rtplan"


def test_parse_patient_rtplan_ok(tmp_path):
    patient_dir = tmp_path / "P1"
    os.makedirs(patient_dir)

    ds = _make_rtplan_dataset(patient_id="XYZ", num_beams=1)
    _write_dicom_file(ds, str(patient_dir / "plan.cdms"))

    record = parse_patient_rtplan("P1", str(patient_dir))

    assert record.status == "ok"
    assert record.patient_info["PatientID"] == "XYZ"
    assert len(record.beams) == 1
    assert record.beams[0]["BeamName"] == "H01 HELICAL"
    assert record.fraction_groups[0]["NumberOfFractionsPlanned"] == 15


# =====================================================================
# Summary flattening + full pipeline
# =====================================================================

def test_build_summary_row_flattens_first_beam():
    ds = _make_rtplan_dataset(patient_id="XYZ", num_beams=1)

    record = {
        "status": "ok",
        "source_file": "some/path.cdms",
        "error": None,
        "patient_info": extract_patient_info(ds),
        "beams": [extract_beam(b) for b in ds.BeamSequence],
        "fraction_groups": extract_fraction_groups(ds),
    }

    row = build_summary_row("P1", record)

    assert row["patient"] == "P1"
    assert row["patient_id_dicom"] == "XYZ"
    assert row["num_beams"] == 1
    assert row["pitch"] == 0.2
    assert row["number_of_fractions_planned"] == 15
    assert row["total_beam_meterset_mu"] == 7.95


def test_build_rtplan_dataset_and_save_outputs(tmp_path):
    complete_dir = tmp_path / "Complete"
    patient_dir = complete_dir / "P1"
    os.makedirs(patient_dir)

    ds = _make_rtplan_dataset(patient_id="XYZ", num_beams=1)
    _write_dicom_file(ds, str(patient_dir / "plan.cdms"))

    dataset = build_rtplan_dataset(str(complete_dir))
    assert list(dataset.keys()) == ["P1"]
    assert dataset["P1"]["status"] == "ok"

    metadata_dir = tmp_path / "metadata"
    json_path = metadata_dir / "rtplan_raw.json"
    csv_path = metadata_dir / "rtplan_summary.csv"

    save_raw_json(dataset, str(json_path))
    save_summary_csv(build_summary_rows(dataset), str(csv_path))

    assert json_path.exists()
    with open(json_path) as f:
        reloaded = json.load(f)
    assert reloaded["P1"]["patient_info"]["PatientID"] == "XYZ"

    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["patient"] == "P1"


def test_run_writes_both_output_files(tmp_path):
    complete_dir = tmp_path / "Complete"
    patient_dir = complete_dir / "P1"
    os.makedirs(patient_dir)

    ds = _make_rtplan_dataset(patient_id="XYZ", num_beams=1)
    _write_dicom_file(ds, str(patient_dir / "plan.cdms"))

    metadata_dir = tmp_path / "metadata"

    dataset = run(complete_dir=str(complete_dir), metadata_dir=str(metadata_dir))

    assert dataset["P1"]["status"] == "ok"
    assert (metadata_dir / "rtplan_raw.json").exists()
    assert (metadata_dir / "rtplan_summary.csv").exists()

"""
Sanity checks for SlidingWindowDataset: sample shapes, channel
dimension, and residual-target correctness, plus the short-patient
zero-padding path.
"""

import numpy as np
import torch

from data.dataset import SlidingWindowDataset


def _make_patient(num_projections, width=542, seed=0):
    rng = np.random.RandomState(seed)
    planned = rng.rand(num_projections, width).astype(np.float32)
    detector = rng.rand(num_projections, width).astype(np.float32)
    return {"planned": planned, "detector": detector}


def test_sample_shapes_normal_patient():
    patient_data = {"P1": _make_patient(600)}

    dataset = SlidingWindowDataset(patient_data, window_size=256, stride=128)

    assert len(dataset) > 0

    planned, residual = dataset[0]

    assert planned.shape == (1, 256, 542)
    assert residual.shape == (1, 256, 542)
    assert planned.dtype == torch.float32
    assert residual.dtype == torch.float32


def test_residual_equals_detector_minus_planned():
    patient_data = {"P1": _make_patient(300, width=64)}

    dataset = SlidingWindowDataset(patient_data, window_size=64, stride=32)

    planned_t, residual_t = dataset[0]

    patient_id, start, padded = dataset.samples[0]
    raw_planned = patient_data[patient_id]["planned"][start:start + 64]
    raw_detector = patient_data[patient_id]["detector"][start:start + 64]

    expected_residual = raw_detector - raw_planned

    assert np.allclose(planned_t.squeeze(0).numpy(), raw_planned)
    assert np.allclose(residual_t.squeeze(0).numpy(), expected_residual)


def test_short_patient_is_zero_padded():
    short_len = 100
    window_size = 256

    patient_data = {"Pshort": _make_patient(short_len, width=64)}

    dataset = SlidingWindowDataset(
        patient_data, window_size=window_size, stride=128
    )

    # Short patients get exactly one (padded) sample.
    assert len(dataset) == 1

    planned, residual = dataset[0]

    assert planned.shape == (1, window_size, 64)
    assert residual.shape == (1, window_size, 64)

    # Padded rows (beyond the real data) should be exactly zero in both
    # planned and residual (since detector padding is also zero there).
    assert torch.all(planned[:, short_len:, :] == 0)
    assert torch.all(residual[:, short_len:, :] == 0)

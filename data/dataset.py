"""
`SlidingWindowDataset`, extracted verbatim (logic-wise) from
`unetablation.ipynb`.

Creates overlapping sliding-window samples over each patient's sinogram
for residual learning: the network input is the planned fluence window,
and the target is the residual (detector - planned) over that same
window.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import WINDOW_SIZE, STRIDE


class SlidingWindowDataset(Dataset):
    """
    Creates overlapping sliding-window samples.

    Input
    -----
    planned   : (N, W)
    detector  : (N, W)

    Network Input
    -------------
    planned

    Network Target
    --------------
    residual = detector - planned
    """

    def __init__(
        self,
        patient_data,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        dtype=torch.float32,
    ):
        super().__init__()

        self.patient_data = patient_data
        self.window_size = window_size
        self.stride = stride
        self.dtype = dtype

        self.samples = []

        for patient_id, patient in patient_data.items():
            planned = patient["planned"]
            detector = patient["detector"]

            assert planned.shape == detector.shape

            num_projections = planned.shape[0]

            if num_projections < self.window_size:
                self.samples.append((patient_id, 0, True))
                continue

            starts = list(
                range(0, num_projections - self.window_size + 1, self.stride)
            )

            last_start = num_projections - self.window_size

            if starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                self.samples.append((patient_id, start, False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, start, padded = self.samples[idx]

        patient = self.patient_data[patient_id]

        planned = patient["planned"]
        detector = patient["detector"]

        # Zero padding
        if padded:
            pad_rows = self.window_size - planned.shape[0]

            planned = np.pad(
                planned, ((0, pad_rows), (0, 0)), mode="constant"
            )

            detector = np.pad(
                detector, ((0, pad_rows), (0, 0)), mode="constant"
            )
        else:
            planned = planned[start:start + self.window_size]
            detector = detector[start:start + self.window_size]

        # Residual Target
        assert isinstance(planned, np.ndarray), type(planned)
        assert isinstance(detector, np.ndarray), type(detector)

        assert planned.dtype != object
        assert detector.dtype != object

        assert np.isfinite(planned).all(), "planned contains NaN/Inf"
        assert np.isfinite(detector).all(), "detector contains NaN/Inf"

        assert planned.shape == detector.shape

        residual = detector - planned

        # Convert to Tensor
        planned = torch.from_numpy(planned).to(self.dtype)
        residual = torch.from_numpy(residual).to(self.dtype)

        # Add Channel Dimension
        planned = planned.unsqueeze(0)
        residual = residual.unsqueeze(0)

        return planned, residual

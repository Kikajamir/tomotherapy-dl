"""
Sanity checks for the pure-function pieces of the preprocessing pipeline
(shift_image, normalize, coordinate system) and the training loss, none
of which require the actual TomoTherapy CSV data to test.
"""

import numpy as np
import torch

from preprocessing.build_dataset import (
    shift_image,
    normalize,
    build_coordinate_system,
)
from model.losses import WeightedHuberLoss


def test_shift_image_zero_is_noop():
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(shift_image(img, 0), img)


def test_shift_image_right_and_left_are_inverses_of_shape():
    img = np.arange(12, dtype=np.float32).reshape(3, 4)

    right = shift_image(img, 2)
    left = shift_image(img, -2)

    assert right.shape == img.shape
    assert left.shape == img.shape

    # Shifting right by 2 drops the last 2 original columns and zero-fills
    # the first 2.
    assert np.array_equal(right[:, 2:], img[:, :-2])
    assert np.array_equal(right[:, :2], np.zeros((3, 2)))


def test_normalize_scales_to_unit_max():
    img = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    normed = normalize(img)
    assert normed.max() == 1.0
    assert normed.min() == 0.0


def test_normalize_handles_all_zero_image():
    img = np.zeros((2, 2), dtype=np.float32)
    normed = normalize(img)
    assert np.array_equal(normed, img)


def test_coordinate_system_shapes_and_width():
    leaf_mm, detector_iso_mm, common_mm = build_coordinate_system()

    assert leaf_mm.shape == (64,)
    assert detector_iso_mm.shape == (640,)
    # Matches the notebook's printed "Output width : 542"
    assert len(common_mm) == 542


def test_weighted_huber_loss_is_zero_for_perfect_prediction():
    criterion = WeightedHuberLoss()
    target = torch.randn(4, 1, 16, 16)
    loss = criterion(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_weighted_huber_loss_is_positive_for_imperfect_prediction():
    criterion = WeightedHuberLoss()
    target = torch.zeros(4, 1, 16, 16)
    pred = torch.ones(4, 1, 16, 16) * 0.1
    loss = criterion(pred, target)
    assert loss.item() > 0

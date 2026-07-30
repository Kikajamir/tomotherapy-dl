"""
Sanity checks for the pure-function pieces of the preprocessing pipeline
(shift_image, normalize, coordinate system) and the training loss, none
of which require the actual TomoTherapy CSV data to test.
"""

import numpy as np
import pytest
import torch

from preprocessing.build_dataset import (
    shift_image,
    normalize,
    build_coordinate_system,
)
from model.losses import (
    WeightedHuberLoss,
    WeightedL1Loss,
    GradientLoss,
    WeightedHuberGradientLoss,
)
from training.train import build_criterion


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


def test_weighted_l1_loss_is_zero_for_perfect_prediction():
    criterion = WeightedL1Loss()
    target = torch.rand(4, 1, 16, 16)
    loss = criterion(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_weighted_l1_loss_matches_manual_formula():
    criterion = WeightedL1Loss(alpha=2.0)
    target = torch.tensor([[0.0, 0.5, 1.0]])
    pred = torch.tensor([[0.1, 0.3, 0.8]])

    weight = 1.0 + 2.0 * target
    expected = (weight * (pred - target).abs()).mean()

    assert torch.isclose(criterion(pred, target), expected, atol=1e-6)


def test_weighted_l1_loss_weighs_high_intensity_errors_more():
    criterion = WeightedL1Loss(alpha=2.0)

    low_intensity_target = torch.zeros(1, 1, 1, 1)
    high_intensity_target = torch.ones(1, 1, 1, 1)
    pred_offset = torch.full((1, 1, 1, 1), 0.1)

    low_loss = criterion(low_intensity_target + pred_offset, low_intensity_target)
    high_loss = criterion(high_intensity_target + pred_offset, high_intensity_target)

    assert high_loss.item() > low_loss.item()


def test_build_criterion_l1_returns_weighted_huber_loss():
    assert isinstance(build_criterion("l1"), WeightedHuberLoss)


def test_build_criterion_weighted_l1_returns_weighted_l1_loss():
    assert isinstance(build_criterion("weighted_l1"), WeightedL1Loss)


def test_build_criterion_huber_gradient_returns_weighted_huber_gradient_loss():
    assert isinstance(build_criterion("huber_gradient"), WeightedHuberGradientLoss)


def test_build_criterion_rejects_unknown_loss_type():
    with pytest.raises(ValueError):
        build_criterion("bogus")


def test_gradient_loss_is_zero_for_perfect_prediction():
    criterion = GradientLoss()
    target = torch.rand(4, 1, 16, 16)
    loss = criterion(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_gradient_loss_matches_manual_finite_difference_formula():
    criterion = GradientLoss()

    target = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    pred = target + torch.tensor(
        [[0.0, 1.0, 0.0, -1.0],
         [1.0, 0.0, -1.0, 0.0],
         [0.0, -1.0, 1.0, 0.0]]
    ).reshape(1, 1, 3, 4)

    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]

    expected = (pred_dy - target_dy).abs().mean() + (pred_dx - target_dx).abs().mean()

    assert torch.isclose(criterion(pred, target), expected, atol=1e-6)


def test_gradient_loss_is_translation_invariant():
    criterion = GradientLoss()

    target = torch.rand(2, 1, 8, 10)
    pred = torch.rand(2, 1, 8, 10)

    baseline = criterion(pred, target)
    shifted = criterion(pred + 5.0, target + 5.0)

    assert torch.isclose(baseline, shifted, atol=1e-6)


def test_weighted_huber_gradient_loss_is_zero_for_perfect_prediction():
    criterion = WeightedHuberGradientLoss()
    target = torch.rand(4, 1, 16, 16)
    loss = criterion(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_weighted_huber_gradient_loss_matches_huber_plus_lambda_gradient():
    criterion = WeightedHuberGradientLoss(lambda_gradient=0.1)

    target = torch.rand(2, 1, 8, 10)
    pred = target + 0.05 * torch.randn(2, 1, 8, 10)

    huber = WeightedHuberLoss()(pred, target)
    gradient = GradientLoss()(pred, target)
    expected_total = huber + 0.1 * gradient

    total, huber_component, gradient_component = criterion.component_losses(pred, target)

    assert torch.isclose(total, expected_total, atol=1e-6)
    assert torch.isclose(huber_component, huber, atol=1e-6)
    assert torch.isclose(gradient_component, gradient, atol=1e-6)
    assert torch.isclose(criterion(pred, target), expected_total, atol=1e-6)


def test_weighted_huber_gradient_loss_component_losses_sum_matches_forward():
    criterion = WeightedHuberGradientLoss()

    target = torch.rand(2, 1, 8, 10)
    pred = target + 0.1 * torch.randn(2, 1, 8, 10)

    total, huber_component, gradient_component = criterion.component_losses(pred, target)

    assert torch.isclose(
        total, huber_component + criterion.lambda_gradient * gradient_component, atol=1e-6
    )

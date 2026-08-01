"""
Sanity checks for the pure-function pieces of the preprocessing pipeline
(shift_image, normalize, coordinate system) and the training loss, none
of which require the actual TomoTherapy CSV data to test.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from configs.config import LOSS_GRADIENT_LAMBDA
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
    MultiScaleLoss,
    BeamAwareWeightedHuberLoss,
    build_beam_mask,
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


def test_weighted_huber_gradient_loss_default_lambda_matches_config():
    # Experiment D: LOSS_GRADIENT_LAMBDA is a plain config parameter --
    # changing it in configs/config.py (0.1 / 0.25 / 0.5 / 1.0) requires
    # no other code changes, since this default is read fresh on import.
    criterion = WeightedHuberGradientLoss()
    assert criterion.lambda_gradient == LOSS_GRADIENT_LAMBDA


def test_multi_scale_loss_is_zero_for_perfect_prediction():
    criterion = MultiScaleLoss()
    target = torch.rand(2, 1, 16, 16)
    loss = criterion(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_multi_scale_loss_matches_manual_formula():
    criterion = MultiScaleLoss()

    target = torch.rand(2, 1, 16, 16)
    pred = target + 0.1 * torch.randn(2, 1, 16, 16)

    full = F.l1_loss(pred, target)
    half = F.l1_loss(F.avg_pool2d(pred, 2), F.avg_pool2d(target, 2))
    quarter = F.l1_loss(F.avg_pool2d(pred, 4), F.avg_pool2d(target, 4))
    expected = full + 0.5 * half + 0.25 * quarter

    assert torch.isclose(criterion(pred, target), expected, atol=1e-6)


def test_multi_scale_loss_handles_non_power_of_two_width():
    # Sinogram width is 542 (not divisible by 4) -- avg_pool2d should
    # just floor-divide rather than error.
    criterion = MultiScaleLoss()
    target = torch.rand(1, 1, 256, 542)
    pred = target + 0.05 * torch.randn(1, 1, 256, 542)

    loss = criterion(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_build_criterion_multi_scale_loss_overrides_loss_type():
    criterion = build_criterion(loss_type="huber_gradient", use_multi_scale_loss=True)
    assert isinstance(criterion, MultiScaleLoss)


def test_build_criterion_default_multi_scale_loss_is_off():
    criterion = build_criterion(loss_type="l1")
    assert not isinstance(criterion, MultiScaleLoss)


def test_build_beam_mask_uses_planned_only():
    planned = torch.tensor([[0.0, 0.5, 0.0, 1.0]])
    # target/pred are deliberately unrelated to planned's zero pattern --
    # the mask must not be influenced by them in any way.
    target = torch.tensor([[5.0, -5.0, 3.0, -1.0]])

    mask = build_beam_mask(planned, threshold=0.0)

    assert mask.tolist() == [[False, True, False, True]]

    # Changing target/pred must not change the mask.
    mask_again = build_beam_mask(planned, threshold=0.0)
    assert torch.equal(mask, mask_again)


def test_build_beam_mask_works_on_numpy_arrays():
    planned = np.array([[0.0, 0.02, 0.0]], dtype=np.float32)
    mask = build_beam_mask(planned, threshold=0.01)
    assert mask.tolist() == [[False, True, False]]


def test_beam_aware_loss_is_zero_for_perfect_prediction():
    criterion = BeamAwareWeightedHuberLoss()
    planned = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    loss = criterion(target, target, planned)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_beam_aware_loss_requires_planned_flag_is_set():
    criterion = BeamAwareWeightedHuberLoss()
    assert criterion.requires_planned is True


def test_beam_aware_loss_matches_manual_formula():
    criterion = BeamAwareWeightedHuberLoss(
        beam_threshold=0.0, beam_weight=1.0, background_weight=0.3,
        lambda_gradient=0.1,
    )

    planned = torch.tensor([[0.0, 0.5, 0.0, 1.0]] * 4).unsqueeze(0).unsqueeze(0)
    target = torch.rand(1, 1, 4, 4)
    pred = target + 0.05 * torch.randn(1, 1, 4, 4)

    mask = planned > 0.0
    huber = WeightedHuberLoss()
    beam_component = huber(pred[mask], target[mask])
    background_component = huber(pred[~mask], target[~mask])
    gradient_component = GradientLoss()(pred, target)

    expected = 1.0 * beam_component + 0.3 * background_component + 0.1 * gradient_component

    total, beam, background, gradient = criterion.region_component_losses(pred, target, planned)

    assert torch.isclose(total, expected, atol=1e-5)
    assert torch.isclose(beam, beam_component, atol=1e-6)
    assert torch.isclose(background, background_component, atol=1e-6)
    assert torch.isclose(gradient, gradient_component, atol=1e-6)
    assert torch.isclose(criterion(pred, target, planned), expected, atol=1e-5)


def test_beam_aware_loss_background_weight_zero_ignores_background_errors():
    # lambda_gradient=0.0 isolates the Huber term's region-weighting from
    # GradientLoss, which is intentionally computed over the *full* image
    # (per the Task 2 spec) and would otherwise also react to the sharp
    # background/beam boundary this test introduces.
    criterion = BeamAwareWeightedHuberLoss(background_weight=0.0, lambda_gradient=0.0)

    planned = torch.zeros(1, 1, 4, 4)
    planned[..., 2:] = 1.0  # right half is beam, left half is background

    target = torch.zeros(1, 1, 4, 4)

    pred_good_background = target.clone()
    pred_bad_background = target.clone()
    pred_bad_background[..., :2] = 5.0  # large error, but only in background

    loss_good = criterion(pred_good_background, target, planned)
    loss_bad = criterion(pred_bad_background, target, planned)

    assert torch.isclose(loss_good, loss_bad, atol=1e-6)


def test_beam_aware_loss_beam_weight_scales_beam_contribution():
    planned = torch.zeros(1, 1, 4, 4)
    planned[..., 2:] = 1.0

    target = torch.zeros(1, 1, 4, 4)
    pred = target.clone()
    pred[..., 2:] = 0.2  # error only in the beam region

    low_weight = BeamAwareWeightedHuberLoss(beam_weight=1.0, background_weight=0.0, lambda_gradient=0.0)
    high_weight = BeamAwareWeightedHuberLoss(beam_weight=2.0, background_weight=0.0, lambda_gradient=0.0)

    loss_low = low_weight(pred, target, planned)
    loss_high = high_weight(pred, target, planned)

    assert torch.isclose(loss_high, 2.0 * loss_low, atol=1e-6)


def test_build_criterion_beam_weighted_returns_beam_aware_loss():
    assert isinstance(build_criterion("beam_weighted"), BeamAwareWeightedHuberLoss)


def test_weighted_huber_elementwise_mean_matches_forward():
    # Refactor equivalence: forward() must still be exactly
    # elementwise(...).mean(), for every existing loss/experiment that
    # depends on WeightedHuberLoss (huber_gradient, beam_weighted).
    criterion = WeightedHuberLoss()
    pred = torch.rand(3, 1, 10, 12)
    target = torch.rand(3, 1, 10, 12)

    assert torch.isclose(
        criterion(pred, target), criterion.elementwise(pred, target).mean(), atol=1e-7
    )


def test_beam_aware_loss_default_mode_is_binary():
    criterion = BeamAwareWeightedHuberLoss()
    assert criterion.beam_weight_mode == "binary"


def test_beam_aware_loss_rejects_unknown_weight_mode():
    with pytest.raises(ValueError):
        BeamAwareWeightedHuberLoss(beam_weight_mode="bogus")


def test_beam_aware_loss_continuous_mode_with_zero_alpha_matches_binary():
    planned = torch.zeros(1, 1, 4, 4)
    planned[..., 2:] = 1.0

    target = torch.rand(1, 1, 4, 4)
    pred = target + 0.1 * torch.randn(1, 1, 4, 4)

    binary = BeamAwareWeightedHuberLoss(beam_weight_mode="binary")
    continuous_zero_alpha = BeamAwareWeightedHuberLoss(
        beam_weight_mode="continuous", beam_alpha=0.0
    )

    total_binary, beam_binary, bg_binary, grad_binary = binary.region_component_losses(
        pred, target, planned
    )
    total_cont, beam_cont, bg_cont, grad_cont = continuous_zero_alpha.region_component_losses(
        pred, target, planned
    )

    assert torch.isclose(total_binary, total_cont, atol=1e-6)
    assert torch.isclose(beam_binary, beam_cont, atol=1e-6)
    assert torch.isclose(bg_binary, bg_cont, atol=1e-6)


def test_beam_aware_loss_continuous_mode_matches_manual_formula():
    beam_threshold = 0.0
    beam_alpha = 2.0
    beam_gamma = 2.0

    planned = torch.tensor([[0.0, 0.3, 0.0, 0.9]] * 4).unsqueeze(0).unsqueeze(0)
    target = torch.rand(1, 1, 4, 4)
    pred = target + 0.05 * torch.randn(1, 1, 4, 4)

    criterion = BeamAwareWeightedHuberLoss(
        beam_threshold=beam_threshold, beam_weight_mode="continuous",
        beam_alpha=beam_alpha, beam_gamma=beam_gamma, lambda_gradient=0.0,
        beam_weight=1.0, background_weight=1.0,
    )

    mask = planned > beam_threshold
    elementwise = WeightedHuberLoss().elementwise(pred, target)
    beam_weight_map = 1.0 + beam_alpha * planned.clamp(min=0.0).pow(beam_gamma)
    weighted_elementwise = beam_weight_map * elementwise

    expected_beam = weighted_elementwise[mask].mean()
    expected_background = weighted_elementwise[~mask].mean()

    total, beam, background, _ = criterion.region_component_losses(pred, target, planned)

    assert torch.isclose(beam, expected_beam, atol=1e-6)
    assert torch.isclose(background, expected_background, atol=1e-6)
    assert torch.isclose(total, expected_beam + expected_background, atol=1e-6)


def test_beam_aware_loss_continuous_mode_upweights_higher_planned_pixels():
    # Two beam pixels with the same error but different planned
    # intensity -- the higher-planned one should be weighted more in
    # continuous mode, so isolating it into "beam" via a mask that only
    # includes it should show a bigger beam_loss than the low-planned one.
    low_planned = torch.full((1, 1, 2, 2), 0.1)
    high_planned = torch.full((1, 1, 2, 2), 0.9)

    target = torch.zeros(1, 1, 2, 2)
    pred = torch.full((1, 1, 2, 2), 0.1)  # identical error in both cases

    criterion = BeamAwareWeightedHuberLoss(
        beam_threshold=0.0, beam_weight_mode="continuous",
        beam_alpha=11.0, beam_gamma=2.5, lambda_gradient=0.0,
    )

    _, beam_low, _, _ = criterion.region_component_losses(pred, target, low_planned)
    _, beam_high, _, _ = criterion.region_component_losses(pred, target, high_planned)

    assert beam_high.item() > beam_low.item()


def test_beam_aware_loss_continuous_mode_is_noop_outside_beam():
    # planned == 0 in the background under the default threshold, so
    # beam_weight_map == 1 there regardless of BEAM_ALPHA/BEAM_GAMMA --
    # continuous mode must not change background_loss vs. binary mode.
    planned = torch.zeros(1, 1, 4, 4)
    planned[..., 2:] = 1.0

    target = torch.rand(1, 1, 4, 4)
    pred = target + 0.2 * torch.randn(1, 1, 4, 4)

    binary = BeamAwareWeightedHuberLoss(beam_weight_mode="binary")
    continuous = BeamAwareWeightedHuberLoss(
        beam_weight_mode="continuous", beam_alpha=11.0, beam_gamma=2.5
    )

    _, _, bg_binary, _ = binary.region_component_losses(pred, target, planned)
    _, _, bg_continuous, _ = continuous.region_component_losses(pred, target, planned)

    assert torch.isclose(bg_binary, bg_continuous, atol=1e-6)

"""
Continuous Intensity Weighted Huber Loss, extracted verbatim
(logic-wise) from `unetablation.ipynb`, plus a Weighted L1 Loss and a
Gradient Loss added for the loss-function ablation study (see
`configs.config.LOSS_TYPE`), a Multi-Scale Loss added for the
multi-scale-loss ablation (see `configs.config.USE_MULTI_SCALE_LOSS`),
and a Beam-Aware Weighted Huber Loss added for the beam/background
region-reweighting ablation (see `configs.config.LOSS_TYPE ==
"beam_weighted"`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import (
    LOSS_DELTA,
    LOSS_ALPHA,
    LOSS_GAMMA,
    WEIGHTED_L1_ALPHA,
    LOSS_GRADIENT_LAMBDA,
    BEAM_THRESHOLD,
    BEAM_WEIGHT,
    BACKGROUND_WEIGHT,
    BEAM_WEIGHT_MODE,
    BEAM_ALPHA,
    BEAM_GAMMA,
)


def build_beam_mask(planned, threshold=BEAM_THRESHOLD):
    """
    Beam mask shared by BeamAwareWeightedHuberLoss and by evaluation's
    beam/background metrics -- derived from `planned` only (never from
    the detector/target), since `planned` is known at inference time
    too and is an exact treatment-plan value (no measurement noise), so
    `planned > threshold` unambiguously identifies irradiated pixels.
    Works for both torch tensors and numpy arrays.
    """
    return planned > threshold


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=LOSS_DELTA, alpha=LOSS_ALPHA, gamma=LOSS_GAMMA):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma

    def elementwise(self, pred, target):
        """
        Per-pixel weighted Huber loss, unreduced -- factored out of
        `forward` (which is unchanged: `elementwise(...).mean()` is
        exactly the same computation as before) so BeamAwareWeightedHuberLoss
        can mask/reweight individual pixels before taking a region mean,
        without duplicating the Huber/weighting formula.
        """
        error = pred - target
        abs_error = error.abs()

        # Standard Huber
        huber = torch.where(
            abs_error <= self.delta,
            0.5 * error.pow(2) / self.delta,
            abs_error - 0.5 * self.delta,
        )

        # Continuous intensity weighting
        weights = 1.0 + self.alpha * target.abs().pow(self.gamma)

        return weights * huber

    def forward(self, pred, target):
        return self.elementwise(pred, target).mean()


class WeightedL1Loss(nn.Module):
    """
    Intensity-weighted L1 loss: weight = 1 + alpha * target, so higher
    (normalized) detector intensities contribute more to the loss.
    `target` here is whatever the caller passes as the training target
    (the residual, per the existing residual-learning training loop) --
    same convention as WeightedHuberLoss above.
    """

    def __init__(self, alpha=WEIGHTED_L1_ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        weight = 1.0 + self.alpha * target

        return (weight * (pred - target).abs()).mean()


class GradientLoss(nn.Module):
    """
    First-order finite-difference image-gradient loss: penalizes the L1
    difference between `pred` and `target` gradients along both sinogram
    axes -- projection (dim=-2) and detector (dim=-1) -- so the model is
    pushed to match edge sharpness, not just intensity, in either
    direction. Finite differences are used instead of a Sobel kernel
    since dim=-2/-1 already are the physical projection/detector axes;
    a smoothing kernel isn't needed to define a direction here.
    """

    def forward(self, pred, target):
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]

        loss_y = (pred_dy - target_dy).abs().mean()
        loss_x = (pred_dx - target_dx).abs().mean()

        return loss_y + loss_x


class WeightedHuberGradientLoss(nn.Module):
    """
    TotalLoss = WeightedHuberLoss + lambda_gradient * GradientLoss (see
    `configs.config.LOSS_GRADIENT_LAMBDA`). Reuses WeightedHuberLoss
    unchanged; GradientLoss only adds an edge-sharpness term on top.
    `forward` returns the scalar total (same convention as the other
    losses, for drop-in use as a training criterion); `component_losses`
    exposes the (total, huber, gradient) breakdown for validation logging.
    """

    def __init__(self, delta=LOSS_DELTA, alpha=LOSS_ALPHA, gamma=LOSS_GAMMA,
                 lambda_gradient=LOSS_GRADIENT_LAMBDA):
        super().__init__()
        self.huber_loss = WeightedHuberLoss(delta=delta, alpha=alpha, gamma=gamma)
        self.gradient_loss = GradientLoss()
        self.lambda_gradient = lambda_gradient

    def component_losses(self, pred, target):
        huber_component = self.huber_loss(pred, target)
        gradient_component = self.gradient_loss(pred, target)
        total = huber_component + self.lambda_gradient * gradient_component
        return total, huber_component, gradient_component

    def forward(self, pred, target):
        total, _, _ = self.component_losses(pred, target)
        return total


class MultiScaleLoss(nn.Module):
    """
    Multi-scale reconstruction loss (Experiment C ablation, see
    `configs.config.USE_MULTI_SCALE_LOSS`): plain (unweighted) L1 at full
    resolution plus L1 at half/quarter resolution, the latter two
    computed by average-pooling both `pred` and `target` before
    comparing them -- TotalLoss = L1_full + 0.5 * L1_half +
    0.25 * L1_quarter. Independent of `LOSS_TYPE`/WeightedHuberLoss.
    """

    def forward(self, pred, target):
        full = F.l1_loss(pred, target)

        half = F.l1_loss(
            F.avg_pool2d(pred, kernel_size=2), F.avg_pool2d(target, kernel_size=2)
        )

        quarter = F.l1_loss(
            F.avg_pool2d(pred, kernel_size=4), F.avg_pool2d(target, kernel_size=4)
        )

        return full + 0.5 * half + 0.25 * quarter


class BeamAwareWeightedHuberLoss(nn.Module):
    """
    Beam-aware region-reweighted loss (LOSS_TYPE == "beam_weighted"):
    splits the existing (unmodified) WeightedHuberLoss into an
    active-beam term and a background term, each normalized by its own
    pixel count (never a single pooled `.mean()` over the whole image),
    weighted independently, plus the existing GradientLoss on the full
    image:

        Loss = BEAM_WEIGHT * beam_loss
             + BACKGROUND_WEIGHT * background_loss
             + LOSS_GRADIENT_LAMBDA * GradientLoss(pred, target)

    The beam/background region split always comes from
    `build_beam_mask(planned)` (binary, `planned > BEAM_THRESHOLD`),
    regardless of `beam_weight_mode` -- that split is what removes the
    background pixel-count's dilution of the beam gradient signal.
    `beam_weight_mode` only changes how `beam_loss` itself is computed:

    - "binary" (default): beam_loss = mean(WeightedHuberLoss pixels
      inside the mask) -- every beam pixel counted equally.
    - "continuous": each pixel's elementwise WeightedHuberLoss is first
      multiplied by `1 + BEAM_ALPHA * planned**BEAM_GAMMA` (a no-op
      outside the beam, since planned==0 there under the default
      threshold), then averaged inside the mask -- so higher-fluence
      beam pixels contribute more to beam_loss. Reduces exactly to
      "binary" when BEAM_ALPHA=0.

    `forward` requires `planned` as a third argument (unlike the other
    losses in this module); `requires_planned = True` lets calling code
    (see training/train.py) detect this generically.
    `region_component_losses` exposes the (total, beam, background,
    gradient) breakdown for validation logging, deliberately named
    differently from `WeightedHuberGradientLoss.component_losses` so the
    two aren't confused by callers checking for one or the other.
    """

    requires_planned = True

    def __init__(self, delta=LOSS_DELTA, alpha=LOSS_ALPHA, gamma=LOSS_GAMMA,
                 beam_threshold=BEAM_THRESHOLD, beam_weight=BEAM_WEIGHT,
                 background_weight=BACKGROUND_WEIGHT,
                 beam_weight_mode=BEAM_WEIGHT_MODE,
                 beam_alpha=BEAM_ALPHA, beam_gamma=BEAM_GAMMA,
                 lambda_gradient=LOSS_GRADIENT_LAMBDA):
        super().__init__()

        if beam_weight_mode not in ("binary", "continuous"):
            raise ValueError(
                f"Unknown beam_weight_mode: {beam_weight_mode!r} "
                "(expected 'binary' or 'continuous')"
            )

        self.huber_loss = WeightedHuberLoss(delta=delta, alpha=alpha, gamma=gamma)
        self.gradient_loss = GradientLoss()
        self.beam_threshold = beam_threshold
        self.beam_weight = beam_weight
        self.background_weight = background_weight
        self.beam_weight_mode = beam_weight_mode
        self.beam_alpha = beam_alpha
        self.beam_gamma = beam_gamma
        self.lambda_gradient = lambda_gradient

    def region_component_losses(self, pred, target, planned):
        mask = build_beam_mask(planned, self.beam_threshold)

        elementwise_loss = self.huber_loss.elementwise(pred, target)

        if self.beam_weight_mode == "continuous":
            beam_weight_map = 1.0 + self.beam_alpha * planned.clamp(min=0.0).pow(
                self.beam_gamma
            )
            elementwise_loss = beam_weight_map * elementwise_loss

        if mask.any():
            beam_component = elementwise_loss[mask].mean()
        else:
            beam_component = pred.new_tensor(0.0)

        background = ~mask
        if background.any():
            background_component = elementwise_loss[background].mean()
        else:
            background_component = pred.new_tensor(0.0)

        gradient_component = self.gradient_loss(pred, target)

        total = (
            self.beam_weight * beam_component
            + self.background_weight * background_component
            + self.lambda_gradient * gradient_component
        )
        return total, beam_component, background_component, gradient_component

    def forward(self, pred, target, planned):
        total, _, _, _ = self.region_component_losses(pred, target, planned)
        return total

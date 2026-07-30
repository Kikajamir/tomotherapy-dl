"""
Continuous Intensity Weighted Huber Loss, extracted verbatim
(logic-wise) from `unetablation.ipynb`, plus a Weighted L1 Loss and a
Gradient Loss added for the loss-function ablation study (see
`configs.config.LOSS_TYPE`).
"""

import torch
import torch.nn as nn

from configs.config import (
    LOSS_DELTA,
    LOSS_ALPHA,
    LOSS_GAMMA,
    WEIGHTED_L1_ALPHA,
    LOSS_GRADIENT_LAMBDA,
)


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=LOSS_DELTA, alpha=LOSS_ALPHA, gamma=LOSS_GAMMA):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
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

        loss = weights * huber

        return loss.mean()


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

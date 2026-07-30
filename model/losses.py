"""
Continuous Intensity Weighted Huber Loss, extracted verbatim
(logic-wise) from `unetablation.ipynb`, plus a Weighted L1 Loss added
for the loss-function ablation study (see `configs.config.LOSS_TYPE`).
"""

import torch
import torch.nn as nn

from configs.config import LOSS_DELTA, LOSS_ALPHA, LOSS_GAMMA, WEIGHTED_L1_ALPHA


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

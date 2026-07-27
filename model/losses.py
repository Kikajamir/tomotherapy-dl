"""
Continuous Intensity Weighted Huber Loss, extracted verbatim
(logic-wise) from `unetablation.ipynb`.
"""

import torch
import torch.nn as nn

from configs.config import LOSS_DELTA, LOSS_ALPHA, LOSS_GAMMA


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

"""
Building blocks of the Attention Residual U-Net generator, extracted
verbatim (logic-wise) from `unetablation.ipynb`.
"""

import torch
import torch.nn as nn

from configs.config import GROUPS


class ConvBlock(nn.Module):
    """
    Double Convolution Block

    Conv3x3
        |
    GroupNorm
        |
    PReLU
        |
    Conv3x3
        |
    GroupNorm
        |
    PReLU
    """

    def __init__(self, in_channels, out_channels, groups=GROUPS):
        super().__init__()

        num_groups = min(groups, out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.PReLU(out_channels),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Standard residual block.

           Input
             |
          ConvBlock
             |
      Residual Addition
             |
           Output
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = ConvBlock(in_channels=channels, out_channels=channels)

    def forward(self, x):
        return x + self.conv(x)


class EncoderBlock(nn.Module):
    """
    Feature extraction block.

    Input
        |
    ConvBlock
        |
    Skip Feature
    """

    def __init__(self, in_channels, out_channels, groups=GROUPS):
        super().__init__()

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)


class DownsampleBlock(nn.Module):
    """
    Learnable downsampling using stride-2 convolution.

    Input
        |
    Conv3x3 (stride=2)
        |
    GroupNorm
        |
    PReLU
    """

    def __init__(self, in_channels, out_channels, groups=GROUPS):
        super().__init__()

        num_groups = min(groups, out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate.

    Inputs
    ------
    x : Skip connection feature
    g : Decoder gating feature

    Output
    ------
    Filtered skip feature
    """

    def __init__(self, skip_channels, gate_channels, inter_channels):
        super().__init__()

        self.theta = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(GROUPS, inter_channels),
                num_channels=inter_channels,
            ),
        )

        self.phi = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(GROUPS, inter_channels),
                num_channels=inter_channels,
            ),
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        attention = self.psi(self.theta(x) + self.phi(g))
        return x * attention


class DecoderBlock(nn.Module):
    """
    Decoder Block

    Upsample
        |
    Attention Gate (optional)
        |
    Concatenate Skip
        |
    ConvBlock
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_attention=True,
        groups=GROUPS,
    ):
        super().__init__()

        self.use_attention = use_attention

        # Upsampling
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(groups, out_channels),
                num_channels=out_channels,
            ),
            nn.PReLU(out_channels),
        )

        # Attention Gate
        if use_attention:
            self.attention = AttentionGate(
                skip_channels=skip_channels,
                gate_channels=out_channels,
                inter_channels=max(out_channels // 2, 1),
            )

        # Feature Fusion
        self.conv = ConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            groups=groups,
        )

    def forward(self, x, skip):
        x = self.up(x)

        if self.use_attention:
            skip = self.attention(skip, x)

        x = torch.cat([skip, x], dim=1)

        x = self.conv(x)

        return x


def initialize_weights(model):
    """
    Weight initialization.

    Conv2d / ConvTranspose2d:
        Kaiming Normal (fan_in)

    GroupNorm:
        weight = 1
        bias = 0

    Linear:
        Kaiming Normal
    """
    for m in model.modules():
        # Convolution
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # GroupNorm
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        # Linear
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="linear"
            )

            if m.bias is not None:
                nn.init.zeros_(m.bias)

"""
Attention Residual U-Net generator, extracted verbatim (logic-wise) from
`unetablation.ipynb`.
"""

import torch.nn as nn

from configs.config import (
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    BASE_CHANNELS,
    NUM_RES_BLOCKS,
    PAD_WIDTH,
)
from model.blocks import (
    ConvBlock,
    ResidualBlock,
    EncoderBlock,
    DownsampleBlock,
    DecoderBlock,
)


class Generator(nn.Module):
    """
    Attention Residual U-Net Generator

    Input (1x256x542)
            |
    ReflectionPad2d
            |
    Encoder
            |
    Residual Bottleneck
            |
    Decoder
            |
    Output Conv
            |
    Center Crop
            |
    Output (1x256x542)
    """

    def __init__(
        self,
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        num_residual_blocks=NUM_RES_BLOCKS,
    ):
        super().__init__()

        # Reflection Padding
        # Width: 542 -> 544
        self.pad = nn.ReflectionPad2d((1, 1, 0, 0))

        # ================================================
        # Encoder
        # ================================================
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)

        self.enc2 = EncoderBlock(base_channels * 2, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)

        self.enc3 = EncoderBlock(base_channels * 4, base_channels * 4)
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8)

        self.enc4 = EncoderBlock(base_channels * 8, base_channels * 8)
        self.down4 = DownsampleBlock(base_channels * 8, base_channels * 16)

        # ================================================
        # Bottleneck
        # ================================================
        self.bottleneck = ConvBlock(base_channels * 16, base_channels * 16)

        self.residual = nn.Sequential(
            *[
                ResidualBlock(base_channels * 16)
                for _ in range(num_residual_blocks)
            ]
        )

        # ================================================
        # Decoder
        # ================================================
        self.dec4 = DecoderBlock(
            in_channels=base_channels * 16,
            skip_channels=base_channels * 8,
            out_channels=base_channels * 8,
            use_attention=True,
        )

        self.dec3 = DecoderBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            use_attention=True,
        )

        self.dec2 = DecoderBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            use_attention=False,
        )

        self.dec1 = DecoderBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            use_attention=False,
        )

        # ================================================
        # Output Layer
        # ================================================
        self.output = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        # Save original size
        original_width = x.shape[-1]

        # Reflection Padding
        x = self.pad(x)

        # ================================================
        # Encoder
        # ================================================
        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        x = self.down2(s2)

        s3 = self.enc3(x)
        x = self.down3(s3)

        s4 = self.enc4(x)
        x = self.down4(s4)

        # ================================================
        # Bottleneck
        # ================================================
        x = self.bottleneck(x)
        x = self.residual(x)

        # ================================================
        # Decoder
        # ================================================
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # ================================================
        # Output
        # ================================================
        x = self.output(x)

        # Remove reflection padding
        x = x[..., PAD_WIDTH:PAD_WIDTH + original_width]

        return x

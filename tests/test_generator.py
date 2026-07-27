"""
Sanity checks for the model components: verifies tensor shapes through
the individual blocks and the full Generator, matching the shape checks
that were run ad-hoc in `unetablation.ipynb`.
"""

import torch

from model.blocks import (
    ConvBlock,
    ResidualBlock,
    EncoderBlock,
    DownsampleBlock,
    AttentionGate,
    DecoderBlock,
)
from model.generator import Generator


def test_conv_block_shape():
    x = torch.randn(2, 1, 256, 544)
    block = ConvBlock(in_channels=1, out_channels=64)
    y = block(x)
    assert y.shape == (2, 64, 256, 544)


def test_residual_block_shape():
    x = torch.randn(2, 512, 16, 34)
    block = ResidualBlock(512)
    y = block(x)
    assert y.shape == x.shape


def test_encoder_downsample_shape():
    encoder = EncoderBlock(1, 64)
    down = DownsampleBlock(64, 128)

    x = torch.randn(2, 1, 256, 544)

    skip = encoder(x)
    y = down(skip)

    assert skip.shape == (2, 64, 256, 544)
    assert y.shape == (2, 128, 128, 272)


def test_attention_gate_shape():
    skip = torch.randn(2, 256, 64, 136)
    gate = torch.randn(2, 256, 64, 136)

    att = AttentionGate(skip_channels=256, gate_channels=256, inter_channels=128)
    out = att(skip, gate)

    assert out.shape == skip.shape


def test_decoder_block_shape():
    x = torch.randn(2, 1024, 16, 34)
    skip = torch.randn(2, 512, 32, 68)

    decoder = DecoderBlock(
        in_channels=1024,
        skip_channels=512,
        out_channels=512,
        use_attention=True,
    )

    y = decoder(x, skip)

    assert y.shape == (2, 512, 32, 68)


def test_generator_output_shape_matches_input():
    model = Generator()

    x = torch.randn(2, 1, 256, 542)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape == (2, 1, 256, 542)


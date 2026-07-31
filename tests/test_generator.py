"""
Sanity checks for the model components: verifies tensor shapes through
the individual blocks and the full Generator, matching the shape checks
that were run ad-hoc in `unetablation.ipynb`. Also covers the Sigmoid
Output and Deep Supervision ablations (both off by default).
"""

import pytest
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


def test_generator_default_output_activation_is_linear_and_unbounded():
    model = Generator()
    assert model.output_activation == "linear"
    assert isinstance(model.activation, torch.nn.Identity)

    x = torch.randn(1, 1, 256, 542) * 100

    with torch.no_grad():
        y = model(x)

    # A linear (Identity) output should not be squashed into [0, 1].
    assert y.abs().max() > 1.0


def test_generator_sigmoid_output_activation_bounds_to_unit_interval():
    model = Generator(output_activation="sigmoid")
    assert isinstance(model.activation, torch.nn.Sigmoid)

    x = torch.randn(2, 1, 256, 542) * 100

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1, 256, 542)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_generator_rejects_unknown_output_activation():
    with pytest.raises(ValueError):
        Generator(output_activation="bogus")


def test_generator_deep_supervision_off_by_default():
    model = Generator()
    assert model.use_deep_supervision is False
    assert model.aux_outputs is None

    x = torch.randn(1, 1, 256, 542)
    with torch.no_grad():
        model(x)

    assert model.aux_outputs is None


def test_generator_deep_supervision_aux_outputs_only_populated_in_train_mode():
    model = Generator(use_deep_supervision=True)
    x = torch.randn(2, 1, 256, 542)

    model.train()
    y = model(x)

    assert y.shape == (2, 1, 256, 542)
    assert model.aux_outputs is not None

    aux2, aux3 = model.aux_outputs
    # dec2 runs at half resolution (in padded 544-width space), dec3 at
    # quarter resolution -- see model/generator.py's decoder stage sizes.
    assert aux2.shape == (2, 1, 128, 272)
    assert aux3.shape == (2, 1, 64, 136)

    model.eval()
    with torch.no_grad():
        y_eval = model(x)

    assert y_eval.shape == (2, 1, 256, 542)
    assert model.aux_outputs is None


def test_generator_deep_supervision_does_not_change_final_output_shape():
    model = Generator(use_deep_supervision=True)
    x = torch.randn(1, 1, 256, 542)

    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 256, 542)


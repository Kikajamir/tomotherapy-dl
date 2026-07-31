"""
Lightweight integration checks that the new ablation flags actually wire
through `training.train.train()` end to end, using a tiny synthetic
Generator/dataset (small base_channels/residual blocks, 1-2 samples, 1
epoch, CPU) so this stays fast -- no real patient data, no GPU.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.generator import Generator
from model.losses import WeightedHuberLoss
from training.train import build_optimizer, train


def _tiny_loader(num_samples=2, batch_size=2):
    planned = torch.rand(num_samples, 1, 256, 542)
    residual = torch.rand(num_samples, 1, 256, 542) * 0.1
    dataset = TensorDataset(planned, residual)
    return DataLoader(dataset, batch_size=batch_size)


def test_train_runs_one_epoch_with_deep_supervision_enabled(tmp_path):
    generator = Generator(
        base_channels=4, num_residual_blocks=1, use_deep_supervision=True
    )
    optimizer = build_optimizer(generator)
    criterion = WeightedHuberLoss()

    train_loader = _tiny_loader()
    val_loader = _tiny_loader()

    save_path = str(tmp_path / "checkpoint.pth")

    train_history, val_history, val_mae_history = train(
        generator,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        num_epochs=1,
        early_stopping_patience=1,
        save_path=save_path,
        use_deep_supervision=True,
    )

    assert len(train_history) == 1
    assert len(val_history) == 1
    assert (tmp_path / "checkpoint.pth").exists()


def test_train_runs_one_epoch_with_deep_supervision_disabled(tmp_path):
    generator = Generator(
        base_channels=4, num_residual_blocks=1, use_deep_supervision=False
    )
    optimizer = build_optimizer(generator)
    criterion = WeightedHuberLoss()

    train_loader = _tiny_loader()
    val_loader = _tiny_loader()

    save_path = str(tmp_path / "checkpoint.pth")

    train_history, val_history, val_mae_history = train(
        generator,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device=torch.device("cpu"),
        num_epochs=1,
        early_stopping_patience=1,
        save_path=save_path,
        use_deep_supervision=False,
    )

    assert len(train_history) == 1
    assert generator.aux_outputs is None

"""
Training pipeline extracted verbatim (logic-wise) from
`unetablation.ipynb`: patient-wise split, dataset/dataloader
construction, model + optimizer + loss setup, and the AMP training loop
with early stopping and best-checkpoint saving.
"""

import pickle

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from configs.config import (
    DEVICE,
    DATA_PATH,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    EXCLUDED_PATIENTS,
    WINDOW_SIZE,
    STRIDE,
    BATCH_SIZE,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    BASE_CHANNELS,
    NUM_RES_BLOCKS,
    LEARNING_RATE,
    BETAS,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SAVE_PATH,
    LOSS_TYPE,
)
from data.dataset import SlidingWindowDataset
from model.blocks import initialize_weights
from model.generator import Generator
from model.losses import WeightedHuberLoss, WeightedL1Loss


# =====================================================================
# Patient-wise Train / Validation / Test Split
# =====================================================================

def load_and_split_patients(data_path=DATA_PATH):
    """
    Loads processed_data.pkl, removes excluded (TomoDirect) patients, and
    performs a patient-wise train/val/test split identical to the
    notebook (70/15/15, random_state=RANDOM_SEED, shuffled).

    Returns (train_data, val_data, test_data) dicts.
    """
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    with open(data_path, "rb") as f:
        processed_data = pickle.load(f)

    print(f"Loaded {len(processed_data)} patients.")

    processed_data = {
        pid: data
        for pid, data in processed_data.items()
        if pid not in EXCLUDED_PATIENTS
    }

    print(f"Remaining Helical Patients : {len(processed_data)}")

    patient_ids = sorted(processed_data.keys())

    train_ids, temp_ids = train_test_split(
        patient_ids,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    val_fraction = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_fraction,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_data = {pid: processed_data[pid] for pid in train_ids}
    val_data = {pid: processed_data[pid] for pid in val_ids}
    test_data = {pid: processed_data[pid] for pid in test_ids}

    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0

    assert not (set(train_ids) & EXCLUDED_PATIENTS)
    assert not (set(val_ids) & EXCLUDED_PATIENTS)
    assert not (set(test_ids) & EXCLUDED_PATIENTS)

    print(f"Excluded Patients : {sorted(EXCLUDED_PATIENTS)}")
    print(f"Total Helical Patients : {len(patient_ids)}")
    print(f"Train Patients : {len(train_ids)}")
    print(f"Val Patients   : {len(val_ids)}")
    print(f"Test Patients  : {len(test_ids)}")
    print("No patient leakage detected.")

    return train_data, val_data, test_data


# =====================================================================
# Datasets / DataLoaders
# =====================================================================

def build_dataloaders(train_data, val_data, test_data,
                       window_size=WINDOW_SIZE, stride=STRIDE,
                       batch_size=BATCH_SIZE):
    train_dataset = SlidingWindowDataset(
        train_data, window_size=window_size, stride=stride
    )
    val_dataset = SlidingWindowDataset(
        val_data, window_size=window_size, stride=stride
    )
    test_dataset = SlidingWindowDataset(
        test_data, window_size=window_size, stride=stride
    )

    print(f"Train Samples : {len(train_dataset)}")
    print(f"Val Samples   : {len(val_dataset)}")
    print(f"Test Samples  : {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# =====================================================================
# Model / Optimizer
# =====================================================================

def build_generator(device=DEVICE):
    generator = Generator(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
        num_residual_blocks=NUM_RES_BLOCKS,
    ).to(device)

    initialize_weights(generator)

    return generator


def build_optimizer(generator, learning_rate=LEARNING_RATE, betas=BETAS):
    return torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)


def build_criterion(loss_type=LOSS_TYPE):
    """
    Loss-function ablation switch: "l1" keeps the existing baseline loss
    (WeightedHuberLoss, unchanged); "weighted_l1" selects the new
    WeightedL1Loss. Nothing else about the training pipeline changes.
    """
    if loss_type == "l1":
        return WeightedHuberLoss()
    elif loss_type == "weighted_l1":
        return WeightedL1Loss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r} (expected 'l1' or 'weighted_l1')")


# =====================================================================
# Training Loop
# =====================================================================

def train(
    generator,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device=DEVICE,
    num_epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    save_path=SAVE_PATH,
):
    """
    Runs the AMP training loop with early stopping and best-checkpoint
    saving, exactly as in the notebook. Returns (train_history,
    val_history).
    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_history = []
    val_history = []
    val_mae_history = []

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(num_epochs):
        generator.train()

        running_train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{num_epochs}] Training",
            leave=False,
        )

        for planned, residual in train_bar:
            planned = planned.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                predicted_residual = generator(planned)
                loss = criterion(predicted_residual, residual)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            train_bar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_train_loss = running_train_loss / len(train_loader)

        # Validation
        generator.eval()

        running_val_loss = 0.0
        running_val_mae = 0.0

        with torch.no_grad():
            for planned, residual in val_loader:
                planned = planned.to(device, non_blocking=True)
                residual = residual.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    predicted_residual = generator(planned)
                    loss = criterion(predicted_residual, residual)
                    mae = (predicted_residual - residual).abs().mean()

                running_val_loss += loss.item()
                running_val_mae += mae.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_mae = running_val_mae / len(val_loader)

        train_history.append(epoch_train_loss)
        val_history.append(epoch_val_loss)
        val_mae_history.append(epoch_val_mae)

        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0

            torch.save(generator.state_dict(), save_path)

            model_status = "Best"
        else:
            epochs_without_improvement += 1
            model_status = ""

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train {epoch_train_loss:.6f} | "
            f"Val (weighted) {epoch_val_loss:.6f} | "
            f"Val MAE (unweighted) {epoch_val_mae:.6f} | "
            f"LR {current_lr:.2e} "
            f"{model_status}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training Complete.")
    print(f"Best Validation Loss : {best_val_loss:.6f}")
    print(f"Best Model Saved To  : {save_path}")
    print(f"Epochs Completed     : {len(train_history)}")

    return train_history, val_history, val_mae_history


# =====================================================================
# Entry point
# =====================================================================

def main():
    train_data, val_data, test_data = load_and_split_patients()

    train_loader, val_loader, test_loader = build_dataloaders(
        train_data, val_data, test_data
    )

    generator = build_generator()
    optimizer = build_optimizer(generator)
    criterion = build_criterion()

    train_history, val_history, val_mae_history = train(
        generator, optimizer, criterion, train_loader, val_loader
    )

    return generator, train_history, val_history, (train_data, val_data, test_data)


if __name__ == "__main__":
    main()

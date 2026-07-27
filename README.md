# TomoQA

Attention Residual U-Net for TomoTherapy delivered-dose (detector)
prediction from planned fluence, via residual learning on sinograms.

This project is a modular extraction of two original Kaggle notebooks:

- `pix2pix.ipynb` — preprocesses raw TomoTherapy CSV exports into
  `processed_data.pkl`.
- `unetablation.ipynb` — trains and evaluates an Attention Residual
  U-Net generator on that dataset.

The notebooks are kept as thin wrappers around the modules below; no
algorithm, tensor shape, or training behavior was changed during the
extraction.

## Project layout

```
configs/
  config.py           All constants shared across the project (paths,
                       geometry, model, training, evaluation).
preprocessing/
  build_dataset.py     Raw CSVs -> processed_data.pkl.
data/
  dataset.py           SlidingWindowDataset (residual-learning samples).
model/
  blocks.py            ConvBlock, ResidualBlock, EncoderBlock,
                       DownsampleBlock, AttentionGate, DecoderBlock,
                       initialize_weights.
  generator.py         Generator (Attention Residual U-Net).
  losses.py            WeightedHuberLoss.
training/
  train.py             Patient-wise split, dataloaders, training loop.
evaluation/
  evaluate.py           Checkpoint loading, sliding-window
                       reconstruction, metrics, plots, saving.
tests/                 Sanity checks (shapes, residual math, losses).
pix2pix.ipynb          Thin wrapper -> preprocessing.build_dataset
unetablation.ipynb     Thin wrapper -> training.train + evaluation.evaluate
```

## Setup

```bash
pip install -r requirements.txt
```

The code was written to run wherever the original notebooks ran
(Kaggle, with the default paths in `configs/config.py` pointing at
`/kaggle/input/...`). To run it elsewhere, edit `BASE_PATH` and
`DATA_PATH` in `configs/config.py` to point at your local data.

## 1. Preprocessing

Turns the raw per-patient `planned.csv` / `D1.csv` files into
`processed_data.pkl` (planned + detector sinograms resampled onto a
shared coordinate axis).

```bash
python -m preprocessing.build_dataset
```

or from Python:

```python
from preprocessing.build_dataset import build_dataset
processed_data = build_dataset()
```

This writes `processed_data.pkl` to the current working directory
(`SAVE_FILE` in `configs/config.py`).

## 2. Training

Loads `processed_data.pkl`, performs the patient-wise 70/15/15
train/val/test split (patients `P6` and `P25` excluded as TomoDirect),
builds the sliding-window datasets/dataloaders, and trains the
Generator with the Weighted Huber Loss, Adam optimizer, AMP, and early
stopping.

```bash
python -m training.train
```

or from Python:

```python
from training.train import main
generator, train_history, val_history, (train_data, val_data, test_data) = main()
```

The best checkpoint (by validation loss) is saved to
`best_generator_residual.pth` (`SAVE_PATH` in `configs/config.py`).

## 3. Evaluation

Loads the best checkpoint, reconstructs full test-patient sinograms via
overlapping sliding-window inference (finer stride than training, per
the original notebook), computes per-patient MAE/RMSE/PSNR/SSIM, and
saves everything to `evaluation_results/`.

```bash
python -m evaluation.evaluate
```

or from Python (e.g. reusing a `test_data` split from training):

```python
from evaluation.evaluate import evaluate
reconstructed_results, metrics_df, summary = evaluate(test_data)
```

Diagnostic plots from the original notebook are available as
individual functions in `evaluation/evaluate.py`, e.g.:

```python
from evaluation.evaluate import plot_sinogram_comparison
plot_sinogram_comparison(reconstructed_results, patient_id="P23")
```

## Tests

```bash
pytest
```

Covers: generator/block output shapes, dataset sample shapes and
residual-target correctness (including the short-patient zero-padding
path), preprocessing helper functions, and the loss function.

## Notes on behavior preservation

- `LEARNING_RATE` in `configs/config.py` is `2e-4`, matching the value
  actually used to build the optimizer in the original notebook (an
  earlier, unused `1e-4` assignment was dropped since it never took
  effect).
- Training uses `WINDOW_SIZE=256`, `STRIDE=128` for sliding-window
  sampling; test-time reconstruction uses the same `WINDOW_SIZE=256`
  but a finer `STRIDE=32` for smoother averaged reconstructions — this
  distinction is preserved exactly as in the notebook via
  `RECON_WINDOW_SIZE`/`RECON_STRIDE`.

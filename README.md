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
  audit_dataset.py     Dataset-folder discovery/audit (Complete/Incomplete).
  build_dataset.py     Raw CSVs -> processed_data.pkl (the validated
                       preprocessing algorithm -- flip/shift/interp/norm).
  pipeline.py          Auto-discovery front-end for build_dataset's
                       unmodified functions (used by run_pipeline.py).
  validate_dataset.py  Per-sample pkl validation (shape/NaN/Inf/stats).
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
run_pipeline.py         Local entry point: discover -> preprocess -> pkl
                       -> validate -> Dataset/DataLoader smoke test.
train.py                 Kaggle-ready entry point: python train.py
predict.py               Kaggle-ready entry point: python predict.py
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
`/kaggle/input/...`). Rather than editing those defaults, every
entry-point script below (`run_pipeline.py`, `train.py`, `predict.py`)
takes the relevant path as a `--data-path`/`--dataset-root` CLI flag, so
switching between local and Kaggle is a command-line change, not a code
change -- see [Local baseline pipeline](#local-baseline-pipeline) and
[Kaggle training](#kaggle-training) below.

## Local baseline pipeline

```bash
python run_pipeline.py
```

Runs, end to end, on your machine: dataset discovery -> preprocessing ->
`processed_data.pkl` -> a validation pass over every sample -> a
`SlidingWindowDataset`/`DataLoader` smoke test. It does **not** train --
training is meant to run on Kaggle against the PKL this produces (see
below). Pass `--dataset-root "/path/to/DA exit data_new"` if your raw
data isn't in the default location, and `--output` to change where the
PKL is written.

Discovery is automatic: it lists every subfolder of the dataset root
(and of `Complete/`/`Incomplete/`, wherever the earlier dataset audit
left things) and keeps whichever ones have a `planned.csv`/`PLANNED.csv`
and a `D1.csv` -- no patient name, number, or count is hardcoded, and a
patient filed under `Incomplete/` only for a missing RTPLAN (this
pipeline never reads RTPLAN) is still picked up.

## 1. Preprocessing

Turns the raw per-patient `planned.csv` / `D1.csv` files into
`processed_data.pkl` (planned + detector sinograms resampled onto a
shared coordinate axis). `python run_pipeline.py` (above) is the
recommended way to run this with automatic patient discovery. The
original single-patient-range call is still there, unchanged, if you
want to invoke preprocessing directly:

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
stopping. **This step is meant to run on Kaggle, not locally** -- see
[Kaggle training](#kaggle-training) below for `train.py`.

The underlying functions are still directly callable:

```bash
python -m training.train
```

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
saves everything to `evaluation_results/`. `predict.py` (below) is the
Kaggle/script-friendly entry point; the underlying function is also
directly callable:

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

These call `plt.show()` and are meant for interactive/notebook use;
`predict.py` saves its own PNGs to disk instead (see below), without
modifying `evaluate.py`.

## Kaggle training

No preprocessing runs on Kaggle. The workflow is:

1. **Locally**: `python run_pipeline.py` to produce `processed_data.pkl`.
2. Upload `processed_data.pkl` as a Kaggle Dataset (Kaggle -> Datasets ->
   New Dataset -> upload the file). Note the dataset slug Kaggle gives
   it -- your data will be mounted read-only at
   `/kaggle/input/<dataset-name>/processed_data.pkl`.
3. Upload or `git clone` this repository into a Kaggle Notebook, and
   attach the dataset from step 2 to that notebook.
4. Install dependencies (Kaggle already has nearly everything in
   `requirements.txt` preinstalled; `pip install -r requirements.txt`
   covers the rest) and run training with **one command**:

   ```bash
   python train.py --data-path /kaggle/input/<dataset-name>/processed_data.pkl
   ```

   `train.py` auto-detects CUDA via `configs.config.DEVICE`
   (`torch.cuda.is_available()`), reuses the existing patient-wise
   split/dataloaders/generator/loss/optimizer/training loop unchanged,
   and saves the best checkpoint, a `training_history.json`
   (per-epoch train/val loss), and a `loss_curve.png` to the current
   directory (`/kaggle/working/` by default on Kaggle -- already
   writable, no path setup needed).

5. Generate predictions/metrics with the checkpoint from step 4:

   ```bash
   python predict.py --data-path /kaggle/input/<dataset-name>/processed_data.pkl \
                      --checkpoint best_generator_residual.pth
   ```

   Saves per-patient prediction/ground-truth/planned `.npy` arrays,
   `patient_metrics.csv`/`overall_metrics.csv` (MAE/RMSE/PSNR/SSIM), and
   comparison + absolute-error-map PNGs under
   `evaluation_results/figures/`.

Changing the dataset path (`--data-path`) is the only thing that differs
between a local smoke-run and a Kaggle GPU run -- everything else
(architecture, loss, optimizer, split logic, CUDA selection) is the same
code path either way.

## Tests

```bash
pytest
```

Covers: generator/block output shapes, dataset sample shapes and
residual-target correctness (including the short-patient zero-padding
path), preprocessing helper functions, the loss function, dataset
discovery/audit (`Complete`/`Incomplete`, RTPLAN presence), the
auto-discovery preprocessing pipeline, and pkl validation -- all against
small synthetic data, never real patient files.

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

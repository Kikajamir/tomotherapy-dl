# CLAUDE.md — TomoQA

Guidance for Claude Code sessions in this repo. This is an M.Tech research
codebase for TomoTherapy detector sinogram prediction via deep learning.
Correctness and reproducibility of a scientific result matter more here
than in a typical app repo — treat changes accordingly.

## 1. Project overview

Attention Residual U-Net that predicts TomoTherapy delivered-detector
sinograms from planned fluence, via **residual learning**:

- Input: planned detector sinogram, shape `1x256x542`.
- Target: residual = delivered − planned.
- Model output: predicted residual (not the final detector signal).
- Final detector prediction = `planned + predicted_residual` (this
  reconstruction step lives in `evaluation/evaluate.py`, not in the model).

The repo was modularized out of two Kaggle notebooks:
- `pix2pix.ipynb` — original preprocessing (raw CSVs -> `processed_data.pkl`).
- `unetablation.ipynb` — original training/evaluation.

Both notebooks are now **thin wrappers** that just call into the package
below (see the markdown cells in each — they say this explicitly). The
original monolithic notebook code no longer exists anywhere in this repo
(no checkpoints, no git history to recover it from), so you cannot diff
against "the original cells" directly — treat `configs/config.py`,
docstrings, and the test suite as the record of original behavior instead.

## 2. Repository structure

```
configs/config.py        All shared constants (paths, geometry, model,
                          training, eval). Comments state which notebook
                          each block came from.
preprocessing/build_dataset.py   Raw CSVs -> processed_data.pkl.
data/dataset.py          SlidingWindowDataset (residual-learning samples).
model/blocks.py          ConvBlock, ResidualBlock, EncoderBlock,
                          DownsampleBlock, AttentionGate, DecoderBlock,
                          initialize_weights.
model/generator.py       Generator (Attention Residual U-Net).
model/losses.py          WeightedHuberLoss.
training/train.py        Patient-wise split, dataloaders, training loop.
evaluation/evaluate.py   Checkpoint loading, sliding-window reconstruction,
                          metrics (MAE/RMSE/PSNR/SSIM), plots, saving.
tests/                   Shape sanity checks, residual-math checks,
                          preprocessing helper checks, loss checks.
pix2pix.ipynb            Thin wrapper -> preprocessing.build_dataset
unetablation.ipynb       Thin wrapper -> training.train + evaluation.evaluate
```

`README.md` has full run instructions (CLI + Python usage per stage) —
read it for how to actually invoke preprocessing/training/evaluation.

## 3. Development workflow

1. Before touching any production file, read `configs/config.py` fully —
   it is the single source of truth for constants and documents which
   notebook each value came from.
2. Understand *why* a value/shape is what it is before changing it (see
   §5 Scientific constraints) — many numbers here look arbitrary but are
   load-bearing (e.g. `PAD_WIDTH=1` only works because 542+2=544=16×34).
3. Make the smallest change that satisfies the request.
4. Run `pytest` (see §6) after every change, not just at the end.
5. Report what changed, per file, and its scientific effect (§9).

## 4. Coding standards

- Match existing style: minimal comments, docstrings only on
  classes/modules describing structure (see `model/blocks.py`,
  `model/generator.py` for the tone to match).
- No comments explaining *what* code does; only non-obvious *why*
  (existing examples: the `LEARNING_RATE` note in `configs/config.py`,
  the `PAD_WIDTH`/544 comment in `model/generator.py`).
- All tunable constants belong in `configs/config.py`, not hardcoded
  inline — follow the existing pattern of importing named constants.
- No new dependencies unless necessary; `requirements.txt` is minimal by
  design (numpy/pandas/scipy/sklearn/skimage/matplotlib/torch/tqdm/pytest).
- Don't add config flags, backwards-compat shims, or generalize an API
  beyond what's asked (e.g. do not make the model width-agnostic
  speculatively — see §5).

## 5. Scientific constraints

- **Width 542 is the only supported production width.** Preprocessing
  always projects planned/delivered detector data onto a common
  coordinate system whose output width is fixed at 542 — this is not a
  configurable/arbitrary value.
- The generator's `nn.ReflectionPad2d((1, 1, 0, 0))` (`PAD_WIDTH=1` in
  config) adds a **fixed** 2 px, not adaptive "pad to a multiple of 16."
  It only works because `542 + 2 = 544 = 16 × 34`, which divides evenly
  through the 4 downsample/upsample stages. Feeding any other width will
  legitimately break shape alignment in `AttentionGate` — that is
  expected, not a bug. Do not generalize this to support arbitrary
  widths unless explicitly requested.
- The model predicts a **residual**, not the final signal. Don't collapse
  `predicted_residual` and `planned + predicted_residual` in code without
  being explicit about which one you're producing — evaluation code and
  any future code both depend on this distinction being preserved.
- `WINDOW_SIZE=256`/`STRIDE=128` (training) vs `RECON_WINDOW_SIZE=256`/
  `RECON_STRIDE=32` (test-time reconstruction) are intentionally
  different — do not unify them.
- `LEARNING_RATE=2e-4` is deliberately the *second* of two assignments
  that existed in the original notebook (top-to-bottom execution means
  the first `1e-4` never took effect) — don't "restore" the 1e-4 value.
- Patient-wise split excludes `P6`/`P25` (TomoDirect, out of scope for
  this Helical-only study) — don't include them without being asked.

## 6. Testing requirements

- Run `pytest` after every modification, no exceptions.
- All tests must pass, or you must explicitly explain why a specific
  test is intentionally failing/skipped (with scientific justification,
  not just "it's inconvenient").
- **If a test contradicts a documented production assumption (e.g. width
  542, a fixed constant, a documented notebook quirk), prefer fixing or
  removing the test over changing production code.** Only change
  production code if you find an actual regression vs. the documented/
  original behavior.
- New tests should follow the existing pattern: plain shape/value
  assertions against small synthetic tensors, no mocking frameworks, no
  reliance on real patient data or GPU.

## 7. Safe editing rules

- Treat `configs/config.py` values as frozen unless the user explicitly
  asks to change one — they encode both notebook-original values and
  documented reconciliations of notebook quirks (see the file's own
  header comment and the `LEARNING_RATE` note).
- Never refactor `model/blocks.py` / `model/generator.py` architecture
  "for cleanliness" — shape-sensitive code where subtle reordering
  changes numerical behavior even if shapes still match.
- Preprocessing geometry (`LEAF_PITCH`, `DETECTOR_PITCH`, `SAD`, `SDD`,
  `PITCH_ISO`, `DETECTOR_SHIFT`, interpolation modes) is physics, not
  style — do not touch without explicit approval.
- Prefer additive, isolated changes over broad edits; this is a small
  modular package specifically so changes stay localized to one file.

## 8. Things that must never change without explicit approval

- Output width of preprocessing (542) and the padding math built around it.
- Model architecture (block types, channel counts, attention placement,
  number of residual blocks) in `model/blocks.py` / `model/generator.py`.
- Loss function definition/hyperparameters in `model/losses.py`
  (`LOSS_DELTA`, `LOSS_ALPHA`, `LOSS_GAMMA`) — if you must change one,
  document exactly why in the commit/response, per the user's explicit rule.
- Preprocessing geometry/coordinate-system constants in `configs/config.py`.
- The residual-learning formulation (predict `delivered − planned`, not
  `delivered` directly).
- Train/val/test split logic, ratios, and excluded-patient set.
- Any values documented as "kept as-is despite looking odd" (e.g.
  `LEARNING_RATE`, `RECON_STRIDE` vs `STRIDE`).

## 9. Preferred Claude workflow

1. Read `configs/config.py` and any directly relevant module before
   editing — don't guess at constants or shapes.
2. Cross-check intended behavior against what's documented (docstrings,
   `README.md`, config comments, tests) — the notebooks themselves are
   now thin wrappers and won't contain the original logic to diff against.
3. Decide: is this a genuine regression, or does a test/expectation
   conflict with a documented production constraint? Default to trusting
   documented production constraints (§5, §8) over a test's assumptions.
4. Make the minimal change; do not bundle unrelated cleanup.
5. Run `pytest`; fix or justify any failure.
6. Summarize per file: what changed and its scientific impact (does it
   affect predicted residual values, reconstruction, training dynamics,
   or is it purely structural/test-only).
7. If a change would touch anything in §8, or you're unsure whether a
   test or production code is at fault, stop and ask before editing —
   do not guess on scientific/architectural ambiguity.

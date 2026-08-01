"""
Central configuration for the TomoQA project.

Every constant here was taken verbatim from the two source notebooks
(`pix2pix.ipynb` for preprocessing, `unetablation.ipynb` for modeling /
training / evaluation). Nothing was tuned or changed -- this file exists
so the same values are shared by every module instead of being redefined
(and potentially drifting) in multiple places.

NOTE on LEARNING_RATE: the original `unetablation.ipynb` notebook assigns
`LEARNING_RATE = 1e-4` in the "Configuration" cell and then re-assigns
`LEARNING_RATE = 2e-4` a few cells later, right before building the
optimizer. Because notebook cells execute top-to-bottom, the value that
was actually used to train the model is 2e-4. That is the only value
kept here to avoid silently reintroducing a dead/unused constant.
"""

import os

import torch

# =====================================================================
# Preprocessing (from pix2pix.ipynb)
# =====================================================================

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_PATH = "/kaggle/input/datasets/kikajamir/newdata/DA exit data_new"
SAVE_FILE = "processed_data.pkl"

# Patients are named P1 .. P40
NUM_PATIENTS = 40

# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
LEAF_PITCH = 6.25
DETECTOR_PITCH = 1.24

SAD = 850.0
SDD = 1450.0

PITCH_ISO = DETECTOR_PITCH * SAD / SDD

# ---------------------------------------------------------------------
# Detector alignment
# ---------------------------------------------------------------------
DETECTOR_SHIFT = 33

# ---------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------
INTERP_PLAN = "nearest"
INTERP_DETECTOR = "linear"

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
NORMALIZE_PLANNED = False
NORMALIZE_DETECTOR = True


# =====================================================================
# Modeling / Training / Evaluation (from unetablation.ipynb)
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
RANDOM_SEED = 42  # used again for the patient-wise train/val/test split

# ---------------------------------------------------------------------
# Sliding window (dataset construction / training)
# ---------------------------------------------------------------------
WINDOW_SIZE = 256
STRIDE = 128

# ---------------------------------------------------------------------
# Sliding window (test-time full-sinogram reconstruction)
# ---------------------------------------------------------------------
# The notebook re-declares WINDOW_SIZE/STRIDE locally in the
# reconstruction cell with a finer stride (more overlap -> smoother
# averaged reconstruction). WINDOW_SIZE is unchanged (256); only the
# stride differs from training.
RECON_WINDOW_SIZE = 256
RECON_STRIDE = 32

# ---------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

# ---------------------------------------------------------------------
# Model configuration (Version 2.0)
# ---------------------------------------------------------------------
BASE_CHANNELS = 32          # Reduced from 64
PAD_WIDTH = 1               # Reflection padding on each side
NUM_RES_BLOCKS = 2          # Reduced from 4
GROUPS = 8                  # GroupNorm groups

# ---------------------------------------------------------------------
# Output activation ablation (Experiment A: Sigmoid Output)
# ---------------------------------------------------------------------
# "linear" leaves the final output Conv2d unactivated (existing/default
# behavior -- the predicted residual is unbounded); "sigmoid" adds an
# optional final nn.Sigmoid to constrain the output to the physical
# detector range [0,1]. Purely additive -- see model/generator.py.
OUTPUT_ACTIVATION = "linear"  # "linear" or "sigmoid"

# ---------------------------------------------------------------------
# Deep supervision ablation (Experiment B)
# ---------------------------------------------------------------------
# When True, the generator exposes auxiliary 1x1-conv outputs at the
# dec2/dec3 decoder resolutions (only while `generator.training` is
# True, i.e. never at inference/eval), and the training loop adds
# 0.5 * L_decoder2 + 0.25 * L_decoder3 on top of the main loss -- see
# model/generator.py (`aux_outputs`) and training/train.py.
# Evaluated and found not to improve performance -- back to False (no
# aux heads, stock Attention U-Net) to match the current baseline.
USE_DEEP_SUPERVISION = False

# ---------------------------------------------------------------------
# Multi-scale loss ablation (Experiment C)
# ---------------------------------------------------------------------
# When True, build_criterion() returns MultiScaleLoss (plain L1 at full
# resolution + 0.5 * L1_half + 0.25 * L1_quarter, via average pooling)
# instead of whatever LOSS_TYPE selects -- see model/losses.py.
USE_MULTI_SCALE_LOSS = False

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)
EARLY_STOPPING_PATIENCE = 10
SAVE_PATH = "best_generator_residual.pth"

# ---------------------------------------------------------------------
# Loss (Continuous Intensity Weighted Huber Loss)
# ---------------------------------------------------------------------
LOSS_DELTA = 0.02
LOSS_ALPHA = 11.0
LOSS_GAMMA = 2.5

# ---------------------------------------------------------------------
# Loss selection (ablation study: baseline vs. Weighted L1 vs.
# Huber + Gradient vs. Beam-Aware Weighted Huber)
# ---------------------------------------------------------------------
# "l1" keeps the existing baseline loss (WeightedHuberLoss, untouched,
# see LOSS_DELTA/ALPHA/GAMMA above) so the default training behavior is
# unchanged; "weighted_l1" switches to WeightedL1Loss (previous
# experiment); "huber_gradient" switches to WeightedHuberGradientLoss
# (WeightedHuberLoss + lambda_gradient * GradientLoss, current
# baseline); "beam_weighted" switches to BeamAwareWeightedHuberLoss (see
# BEAM_THRESHOLD/BEAM_WEIGHT/BACKGROUND_WEIGHT below).
LOSS_TYPE = "beam_weighted"  # "l1", "weighted_l1", "huber_gradient", or "beam_weighted"
WEIGHTED_L1_ALPHA = 2.0

# ---------------------------------------------------------------------
# Beam-aware weighted Huber loss ablation ("beam_weighted")
# ---------------------------------------------------------------------
# EDA showed background pixels (planned == 0) make up the large majority
# of the sinogram, while the planned sinogram already tells us exactly
# where beam exists -- so the beam mask is derived from `planned` only
# (never from the detector/target, which wouldn't be available at
# inference time). BEAM_THRESHOLD=0.0 is the physically exact choice:
# `planned` is a treatment-plan fluence value with no measurement noise,
# so it is exactly zero wherever no beam was ever planned and non-zero
# everywhere beam exists -- no epsilon guessing needed.
#
# Loss = BEAM_WEIGHT * WeightedHuber(beam pixels)
#      + BACKGROUND_WEIGHT * WeightedHuber(background pixels)
#      + LOSS_GRADIENT_LAMBDA * GradientLoss(full image)
#
# Each region's WeightedHuber term is normalized by its OWN pixel count
# (unlike a single pixel-wise .mean()), so the ~5x larger background
# population can no longer dilute the beam region's gradient signal by
# sheer pixel count the way it does under the existing losses above.
# BACKGROUND_WEIGHT < 1 de-emphasizes (but does not zero out) the
# background term, since the background target isn't exactly zero
# either (small but consistent positive pedestal in the real data).
BEAM_THRESHOLD = 0.0
BEAM_WEIGHT = 1.0
BACKGROUND_WEIGHT = 0.3

# Beam intensity weighting inside the beam region: "binary" reproduces
# the original beam_weighted behavior exactly (every beam pixel counted
# equally in the beam-region mean); "continuous" additionally multiplies
# each beam pixel's Huber loss by `1 + BEAM_ALPHA * planned**BEAM_GAMMA`
# before averaging, so higher-fluence (near-peak) pixels get more say in
# the beam-region mean -- real planned leaf-open-time is a continuum
# (empirically spread across the full [0,1] range, not bimodal), so this
# is the more physically faithful option, but "binary" stays the default
# to keep the simplest ablation (region-normalization alone, no extra
# hyperparameters) as the zero-config baseline.
BEAM_WEIGHT_MODE = "continuous"  # "binary" or "continuous"

# Defaults match LOSS_ALPHA/LOSS_GAMMA above so this continuous
# planned-intensity weighting operates on a comparable scale to the
# existing target-magnitude weighting in WeightedHuberLoss; tune
# independently as needed.
BEAM_ALPHA = 11.0
BEAM_GAMMA = 2.5

# Weight of the image-gradient term added on top of the unchanged
# WeightedHuberLoss: TotalLoss = WeightedHuberLoss + LOSS_GRADIENT_LAMBDA
# * GradientLoss.
# (Experiment D: this is already a plain config parameter consumed as
# WeightedHuberGradientLoss's `lambda_gradient` default -- trying 0.1 /
# 0.25 / 0.5 / 1.0 only requires editing this value and rerunning
# training, no other code changes.)
LOSS_GRADIENT_LAMBDA = 0.1

# Ablation checkpoints are written outside SAVE_PATH so the baseline
# checkpoint is never overwritten by an ablation run.
WEIGHTED_L1_SAVE_PATH = "experiments/weighted_l1/best_generator_residual.pth"
HUBER_GRADIENT_SAVE_PATH = "experiments/huber_gradient/best_generator_residual.pth"
SIGMOID_OUTPUT_SAVE_PATH = "experiments/sigmoid_output/best_generator_residual.pth"
DEEP_SUPERVISION_SAVE_PATH = "experiments/deep_supervision/best_generator_residual.pth"
MULTI_SCALE_LOSS_SAVE_PATH = "experiments/multi_scale_loss/best_generator_residual.pth"
BEAM_WEIGHTED_SAVE_PATH = "experiments/beam_weighted/best_generator_residual.pth"

# ---------------------------------------------------------------------
# Patient-wise train / val / test split
# ---------------------------------------------------------------------
# TOMOQA_DATA_PATH overrides this for local development so the Kaggle
# input path below (the default used on Kaggle) never has to be edited
# to run locally -- see also train.py's --data-path CLI flag, which
# takes precedence over both.
DATA_PATH = os.environ.get(
    "TOMOQA_DATA_PATH",
    "/kaggle/input/datasets/kikajamir/tomotherapy-processed-v1/processed_data.pkl",
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# TomoDirect patients excluded from the (Helical-only) study
EXCLUDED_PATIENTS = {"P6", "P25"}

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
EVAL_SAVE_DIR = "evaluation_results"

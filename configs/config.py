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
# Loss selection (ablation study: baseline vs. Weighted L1)
# ---------------------------------------------------------------------
# "l1" keeps the existing baseline loss (WeightedHuberLoss, untouched,
# see LOSS_DELTA/ALPHA/GAMMA above) so the default training behavior is
# unchanged; "weighted_l1" switches to WeightedL1Loss.
LOSS_TYPE = "weighted_l1"  # "l1" or "weighted_l1"
WEIGHTED_L1_ALPHA = 2.0

# Ablation checkpoints are written outside SAVE_PATH so the baseline
# checkpoint is never overwritten by an ablation run.
WEIGHTED_L1_SAVE_PATH = "experiments/weighted_l1/best_generator_residual.pth"

# ---------------------------------------------------------------------
# Patient-wise train / val / test split
# ---------------------------------------------------------------------
DATA_PATH = "/kaggle/input/notebooks/kikajamir/pix2pix/processed_data.pkl"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# TomoDirect patients excluded from the (Helical-only) study
EXCLUDED_PATIENTS = {"P6", "P25"}

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
EVAL_SAVE_DIR = "evaluation_results"

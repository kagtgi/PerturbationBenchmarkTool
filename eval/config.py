"""
Centralized configuration for the perturbation benchmarking toolkit.

All configurable parameters live here. Users should modify this file
(or override at runtime) to adapt the evaluation to their dataset.
"""

import torch

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_PATH: str = "K562.h5ad"  # Path to the .h5ad Perturb-seq file

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SUBSAMPLE_FRAC: float = 0.20       # Fraction of cells kept per perturbation
MIN_CELLS_PER_PERT: int = 5        # Minimum cells after subsampling to evaluate
MAX_T3_CELLS: int = 512            # Max cells per pert for Tier-3 distance matrices

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CTRL_LABEL: str = "non-targeting"   # obs value identifying control cells
PERT_COL: str = "gene"              # obs column holding perturbation labels
TOP_K_DE: int = 50                  # Number of top DE genes for Tier-2 metrics
DIR_ACC_THRESHOLD: float = 0.1      # Minimum |delta| to count for directional accuracy

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "results"         # Directory for saving evaluation results

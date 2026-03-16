"""
Dataset loading and preprocessing for perturbation benchmarking.

Handles reading .h5ad files, ensuring consistent normalization,
and preparing data for downstream evaluation.
"""

import logging

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from . import config

logger = logging.getLogger(__name__)


def load_h5ad(path: str | None = None) -> sc.AnnData:
    """Load a Perturb-seq .h5ad file and return the AnnData object.

    Parameters
    ----------
    path : str, optional
        Path to the .h5ad file.  Falls back to ``config.DATA_PATH``.

    Returns
    -------
    sc.AnnData
    """
    path = path or config.DATA_PATH
    logger.info("Loading %s ...", path)
    adata = sc.read_h5ad(path)
    logger.info("Loaded: %s", adata.shape)
    return adata


def ensure_raw_counts(adata: sc.AnnData) -> sc.AnnData:
    """Store raw counts in ``adata.layers['counts']`` and reset ``adata.X``.

    If ``layers['counts']`` is absent, the current ``adata.X`` is assumed to
    hold raw integer counts and is copied there.  In all cases ``adata.X`` is
    reset to equal ``layers['counts']`` so that every model which reads
    ``adata.X`` directly (GEARS, scGPT auto-detect, C2S X_true capture) always
    receives raw counts and can safely apply its own normalization.
    """
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    # Always expose raw counts through adata.X for consistent downstream access.
    adata.X = adata.layers["counts"].copy()
    return adata


def log_normalize(adata: sc.AnnData, target_sum: float = 1e4) -> sc.AnnData:
    """Normalize total counts and apply log1p in-place.

    Operates on ``adata.X``.  If data already looks log-normalized
    (max value ≤ 50 and non-integer), this is a no-op.
    """
    X_check = adata.X[:1000]
    if sp.issparse(X_check):
        X_check = X_check.toarray()
    looks_raw = bool(
        np.nanmax(X_check) > 50 or np.array_equal(X_check, X_check.astype(int))
    )
    if looks_raw:
        logger.info("Applying normalize_total(%.0f) + log1p", target_sum)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    else:
        logger.info("Data looks log-normalized — skipping normalization.")
    return adata


def dense_X(adata: sc.AnnData) -> np.ndarray:
    """Return ``adata.X`` as a dense float32 numpy array."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def get_ctrl_mask(adata: sc.AnnData, ctrl_label: str | None = None,
                  pert_col: str | None = None) -> np.ndarray:
    """Boolean mask for control cells."""
    ctrl_label = ctrl_label or config.CTRL_LABEL
    pert_col = pert_col or config.PERT_COL
    return (adata.obs[pert_col] == ctrl_label).values


def get_perturbation_list(adata: sc.AnnData, ctrl_label: str | None = None,
                          pert_col: str | None = None) -> list[str]:
    """Return sorted list of unique non-control perturbation labels."""
    ctrl_label = ctrl_label or config.CTRL_LABEL
    pert_col = pert_col or config.PERT_COL
    labels = adata.obs[pert_col].astype(str).unique().tolist()
    return sorted([l for l in labels if l != ctrl_label])

"""
Centralized sampling logic for fair model comparison.

All models must use identical subsampled cells so that evaluation is
comparable.  This module provides a single entry point for stratified
subsampling keyed by a fixed random seed.
"""

import logging

import numpy as np
import scanpy as sc

from . import config

logger = logging.getLogger(__name__)


def stratified_subsample(
    adata: sc.AnnData,
    frac: float | None = None,
    min_cells: int | None = None,
    seed: int | None = None,
    pert_col: str | None = None,
    ctrl_label: str | None = None,
) -> sc.AnnData:
    """Stratified subsample: keep *frac* of cells per perturbation group.

    After subsampling, perturbations with fewer than *min_cells* cells
    are dropped entirely.

    Parameters
    ----------
    adata : sc.AnnData
        Full dataset.
    frac : float
        Fraction to keep (default ``config.SUBSAMPLE_FRAC``).
    min_cells : int
        Minimum cells per perturbation after subsampling
        (default ``config.MIN_CELLS_PER_PERT``).
    seed : int
        Random seed (default ``config.RANDOM_SEED``).
    pert_col : str
        Column in ``adata.obs`` with perturbation labels.
    ctrl_label : str
        Label for control cells.

    Returns
    -------
    sc.AnnData
        Subsampled copy.
    """
    frac = frac if frac is not None else config.SUBSAMPLE_FRAC
    min_cells = min_cells if min_cells is not None else config.MIN_CELLS_PER_PERT
    seed = seed if seed is not None else config.RANDOM_SEED
    pert_col = pert_col or config.PERT_COL
    ctrl_label = ctrl_label or config.CTRL_LABEL

    rng = np.random.default_rng(seed)
    keep_idx: list[str] = []

    obs_col = adata.obs[pert_col]
    for label in obs_col.unique():
        grp_idx = adata.obs.index[obs_col == label]
        n = max(2, int(len(grp_idx) * frac))
        n = min(n, len(grp_idx))
        keep_idx.extend(
            rng.choice(grp_idx, size=n, replace=False).tolist()
        )

    adata = adata[keep_idx].copy()

    # Drop perturbation groups that are too small
    counts = adata.obs[pert_col].value_counts()
    keep_labels = counts[counts >= min_cells].index
    adata = adata[adata.obs[pert_col].isin(keep_labels)].copy()

    n_perts = adata.obs[pert_col].nunique()
    has_ctrl = ctrl_label in adata.obs[pert_col].values
    logger.info(
        "Subsampled: %s  (%d%% stratified, min %d cells) | %d groups%s",
        adata.shape, int(frac * 100), min_cells, n_perts,
        " (incl. ctrl)" if has_ctrl else "",
    )
    return adata

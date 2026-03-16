"""
GEARS evaluation — zero-shot transfer (Norman pretrained → target dataset).

Uses the pretrained GEARS model trained on the Norman dataset and evaluates
on the target .h5ad data using overlapping gene vocabulary.

KEY DESIGN:
- Zero-shot transfer: Norman pretrained → K562 (or other target)
- Log1p-CPM normalization in BOTH prediction and ground-truth spaces
- T1/T2 metrics via shared metrics.py
- T3: point-vs-distribution (GEARS outputs a deterministic centroid,
  not a cell distribution — Energy Distance / MMD compare the single
  predicted mean against the true single-cell distribution)
- Progress tracking every 50 perturbations

Notebook reference: Eval_.ipynb cell 3 (GEARS EVALUATION).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tarfile
import time
import warnings
import zipfile

import numpy as np
import scipy.sparse as sp
import torch

from .. import config
from ..metrics import (
    centroid_accuracy_and_pds,
    directional_accuracy,
    jaccard_topk,
    pearson_delta_topk,
    systema_pearson_delta,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS  (match the pretrained Norman model's training setup)
# =============================================================================

NORMAN_DATA_URL   = "https://dataverse.harvard.edu/api/access/datafile/6979957"
NORMAN_MODEL_URL  = "https://dataverse.harvard.edu/api/access/datafile/10457098"
NORMAN_DATA_PATH  = "./norman_umi_go"
NORMAN_MODEL_CKPT = "./model_ckpt"
# The Norman pretrained model was built with seed=1 — do NOT change.
NORMAN_SPLIT_SEED = 1
MAX_T3_CELLS = 512


# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

def _pip(*pkgs: str) -> None:
    """pip install with --break-system-packages for Colab compatibility."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "--break-system-packages", *pkgs],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Install GEARS dependencies if not already present.

    Install order matters: torch_geometric must precede cell-gears.
    """
    try:
        import torch_geometric  # noqa: F401
        logger.info("torch_geometric: already installed")
    except ImportError:
        logger.info("Installing torch_geometric ...")
        _pip("torch_geometric")

    try:
        import torch_scatter  # noqa: F401
        logger.info("torch_scatter: already installed")
    except ImportError:
        tv  = torch.__version__.split("+")[0]
        cud = torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
        try:
            _pip("torch_scatter", "torch_sparse",
                 "-f", f"https://data.pyg.org/whl/torch-{tv}+cu{cud}.html")
        except Exception as e:
            logger.warning("torch_scatter install skipped: %s", e)

    try:
        import gears  # noqa: F401
        logger.info("cell-gears: already installed")
    except ImportError:
        logger.info("Installing cell-gears ...")
        _pip("cell-gears", "scanpy")

    # Monkey-patch pandas 2.x — Series.nonzero was removed
    import pandas as pd
    if not hasattr(pd.Series, "nonzero"):
        pd.Series.nonzero = lambda self: self.to_numpy().nonzero()


# =============================================================================
# DATA / MODEL DOWNLOAD
# =============================================================================

def _download_norman_assets() -> None:
    """Download Norman dataset and pretrained GEARS weights if not cached."""
    from gears.utils import dataverse_download

    if not os.path.exists(NORMAN_DATA_PATH):
        logger.info("Downloading Norman data ...")
        dataverse_download(NORMAN_DATA_URL, "norman_umi_go.tar.gz")
        with tarfile.open("norman_umi_go.tar.gz", "r:gz") as tar:
            tar.extractall()
        logger.info("Norman data extracted.")
    else:
        logger.info("Norman data: found locally (%s)", NORMAN_DATA_PATH)

    if not os.path.exists(NORMAN_MODEL_CKPT):
        logger.info("Downloading pretrained GEARS model ...")
        dataverse_download(NORMAN_MODEL_URL, "model.zip")
        with zipfile.ZipFile("model.zip", "r") as z:
            z.extractall(path="./")
        logger.info("GEARS model extracted.")
    else:
        logger.info("GEARS model: found locally (%s)", NORMAN_MODEL_CKPT)


# =============================================================================
# T3 HELPERS  (point-mass variants — GEARS gives a single centroid, not cells)
# =============================================================================

def _energy_distance_point(pred_pt: torch.Tensor,
                            x_true: torch.Tensor,
                            max_cells: int = MAX_T3_CELLS) -> float:
    """Energy distance: single predicted point vs true cell distribution.

    Formula: 2·E[d(pred, true_i)] − E[d(true_i, true_j)]

    pred_pt : (1, G) or (G,) tensor — the deterministic GEARS prediction
    x_true  : (N, G) tensor — observed perturbed single cells
    """
    pred_pt = pred_pt.view(1, -1)
    if x_true.size(0) > max_cells:
        idx    = torch.randperm(x_true.size(0), device=x_true.device)[:max_cells]
        x_true = x_true[idx]

    d_pt = torch.cdist(pred_pt, x_true, p=2.0).mean()
    n_t  = x_true.size(0)
    if n_t > 1:
        dtt  = torch.cdist(x_true, x_true, p=2.0)
        off  = ~torch.eye(n_t, dtype=torch.bool, device=x_true.device)
        d_tt = dtt[off].mean()
    else:
        d_tt = torch.tensor(0.0, device=x_true.device)
    return max((2 * d_pt - d_tt).item(), 0.0)


def _mmd_rbf_point(pred_pt: torch.Tensor,
                   x_true: torch.Tensor,
                   max_cells: int = MAX_T3_CELLS) -> float:
    """MMD (RBF kernel, median bandwidth) — single point vs cell distribution.

    pred_pt : (1, G) or (G,) tensor
    x_true  : (N, G) tensor
    """
    pred_pt = pred_pt.view(1, -1)
    if x_true.size(0) > max_cells:
        idx    = torch.randperm(x_true.size(0), device=x_true.device)[:max_cells]
        x_true = x_true[idx]

    dtt = torch.cdist(x_true, x_true, p=2.0)
    n_t = x_true.size(0)
    if n_t > 1:
        off    = ~torch.eye(n_t, dtype=torch.bool, device=x_true.device)
        sigma2 = dtt.pow(2)[off].median().clamp(1e-6).item() / 2.0
    else:
        sigma2 = 1.0

    def rbf(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma2))

    kxx = rbf(pred_pt, pred_pt).mean()
    kyy = rbf(x_true,  x_true ).mean()
    kxy = rbf(pred_pt, x_true ).mean()
    return max((kxx - 2 * kxy + kyy).item(), 0.0)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_eval(adata, cfg: dict) -> dict:
    """Run GEARS zero-shot evaluation.

    Parameters
    ----------
    adata : sc.AnnData
        Subsampled dataset with perturbation labels in ``obs[pert_col]``.
        The dataset should already be subsampled (20% stratified).
    cfg : dict
        Runtime configuration dictionary (see ``eval.config``).

    Returns
    -------
    dict
        Keys: ``model``, ``metrics``, ``pert_names``,
        ``n_overlap_genes``, ``runtime_seconds``.
    """
    warnings.filterwarnings("ignore")

    t_start    = time.time()
    device_str = cfg.get("DEVICE",     config.DEVICE)
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col   = cfg.get("PERT_COL",   config.PERT_COL)
    top_k_de   = cfg.get("TOP_K_DE",   config.TOP_K_DE)
    device     = torch.device(device_str)

    logger.info("Device: %s | numpy: %s | torch: %s",
                device, np.__version__, torch.__version__)

    # ── Install & imports ─────────────────────────────────────────────────
    _install_dependencies()
    from gears import PertData, GEARS

    # ── Download Norman assets ────────────────────────────────────────────
    _download_norman_assets()

    # ── Load Norman model (seed=1 required by pretrained weights) ─────────
    logger.info("Loading Norman PertData + model ...")
    pert_data = PertData("./")
    pert_data.load(data_path=NORMAN_DATA_PATH)
    pert_data.prepare_split(split="no_test", seed=NORMAN_SPLIT_SEED)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    gears_model = GEARS(
        pert_data, device=device_str,
        weight_bias_track=False,
        proj_name="gears",
        exp_name="gears_misc_umi_no_test",
    )
    gears_model.model_initialize(hidden_size=64)
    gears_model.load_pretrained(NORMAN_MODEL_CKPT)

    norman_genes     = set(pert_data.gene_names.tolist())
    norman_gene_list = pert_data.gene_names.tolist()
    logger.info("Norman vocab: %d genes", len(norman_genes))

    # ── Prepare target data ───────────────────────────────────────────────
    adata = adata.copy()
    # Map obs[pert_col] → "ctrl" | "<gene>+ctrl"
    adata.obs["condition"] = adata.obs[pert_col].apply(
        lambda g: "ctrl" if g == ctrl_label else f"{g}+ctrl"
    )
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    k562_genes = adata.var["gene_name"].tolist()

    # Log-normalize control cells to match GEARS log1p-CPM prediction space
    X_ctrl_raw     = adata[adata.obs["condition"] == "ctrl"].X.toarray()
    _lib           = X_ctrl_raw.sum(axis=1, keepdims=True)
    X_ctrl_ln      = np.log1p(X_ctrl_raw / (_lib + 1e-8) * 1e4)
    ctrl_mean_k562 = X_ctrl_ln.mean(axis=0)           # (n_k562_genes,)

    # ── Gene & perturbation overlap ───────────────────────────────────────
    overlap_genes  = norman_genes & set(k562_genes)
    overlap_sorted = sorted(overlap_genes)
    norman_idx     = np.array([norman_gene_list.index(g) for g in overlap_sorted])
    k562_idx       = np.array([k562_genes.index(g)       for g in overlap_sorted])

    ctrl_mean_overlap = torch.tensor(
        ctrl_mean_k562[k562_idx], dtype=torch.float32
    ).to(device)

    k562_perts = sorted([
        c for c in adata.obs["condition"].unique()
        if c != "ctrl" and c.replace("+ctrl", "") in norman_genes
    ])

    logger.info(
        "Gene overlap: Norman=%d  Target=%d  Overlap=%d",
        len(norman_genes), len(k562_genes), len(overlap_genes),
    )
    logger.info("Predictable perturbations: %d", len(k562_perts))

    # ── Zero-shot inference ───────────────────────────────────────────────
    logger.info("[5/5] Running zero-shot predictions (%d perts) ...", len(k562_perts))

    pred_list, true_list, pert_names = [], [], []
    # T3: GEARS gives one deterministic centroid per perturbation, so we
    # compare that single point against the true single-cell distribution.
    true_cells_dict: dict[str, torch.Tensor] = {}
    pred_point_dict: dict[str, torch.Tensor] = {}
    n_skip = 0
    t0     = time.time()

    for i, pert in enumerate(k562_perts):
        gene = pert.replace("+ctrl", "")
        mask = adata.obs["condition"] == pert
        if mask.sum() < 2:
            n_skip += 1
            continue

        # Ground truth — log-normalize to match GEARS log1p CPM space
        X_true_raw        = adata[mask].X.toarray()
        _lib              = X_true_raw.sum(axis=1, keepdims=True)
        X_true            = np.log1p(X_true_raw / (_lib + 1e-8) * 1e4)
        true_mean_overlap = X_true.mean(axis=0)[k562_idx]

        # GEARS prediction (single deterministic point)
        try:
            pred_dict    = gears_model.predict([[gene]])
            pred_full    = np.array(list(pred_dict.values())[0])
            pred_overlap = pred_full[norman_idx]
        except Exception:
            n_skip += 1
            continue

        pred_list.append(torch.tensor(pred_overlap,      dtype=torch.float32).to(device))
        true_list.append(torch.tensor(true_mean_overlap, dtype=torch.float32).to(device))
        true_cells_dict[pert] = torch.tensor(
            X_true[:, k562_idx], dtype=torch.float32
        ).to(device)
        pred_point_dict[pert] = torch.tensor(
            pred_overlap, dtype=torch.float32
        ).unsqueeze(0).to(device)                       # (1, n_overlap)
        pert_names.append(pert)

        if (i + 1) % 50 == 0 or (i + 1) == len(k562_perts):
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1e-6)
            eta     = (len(k562_perts) - i - 1) / rate
            logger.info(
                "  %4d/%d  done=%d  skip=%d  %ds elapsed  ETA %ds",
                i + 1, len(k562_perts), len(pert_names), n_skip,
                int(elapsed), int(eta),
            )

    if not pert_names:
        raise RuntimeError("No perturbations could be evaluated for GEARS.")

    pred_centroids = torch.stack(pred_list)    # (P, n_overlap)
    true_centroids = torch.stack(true_list)
    pred_delta     = pred_centroids - ctrl_mean_overlap.unsqueeze(0)
    true_delta     = true_centroids - ctrl_mean_overlap.unsqueeze(0)

    logger.info(
        "Collected: %d perts | shape %s | skipped: %d",
        len(pert_names), tuple(pred_centroids.shape), n_skip,
    )

    # ── Tier 1 & 2 metrics (centroid-level) ──────────────────────────────
    logger.info("Computing T1/T2 metrics ...")
    ca, pds = centroid_accuracy_and_pds(pred_centroids, true_centroids)
    spd     = systema_pearson_delta(pred_delta, true_delta)
    da      = directional_accuracy(pred_delta, true_delta)
    pde     = pearson_delta_topk(pred_delta, true_delta, k=top_k_de)
    jac     = jaccard_topk(pred_delta, true_delta, k=top_k_de)

    # ── Tier 3: point-vs-distribution ─────────────────────────────────────
    logger.info("Computing T3 metrics (point-vs-distribution) ...")
    e_scores, mmd_scores = [], []
    t0_t3 = time.time()

    for i, pert in enumerate(pert_names):
        e_scores.append(
            _energy_distance_point(pred_point_dict[pert], true_cells_dict[pert])
        )
        mmd_scores.append(
            _mmd_rbf_point(pred_point_dict[pert], true_cells_dict[pert])
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(pert_names):
            elapsed = time.time() - t0_t3
            eta     = elapsed / (i + 1) * (len(pert_names) - i - 1)
            logger.info(
                "  T3 %4d/%d  %ds elapsed  ETA %ds",
                i + 1, len(pert_names), int(elapsed), int(eta),
            )

    e_dist  = float(np.mean(e_scores))
    mmd_val = float(np.mean(mmd_scores))

    metrics = {
        "T1_Centroid_Accuracy":      ca,
        "T1_Profile_Distance_Score": pds,
        "T1_Systema_Pearson_Delta":  spd,
        "T2_Directional_Accuracy":   da,
        "T2_Pearson_Delta_TopK":     pde,
        "T2_Jaccard_TopK":           jac,
        "T3_Energy_Distance":        e_dist,
        "T3_MMD_RBF":                mmd_val,
    }

    # ── Cleanup ───────────────────────────────────────────────────────────
    del gears_model, pert_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    runtime = time.time() - t_start
    logger.info(
        "GEARS done: CA=%.4f  PDS=%.4f  DirAcc=%.4f  "
        "PearsonDE=%.4f  Energy=%.4f  MMD=%.4f  (%.2f min)",
        ca, pds, da, pde, e_dist, mmd_val, runtime / 60,
    )

    return {
        "model":           "GEARS",
        "metrics":         metrics,
        "pert_names":      pert_names,
        "n_overlap_genes": len(overlap_sorted),
        "runtime_seconds": runtime,
    }

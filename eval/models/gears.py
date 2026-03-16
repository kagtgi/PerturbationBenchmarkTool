"""
GEARS evaluation — zero-shot transfer (Norman pretrained → target dataset).

Uses the pretrained GEARS model trained on the Norman dataset and evaluates
on the target .h5ad data using overlapping gene vocabulary.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

from .. import config
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def _pip(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "--break-system-packages", *pkgs],
        stdout=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Install GEARS and torch_geometric if needed."""
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        logger.info("Installing torch_geometric ...")
        _pip("torch_geometric")

    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        tv = torch.__version__.split("+")[0]
        cud = torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
        try:
            _pip("torch_scatter", "torch_sparse",
                 "-f", f"https://data.pyg.org/whl/torch-{tv}+cu{cud}.html")
        except Exception as e:
            logger.warning("torch_scatter install skipped: %s", e)

    _pip("cell-gears", "scanpy")

    # Monkey-patch pandas 2.x (Series.nonzero removed)
    import pandas as pd
    if not hasattr(pd.Series, "nonzero"):
        pd.Series.nonzero = lambda self: self.to_numpy().nonzero()


def _download_norman_data(data_path: str, model_ckpt: str) -> None:
    """Download Norman dataset and pretrained GEARS model if not cached."""
    from gears.utils import dataverse_download

    if not os.path.exists(data_path):
        logger.info("Downloading Norman data ...")
        dataverse_download(
            "https://dataverse.harvard.edu/api/access/datafile/6979957",
            "norman_umi_go.tar.gz",
        )
        import tarfile
        with tarfile.open("norman_umi_go.tar.gz", "r:gz") as tar:
            tar.extractall()

    if not os.path.exists(model_ckpt):
        logger.info("Downloading pretrained GEARS model ...")
        dataverse_download(
            "https://dataverse.harvard.edu/api/access/datafile/10457098",
            "model.zip",
        )
        from zipfile import ZipFile
        with ZipFile("model.zip", "r") as z:
            z.extractall(path="./")


def run_eval(adata, cfg: dict) -> dict:
    """Run GEARS zero-shot evaluation.

    Parameters
    ----------
    adata : sc.AnnData
        Subsampled dataset with perturbation labels in ``obs[pert_col]``.
    cfg : dict
        Runtime configuration dictionary.

    Returns
    -------
    dict with ``model``, ``metrics``, ``pert_names``, ``runtime_seconds``.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import scanpy as sc

    t_start = time.time()
    device_str = cfg.get("DEVICE", config.DEVICE)
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    top_k = cfg.get("TOP_K_DE", config.TOP_K_DE)
    device = torch.device(device_str)

    # --- Install & download ------------------------------------------------
    _install_dependencies()
    from gears import PertData, GEARS

    norman_data_path = "./norman_umi_go"
    norman_model_ckpt = "./model_ckpt"
    _download_norman_data(norman_data_path, norman_model_ckpt)

    # --- Load Norman model -------------------------------------------------
    logger.info("Loading Norman PertData + model ...")
    pert_data = PertData("./")
    pert_data.load(data_path=norman_data_path)
    # Use seed=1 to match the pretrained Norman model's expected split
    pert_data.prepare_split(split="no_test", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    gears_model = GEARS(
        pert_data, device=device_str,
        weight_bias_track=False, proj_name="gears", exp_name="gears_misc_umi_no_test",
    )
    gears_model.model_initialize(hidden_size=64)
    gears_model.load_pretrained(norman_model_ckpt)

    norman_genes = set(pert_data.gene_names.tolist())
    norman_gene_list = pert_data.gene_names.tolist()

    # --- Prepare target data -----------------------------------------------
    adata.obs["condition"] = adata.obs[pert_col].apply(
        lambda g: "ctrl" if g == ctrl_label else f"{g}+ctrl"
    )
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    k562_genes = adata.var["gene_name"].tolist()

    # Log-normalize control cells
    X_ctrl_raw = adata[adata.obs["condition"] == "ctrl"].X.toarray()
    _lib = X_ctrl_raw.sum(axis=1, keepdims=True)
    X_ctrl_ln = np.log1p(X_ctrl_raw / (_lib + 1e-8) * 1e4)
    ctrl_mean = X_ctrl_ln.mean(axis=0)

    # --- Gene & perturbation overlap ---------------------------------------
    overlap = norman_genes & set(k562_genes)
    overlap_sorted = sorted(overlap)
    norman_idx = np.array([norman_gene_list.index(g) for g in overlap_sorted])
    k562_idx = np.array([k562_genes.index(g) for g in overlap_sorted])

    ctrl_mean_overlap = torch.tensor(
        ctrl_mean[k562_idx], dtype=torch.float32
    ).to(device)

    k562_perts = sorted([
        c for c in adata.obs["condition"].unique()
        if c != "ctrl" and c.replace("+ctrl", "") in norman_genes
    ])

    logger.info("Gene overlap: %d | Predictable perts: %d",
                len(overlap), len(k562_perts))

    # --- Inference ---------------------------------------------------------
    pred_list, true_list, pert_names = [], [], []
    true_cells_dict, pred_point_dict = {}, {}

    for pert in k562_perts:
        gene = pert.replace("+ctrl", "")
        mask = adata.obs["condition"] == pert
        if mask.sum() < 2:
            continue

        X_true_raw = adata[mask].X.toarray()
        _lib = X_true_raw.sum(axis=1, keepdims=True)
        X_true = np.log1p(X_true_raw / (_lib + 1e-8) * 1e4)
        true_mean_overlap = X_true.mean(axis=0)[k562_idx]

        try:
            pred_dict = gears_model.predict([[gene]])
            pred_full = np.array(list(pred_dict.values())[0])
            pred_overlap = pred_full[norman_idx]
        except Exception:
            continue

        pred_list.append(torch.tensor(pred_overlap, dtype=torch.float32).to(device))
        true_list.append(torch.tensor(true_mean_overlap, dtype=torch.float32).to(device))
        true_cells_dict[pert] = torch.tensor(
            X_true[:, k562_idx], dtype=torch.float32
        ).to(device)
        pred_point_dict[pert] = torch.tensor(
            pred_overlap, dtype=torch.float32
        ).unsqueeze(0).to(device)
        pert_names.append(pert)

    pred_centroids = torch.stack(pred_list)
    true_centroids = torch.stack(true_list)

    logger.info("Evaluated %d perturbations on %d overlap genes",
                len(pert_names), len(overlap_sorted))

    # --- Metrics -----------------------------------------------------------
    # GEARS outputs one deterministic centroid prediction per perturbation
    # (not a cell distribution), so T3 distributional metrics (energy distance,
    # MMD) are skipped — comparing a single point to a cell distribution is
    # not equivalent to how T3 is computed for models that generate per-cell
    # predictions (scGPT, STATE, C2S, CPA).
    metrics = compute_all_metrics(
        pred_centroids, true_centroids, ctrl_mean_overlap,
    )

    return {
        "model": "GEARS",
        "metrics": metrics,
        "pert_names": pert_names,
        "n_overlap_genes": len(overlap_sorted),
        "runtime_seconds": time.time() - t_start,
    }
"""
GEARS evaluation — zero-shot transfer (Norman pretrained → target dataset).

Uses the pretrained GEARS model trained on the Norman dataset and evaluates
on the target .h5ad data using overlapping gene vocabulary.

KEY FEATURES:
- Zero-shot transfer from Norman → K562 (or other target)
- Overlapping gene vocabulary alignment
- Log1p-CPM normalization (matches GEARS prediction space)
- Comprehensive T1/T2/T3 metrics (CPA-aligned)
- Progress tracking during inference
- Proper dependency installation with version checking

Usage
-----
    # Via eval_runner (recommended):
    python -m eval.eval_runner --data K562.h5ad --models gears

    # Direct import:
    from eval.models.gears import run_eval
    result = run_eval(adata, cfg)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import tarfile
import zipfile
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch

from .. import config
from ..metrics import compute_all_metrics

if TYPE_CHECKING:
    import scanpy as sc

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

NORMAN_DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/6979957"
NORMAN_MODEL_URL = "https://dataverse.harvard.edu/api/access/datafile/10457098"
NORMAN_DATA_PATH = "./norman_umi_go"
NORMAN_MODEL_CKPT = "./model_ckpt"
TOP_K_DE = 50
SEED = 1
CTRL_LABEL = "non-targeting"
SUBSAMPLE_FRAC = 0.20
MIN_CELLS_PER_PERT = 5
MAX_T3_CELLS = 512


# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

def _pip(*pkgs: str) -> None:
    """Install packages with --break-system-packages for Colab compatibility."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "--break-system-packages", *pkgs],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Install GEARS and torch_geometric dependencies if needed."""
    # torch_geometric must come BEFORE cell-gears
    try:
        import torch_geometric  # noqa: F401
        logger.info("✅ torch_geometric: already installed")
    except ImportError:
        logger.info("📦 Installing torch_geometric ...")
        _pip("torch_geometric")
        logger.info("✅ torch_geometric: done")

    # torch_scatter/sparse (PyG wheels)
    try:
        import torch_scatter  # noqa: F401
        logger.info("✅ torch_scatter: already installed")
    except ImportError:
        tv = torch.__version__.split("+")[0]
        cud = torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
        try:
            logger.info("📦 Installing torch_scatter/sparse ...")
            _pip("torch_scatter", "torch_sparse",
                 "-f", f"https://data.pyg.org/whl/torch-{tv}+cu{cud}.html")
            logger.info("✅ torch_scatter: done")
        except Exception as e:
            logger.warning("⚠️  torch_scatter install skipped: %s", e)

    # cell-gears and scanpy
    try:
        import gears  # noqa: F401
        logger.info("✅ cell-gears: already installed")
    except ImportError:
        logger.info("📦 Installing cell-gears ...")
        _pip("cell-gears", "scanpy")
        logger.info("✅ cell-gears: done")

    # Monkey-patch pandas 2.x (Series.nonzero removed)
    import pandas as pd
    if not hasattr(pd.Series, "nonzero"):
        pd.Series.nonzero = lambda self: self.to_numpy().nonzero()
        logger.info("✅ Pandas monkey-patch applied (Series.nonzero)")


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def _download_norman_data(data_path: str, model_ckpt: str) -> None:
    """Download Norman dataset and pretrained GEARS model if not cached."""
    from gears.utils import dataverse_download

    if not os.path.exists(data_path):
        logger.info("📥 Downloading Norman data ...")
        dataverse_download(
            NORMAN_DATA_URL,
            "norman_umi_go.tar.gz",
        )
        with tarfile.open("norman_umi_go.tar.gz", "r:gz") as tar:
            tar.extractall()
        logger.info(f"✅ Norman data extracted to {data_path}")
    else:
        logger.info(f"✅ Norman data: found locally ({data_path})")

    if not os.path.exists(model_ckpt):
        logger.info("📥 Downloading pretrained GEARS model ...")
        dataverse_download(
            NORMAN_MODEL_URL,
            "model.zip",
        )
        with zipfile.ZipFile("model.zip", "r") as z:
            z.extractall(path="./")
        logger.info(f"✅ Model extracted to {model_ckpt}")
    else:
        logger.info(f"✅ GEARS model: found locally ({model_ckpt})")


# =============================================================================
# METRIC FUNCTIONS (T1/T2/T3)
# =============================================================================

def _systema_metrics(pred_c: torch.Tensor, true_c: torch.Tensor) -> tuple[float, float]:
    """T1: Centroid Accuracy and Profile Distance Score."""
    n = pred_c.size(0)
    if n < 2:
        return np.nan, np.nan

    dists = torch.cdist(pred_c, true_c, p=2.0)
    d_t = dists.diag()
    ca = (d_t.unsqueeze(1) < dists).sum(dim=1).float() / (n - 1)

    mask = ~torch.eye(n, dtype=torch.bool, device=pred_c.device)
    pds = d_t / (d_t + (dists * mask).sum(dim=1) / (n - 1) + 1e-8)

    return ca.mean().item(), pds.mean().item()


def _systema_pearson_delta(pred_c: torch.Tensor, true_c: torch.Tensor) -> float:
    """T1: Pearson correlation on delta from control."""
    o = true_c.mean(dim=0, keepdim=True)
    pc = pred_c - o
    tc = true_c - o
    pc = pc - pc.mean(-1, keepdim=True)
    tc = tc - tc.mean(-1, keepdim=True)
    cov = (pc * tc).sum(-1)
    denom = ((pc**2).sum(-1).clamp(1e-8).sqrt() *
             (tc**2).sum(-1).clamp(1e-8).sqrt())
    return (cov / denom).mean().item()


def _directional_accuracy(pred_d: torch.Tensor, true_d: torch.Tensor,
                          thr: float = 0.1) -> float:
    """T2: Directional Accuracy (sign agreement on large changes)."""
    active = true_d.abs() > thr
    matches = (torch.sign(pred_d) == torch.sign(true_d)) & active
    return (matches.sum(-1).float() /
            (active.sum(-1).float() + 1e-8)).mean().item()


def _pearson_delta_topk(pred_d: torch.Tensor, true_d: torch.Tensor,
                        k: int = TOP_K_DE) -> float:
    """T2: Pearson on top-k DE genes by absolute delta."""
    k = min(k, pred_d.size(1))
    _, idx = torch.topk(true_d.abs(), k=k, dim=-1)
    pt = torch.gather(pred_d, 1, idx)
    tt = torch.gather(true_d, 1, idx)
    pc = pt - pt.mean(-1, keepdim=True)
    tc = tt - tt.mean(-1, keepdim=True)
    cov = (pc * tc).sum(-1)
    denom = ((pc**2).sum(-1).clamp(1e-8).sqrt() *
             (tc**2).sum(-1).clamp(1e-8).sqrt())
    return (cov / denom).mean().item()


def _de_gene_jaccard(pred_d: torch.Tensor, true_d: torch.Tensor,
                     k: int = TOP_K_DE) -> float:
    """T2: Jaccard index on top-k DE genes."""
    k = min(k, pred_d.size(1))
    _, ti = torch.topk(true_d.abs(), k=k, dim=-1)
    _, pi = torch.topk(pred_d.abs(), k=k, dim=-1)
    scores = []
    for t, p in zip(ti, pi):
        t_set = set(t.tolist())
        p_set = set(p.tolist())
        if len(t_set | p_set) == 0:
            scores.append(0.0)
        else:
            scores.append(len(t_set & p_set) / len(t_set | p_set))
    return float(np.mean(scores))


def _energy_distance_point(pred_pt: torch.Tensor, x_true: torch.Tensor,
                           max_cells: int = MAX_T3_CELLS) -> float:
    """T3: Energy distance (point prediction vs cell distribution)."""
    if x_true.size(0) > max_cells:
        idx = torch.randperm(x_true.size(0), device=x_true.device)[:max_cells]
        x_true = x_true[idx]

    d_pt = torch.cdist(pred_pt, x_true, p=2.0).mean()
    d_tt = torch.cdist(x_true, x_true, p=2.0).mean()
    return max((2 * d_pt - d_tt).item(), 0.0)


def _mmd_rbf_point(pred_pt: torch.Tensor, x_true: torch.Tensor,
                   max_cells: int = MAX_T3_CELLS) -> float:
    """T3: MMD with RBF kernel (median bandwidth heuristic)."""
    if x_true.size(0) > max_cells:
        idx = torch.randperm(x_true.size(0), device=x_true.device)[:max_cells]
        x_true = x_true[idx]

    with torch.no_grad():
        sigma = torch.cdist(x_true, x_true).median().clamp(1e-3).item()

    def rbf(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma ** 2))

    kxx = rbf(pred_pt, pred_pt).mean()
    kyy = rbf(x_true, x_true).mean()
    kxy = rbf(pred_pt, x_true).mean()
    return max((kxx - 2 * kxy + kyy).item(), 0.0)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_eval(adata: "sc.AnnData", cfg: dict) -> dict:
    """
    Run GEARS zero-shot evaluation.

    Parameters
    ----------
    adata : sc.AnnData
        Subsampled dataset with perturbation labels in ``obs[pert_col]``.
    cfg : dict
        Runtime configuration dictionary.

    Returns
    -------
    dict
        Evaluation result with model name, metrics, perturbation names,
        and runtime in seconds.
    """
    warnings.filterwarnings("ignore")
    import scanpy as sc

    t_start = time.time()
    device_str = cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("RANDOM_SEED", SEED)
    ctrl_label = cfg.get("CTRL_LABEL", CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    top_k_de = cfg.get("TOP_K_DE", TOP_K_DE)
    subsample_frac = cfg.get("SUBSAMPLE_FRAC", SUBSAMPLE_FRAC)
    min_cells = cfg.get("MIN_CELLS_PER_PERT", MIN_CELLS_PER_PERT)

    device = torch.device(device_str)
    logger.info(f"🖥️  Device: {device}")

    # --- Install dependencies ------------------------------------------------
    _install_dependencies()
    from gears import PertData, GEARS

    # --- Download Norman data + model ----------------------------------------
    _download_norman_data(NORMAN_DATA_PATH, NORMAN_MODEL_CKPT)

    # --- Load Norman model ---------------------------------------------------
    logger.info("📊 Loading Norman PertData + model ...")
    pert_data = PertData("./")
    pert_data.load(data_path=NORMAN_DATA_PATH)
    pert_data.prepare_split(split="no_test", seed=seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    gears_model = GEARS(
        pert_data, device=device_str,
        weight_bias_track=False, proj_name="gears", exp_name="gears_misc_umi_no_test",
    )
    gears_model.model_initialize(hidden_size=64)
    gears_model.load_pretrained(NORMAN_MODEL_CKPT)

    norman_genes = set(pert_data.gene_names.tolist())
    norman_gene_list = pert_data.gene_names.tolist()
    logger.info(f"✅ Norman vocab: {len(norman_genes)} genes")

    # --- Prepare target data -------------------------------------------------
    logger.info("📁 Preparing target data ...")

    # Create condition column
    adata.obs["condition"] = adata.obs[pert_col].apply(
        lambda g: "ctrl" if g == ctrl_label else f"{g}+ctrl"
    )

    # Ensure gene_name column
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var.index

    # Subsample (stratified by condition)
    np.random.seed(seed)
    keep_idx = []
    for cond, grp in adata.obs.groupby("condition"):
        n = max(2, int(len(grp) * subsample_frac))
        keep_idx.extend(np.random.choice(grp.index, size=n, replace=False).tolist())
    adata = adata[keep_idx].copy()
    logger.info(f"📉 After subsampling ({subsample_frac*100:.0f}%): {adata.shape}")

    # Filter perturbations with >= min_cells
    counts = adata.obs["condition"].value_counts()
    adata = adata[adata.obs["condition"].isin(counts[counts >= min_cells].index)].copy()
    logger.info(f"📉 After min-{min_cells} filter: {adata.shape}")

    # Convert to sparse if needed
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    k562_genes = adata.var["gene_name"].tolist()

    # Log-normalize control cells (log1p CPM to match GEARS space)
    X_ctrl_raw = adata[adata.obs["condition"] == "ctrl"].X.toarray()
    _lib = X_ctrl_raw.sum(axis=1, keepdims=True)
    X_ctrl_ln = np.log1p(X_ctrl_raw / (_lib + 1e-8) * 1e4)
    ctrl_mean_k562 = X_ctrl_ln.mean(axis=0)
    logger.info(f"✅ Control baseline computed on {len(ctrl_mean_k562)} genes")

    # --- Gene & perturbation overlap -----------------------------------------
    overlap_genes = norman_genes & set(k562_genes)
    overlap_sorted = sorted(overlap_genes)
    norman_idx = np.array([norman_gene_list.index(g) for g in overlap_sorted])
    k562_idx = np.array([k562_genes.index(g) for g in overlap_sorted])

    ctrl_mean_overlap = torch.tensor(
        ctrl_mean_k562[k562_idx], dtype=torch.float32
    ).to(device)

    k562_perts = sorted([
        c for c in adata.obs["condition"].unique()
        if c != "ctrl" and c.replace("+ctrl", "") in norman_genes
    ])

    logger.info(f"🧬 Gene overlap: Norman={len(norman_genes)}, "
                f"Target={len(k562_genes)}, Overlap={len(overlap_genes)}")
    logger.info(f"🎯 Predictable perturbations: {len(k562_perts)}")

    # --- Zero-shot inference -------------------------------------------------
    logger.info(f"🚀 Running zero-shot predictions ({len(k562_perts)} perts) ...")

    pred_list, true_list, pert_names = [], [], []
    true_cells_dict, pred_point_dict = {}, {}
    n_skip = 0
    t0 = time.time()

    for i, pert in enumerate(k562_perts):
        gene = pert.replace("+ctrl", "")
        mask = adata.obs["condition"] == pert
        if mask.sum() < 2:
            n_skip += 1
            continue

        # Ground truth — log-normalize to match GEARS prediction space
        X_true_raw = adata[mask].X.toarray()
        _lib = X_true_raw.sum(axis=1, keepdims=True)
        X_true = np.log1p(X_true_raw / (_lib + 1e-8) * 1e4)
        true_mean_overlap = X_true.mean(axis=0)[k562_idx]

        # GEARS prediction
        try:
            pred_dict = gears_model.predict([[gene]])
            pred_full = np.array(list(pred_dict.values())[0])
            pred_overlap = pred_full[norman_idx]
        except Exception as e:
            logger.warning(f"⚠️  Prediction failed for {gene}: {e}")
            n_skip += 1
            continue

        pred_list.append(torch.tensor(pred_overlap, dtype=torch.float32).to(device))
        true_list.append(torch.tensor(true_mean_overlap, dtype=torch.float32).to(device))
        true_cells_dict[pert] = torch.tensor(
            X_true[:, k562_idx], dtype=torch.float32
        ).to(device)
        pred_point_dict[pert] = torch.tensor(
            pred_overlap, dtype=torch.float32
        ).unsqueeze(0).to(device)
        pert_names.append(pert)

        # Progress tracking
        if (i + 1) % 50 == 0 or (i + 1) == len(k562_perts):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(k562_perts) - i - 1) / rate if rate > 0 else 0
            logger.info(f"  {i+1:>4}/{len(k562_perts)} done={len(pert_names)} "
                        f"skip={n_skip} {elapsed:.0f}s elapsed ETA {eta:.0f}s")

    if len(pert_names) == 0:
        raise RuntimeError("No perturbations could be evaluated for GEARS.")

    pred_centroids = torch.stack(pred_list)
    true_centroids = torch.stack(true_list)
    pred_delta = pred_centroids - ctrl_mean_overlap.unsqueeze(0)
    true_delta = true_centroids - ctrl_mean_overlap.unsqueeze(0)

    logger.info(f"✅ Collected: {len(pert_names)} perts | shape {pred_centroids.shape}")
    logger.info(f"⚠️  Skipped: {n_skip}")

    # --- Compute metrics -----------------------------------------------------
    logger.info("📊 Computing T1/T2/T3 metrics ...")

    # T1: Centroid-level metrics
    ca, pds = _systema_metrics(pred_centroids, true_centroids)
    d_sys = _systema_pearson_delta(pred_centroids, true_centroids)
    logger.info(f"✅ T1 complete: CA={ca:.4f}, PDS={pds:.4f}, PearsonΔ={d_sys:.4f}")

    # T2: DE-level metrics
    dir_acc = _directional_accuracy(pred_delta, true_delta)
    pear_de = _pearson_delta_topk(pred_delta, true_delta, k=top_k_de)
    jaccard = _de_gene_jaccard(pred_delta, true_delta, k=top_k_de)
    logger.info(f"✅ T2 complete: DirAcc={dir_acc:.4f}, PearsonDE={pear_de:.4f}, Jaccard={jaccard:.4f}")

    # T3: Distribution-level metrics (point vs cells)
    e_scores, mmd_scores = [], []
    t0_t3 = time.time()
    for i, pert in enumerate(pert_names):
        e_scores.append(_energy_distance_point(pred_point_dict[pert], true_cells_dict[pert]))
        mmd_scores.append(_mmd_rbf_point(pred_point_dict[pert], true_cells_dict[pert]))
        if (i + 1) % 50 == 0 or (i + 1) == len(pert_names):
            elapsed = time.time() - t0_t3
            eta = elapsed / (i + 1) * (len(pert_names) - i - 1)
            logger.info(f"  T3 {i+1:>4}/{len(pert_names)} {elapsed:.0f}s elapsed ETA {eta:.0f}s")

    e_dist = float(np.mean(e_scores))
    mmd_val = float(np.mean(mmd_scores))
    logger.info(f"✅ T3 complete: Energy={e_dist:.4f}, MMD={mmd_val:.4f}")

    # --- Compile results -----------------------------------------------------
    metrics = {
        "centroid_accuracy": ca,
        "profile_distance_score": pds,
        "pearson_delta": d_sys,
        "directional_accuracy": dir_acc,
        "pearson_delta_de": pear_de,
        "jaccard_de": jaccard,
        "energy_distance": e_dist,
        "mmd": mmd_val,
    }

    # Cleanup
    del gears_model, pert_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    runtime = time.time() - t_start
    logger.info(f"⏱️  Total runtime: {runtime/60:.2f} min")

    return {
        "model": "GEARS",
        "metrics": metrics,
        "pert_names": pert_names,
        "n_overlap_genes": len(overlap_sorted),
        "runtime_seconds": runtime,
    }
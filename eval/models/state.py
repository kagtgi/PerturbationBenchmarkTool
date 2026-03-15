"""
STATE evaluation — zero-shot inference using arcinstitute/ST-HVG-Replogle.

Preprocesses data into 2000 HVGs, runs STATE CLI for inference, and
computes standard 3-tier metrics.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

from .. import config
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)

HF_REPO = "arcinstitute/ST-HVG-Replogle"
LOCAL_DIR = "ST-HVG-Replogle"


def _pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "--break-system-packages", *args],
        stdout=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Install STATE CLI and HuggingFace Hub."""
    _pip("uv", "huggingface_hub")

    uv_bin = os.path.expanduser("~/.local/bin")
    if uv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = uv_bin + ":" + os.environ.get("PATH", "")

    if shutil.which("state") is None:
        logger.info("Installing arc-state via uv ...")
        subprocess.check_call(["uv", "tool", "install", "arc-state"])

    # Evict stale scipy/scanpy modules after potential upgrades
    for k in list(sys.modules.keys()):
        if any(k == m or k.startswith(m + ".")
               for m in ("scanpy", "anndata", "scipy", "pandas")):
            del sys.modules[k]


def _download_model() -> tuple[str, str]:
    """Download the STATE model from HuggingFace and return (model_dir, checkpoint)."""
    from huggingface_hub import snapshot_download

    if not os.path.exists(LOCAL_DIR):
        logger.info("Downloading %s ...", HF_REPO)
        snapshot_download(repo_id=HF_REPO, local_dir=LOCAL_DIR)

    ckpts = sorted(glob.glob(f"{LOCAL_DIR}/**/best.ckpt", recursive=True))
    if not ckpts:
        ckpts = sorted(glob.glob(f"{LOCAL_DIR}/**/*.ckpt", recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found under '{LOCAL_DIR}'.")

    checkpoint = ckpts[0]
    model_dir = os.path.dirname(os.path.dirname(checkpoint))
    return model_dir, checkpoint


def run_eval(adata, cfg: dict) -> dict:
    """Run STATE zero-shot evaluation.

    Parameters
    ----------
    adata : sc.AnnData
        Subsampled dataset.
    cfg : dict
        Runtime configuration.

    Returns
    -------
    dict with ``model``, ``metrics``, ``pert_names``, ``runtime_seconds``.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import scanpy as sc

    t_start = time.time()
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)

    # --- Install & download ------------------------------------------------
    _install_dependencies()
    model_dir, checkpoint = _download_model()

    # --- Prepare data ------------------------------------------------------
    adata.obs[pert_col] = adata.obs[pert_col].astype(str)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()

    if "gem_group" not in adata.obs.columns:
        adata.obs["gem_group"] = "batch1"

    raw_path = "_state_raw.h5ad"
    pre_path = "_state_preprocessed.h5ad"
    pred_path = "_state_predicted.h5ad"
    adata.write_h5ad(raw_path)

    # --- STATE preprocess --------------------------------------------------
    logger.info("STATE preprocess (norm -> log1p -> 2000 HVGs) ...")
    subprocess.check_call([
        "state", "tx", "preprocess_train",
        "--adata", raw_path,
        "--output", pre_path,
        "--num_hvgs", "2000",
    ])

    # --- STATE inference ---------------------------------------------------
    logger.info("STATE inference (zero-shot) ...")
    subprocess.check_call([
        "state", "tx", "infer",
        "--model-dir", model_dir,
        "--checkpoint", checkpoint,
        "--adata", pre_path,
        "--pert-col", pert_col,
        "--batch-col", "gem_group",
        "--control-pert", ctrl_label,
        "--embed-key", "X_hvg",
        "--output", pred_path,
    ])

    # --- Load results ------------------------------------------------------
    true_adata = sc.read_h5ad(pre_path)
    pred_adata = sc.read_h5ad(pred_path)

    true_X = np.array(true_adata.obsm["X_hvg"], dtype=np.float32)
    pred_X = np.array(pred_adata.obsm["X_hvg"], dtype=np.float32)
    conditions = pred_adata.obs[pert_col].values

    # --- Build centroids ---------------------------------------------------
    ctrl_mask = conditions == ctrl_label
    ctrl_mu = true_X[ctrl_mask].mean(0)
    ctrl_t = torch.tensor(ctrl_mu, dtype=torch.float32)

    pred_list, true_list, pert_names = [], [], []
    pred_cells_d, true_cells_d = {}, {}

    for pert in np.unique(conditions):
        if pert == ctrl_label:
            continue
        mask = conditions == pert
        if mask.sum() < 2:
            continue
        pred_list.append(torch.tensor(pred_X[mask].mean(0), dtype=torch.float32))
        true_list.append(torch.tensor(true_X[mask].mean(0), dtype=torch.float32))
        pred_cells_d[pert] = torch.tensor(pred_X[mask], dtype=torch.float32)
        true_cells_d[pert] = torch.tensor(true_X[mask], dtype=torch.float32)
        pert_names.append(pert)

    pred_c = torch.stack(pred_list)
    true_c = torch.stack(true_list)

    logger.info("STATE: %d perts, %d HVG features", len(pert_names), pred_X.shape[1])

    # --- Metrics -----------------------------------------------------------
    metrics = compute_all_metrics(
        pred_c, true_c, ctrl_t,
        pred_cells_dict=pred_cells_d,
        true_cells_dict=true_cells_d,
        pert_names=pert_names,
    )

    # Cleanup temp files
    for p in (raw_path, pre_path, pred_path):
        if os.path.exists(p):
            os.remove(p)

    return {
        "model": "STATE",
        "metrics": metrics,
        "pert_names": pert_names,
        "runtime_seconds": time.time() - t_start,
    }

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
    pert_data.prepare_split(split="no_test", seed=seed)
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
    metrics = compute_all_metrics(
        pred_centroids, true_centroids, ctrl_mean_overlap,
        pred_cells_dict=pred_point_dict,
        true_cells_dict=true_cells_dict,
        pert_names=pert_names,
    )

    return {
        "model": "GEARS",
        "metrics": metrics,
        "pert_names": pert_names,
        "n_overlap_genes": len(overlap_sorted),
        "runtime_seconds": time.time() - t_start,
    }

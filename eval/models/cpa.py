"""
CPA (Compositional Perturbation Autoencoder) evaluation.

Full production pipeline: clones theislab/cpa, patches for compatibility,
loads a pretrained K562 checkpoint, runs predictions, and computes
3-tier metrics in log1p-normalized space.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch

from .. import config
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)

CPA_DIR = "/tmp/theislab_cpa"
PRETRAINED_PT = os.path.join("./pretrained_cpa_k562", "k562_model.pt")
MODEL_DIR = "./pretrained_cpa_k562"
KANG_MODEL_ID = "1IVDsxkCZZlU5MCyiu0MKyAwzeEd_yV4B"

MAX_EVAL_PERTS = 100
MAX_CELLS_SAMPLE = 150


def _pip(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + list(pkgs),
        stdout=subprocess.DEVNULL,
    )


def _pip_no_upgrade(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs),
        stdout=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Clone CPA repo and install all dependencies."""
    if os.path.exists(CPA_DIR):
        shutil.rmtree(CPA_DIR)
    subprocess.check_call(["git", "clone", "-q",
                           "https://github.com/theislab/cpa.git", CPA_DIR])

    # Capture current versions BEFORE any pip operations so we can restore
    # them afterwards.  numba and scvi-tools transitively upgrade scipy; the
    # new scipy wheel's Cython extensions (_upfirdn_apply.pyx) may be compiled
    # against a different _cyutility API than what's on disk, causing
    # "does not export expected C function __Pyx__Import" on re-import.
    NP_VER = np.__version__
    import scipy as _sc_snap
    SC_VER = _sc_snap.__version__
    del _sc_snap

    _pip("anndata>=0.10.0,<0.13.0")
    _pip(f"numpy=={NP_VER}")
    _pip("numba>=0.60.0")
    _pip("scanpy>=1.10.0,<1.11.0")
    _pip("scvi-tools>=1.0.0,<1.5.0")
    _pip("lightning>=2.2.0,<2.4.0")
    _pip("pytorch-lightning>=2.2.0,<2.4.0")
    _pip("pandas", "tqdm", "gdown")
    _pip_no_upgrade("scikit-learn")
    _pip("rdkit", "adjustText", "seaborn")
    _pip("pybiomart")
    # Restore scipy to the pre-install version to avoid any ABI mismatch
    # introduced by the installs above.
    _pip(f"scipy=={SC_VER}")


def _clear_module_cache() -> None:
    """Remove stale modules from sys.modules."""
    CLEAR = ["anndata", "scvi", "scanpy", "cpa", "lightning", "jax", "flax",
             "pytorch_lightning", "torchmetrics", "numba", "scipy"]
    for k in list(sys.modules):
        if any(k == m or k.startswith(m + ".") for m in CLEAR):
            sys.modules.pop(k, None)


def _patch_scvi() -> None:
    """Apply compatibility patches to scvi-tools."""
    spec = importlib.util.find_spec("scvi")
    if not spec or not spec.submodule_search_locations:
        return
    p = spec.submodule_search_locations[0]

    # Patch _types.py
    types_path = os.path.join(p, "_types.py")
    if os.path.exists(types_path):
        with open(types_path, "w") as f:
            f.write(
                "from typing import Union\nimport torch\n"
                "Tensor = torch.Tensor\n"
                "AnnOrMuData = 'anndata.AnnData'\n"
                "Number = Union[int, float]\n"
                "LossRecord = dict\n"
                "MinifiedDataType = str\n"
            )

    # Patch train/__init__.py — remove SaveBestState if it fails
    train_init = os.path.join(p, "train", "__init__.py")
    if os.path.exists(train_init):
        with open(train_init) as f:
            content = f.read()
        content = re.sub(
            r"from\s+\._callbacks\s+import\s+SaveBestState",
            "# removed SaveBestState", content,
        )
        with open(train_init, "w") as f:
            f.write(content)


def _install_jax_stub() -> None:
    """Install a lightweight JAX/FLAX stub so CPA can import without real JAX."""
    import types as _t

    class _JaxStub(_t.ModuleType):
        def __getattr__(self, name):
            return _JaxStub(name)
        def __call__(self, *a, **k):
            return lambda fn: fn

    mods = [
        "jax", "jax.numpy", "jax.random", "jax.config", "jax.interpreters",
        "jax.lib", "jax.dlpack", "flax", "flax.linen", "flax.training",
        "flax.serialization", "flax.struct", "flax.core",
        "flax.training.train_state", "flax.optim", "jaxlib",
        "jaxlib.xla_extension",
    ]
    for m in mods:
        sys.modules.setdefault(m, _JaxStub(m))

    class _Arr:
        __or__ = __ror__ = lambda s, o: o
    sys.modules["jax.numpy"].ndarray = _Arr()

    class _TS:
        __init__ = lambda s, *a, **k: None
        apply_gradients = lambda s, *a, **k: s
    sys.modules["flax.training.train_state"].TrainState = _TS
    sys.modules["flax.struct"].dataclass = lambda *a, **k: (lambda cls: cls)


def _patch_cpa_source() -> None:
    """Patch CPA source files for compatibility with current scvi-tools."""
    # Patch __init__.py
    with open(os.path.join(CPA_DIR, "cpa", "__init__.py"), "w") as f:
        f.write(
            'import warnings; warnings.simplefilter("ignore")\n'
            "from ._model import CPA\n"
            "from ._module import CPAModule\n"
            "from . import _plotting as pl\n"
            "try: from ._api import ComPertAPI\nexcept: ComPertAPI = None\n"
            'try: from ._tuner import run_autotune\nexcept: run_autotune = None\n'
            '__version__ = "0.8.8"\n'
        )

    # Patch _data.py — replace parse_use_gpu_arg
    dp = os.path.join(CPA_DIR, "cpa", "_data.py")
    with open(dp) as f:
        dc = f.read()
    if "from scvi.model._utils import parse_use_gpu_arg" in dc:
        dc = dc.replace(
            "from scvi.model._utils import parse_use_gpu_arg",
            "import torch as _ct\n"
            "def parse_use_gpu_arg(use_gpu, return_device=False):\n"
            '    a, d = ("gpu", _ct.device("cuda")) if (use_gpu and _ct.cuda.is_available()) '
            'else ("cpu", _ct.device("cpu"))\n'
            "    return (a, None, d) if return_device else a",
        )
    with open(dp, "w") as f:
        f.write(dc)

    # Patch _model.py — remove SaveBestState
    mp = os.path.join(CPA_DIR, "cpa", "_model.py")
    with open(mp) as f:
        mc = f.read()
    mc = re.sub(
        r"from scvi\.train\._callbacks import SaveBestState",
        "# removed", mc,
    )
    mc = re.sub(
        r"from scvi\.model\._utils import parse_use_gpu_arg",
        "import torch as _ct\n"
        "def parse_use_gpu_arg(use_gpu, return_device=False):\n"
        '    a, d = ("gpu", _ct.device("cuda")) if (use_gpu and _ct.cuda.is_available()) '
        'else ("cpu", _ct.device("cpu"))\n'
        "    return (a, None, d) if return_device else a",
        mc,
    )
    with open(mp, "w") as f:
        f.write(mc)


def _download_checkpoint(seed: int) -> dict:
    """Download pretrained CPA checkpoint. Returns the state dict."""
    if not os.path.exists(PRETRAINED_PT):
        import gdown
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("Downloading pretrained CPA K562 checkpoint ...")
        gdown.download(id=KANG_MODEL_ID,
                       output=PRETRAINED_PT, quiet=False)

    sd_raw = torch.load(PRETRAINED_PT, map_location="cpu", weights_only=False)

    # Normalize state dict structure
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
        attr = sd_raw.get("attr_dict", {})
        arch = sd_raw.get("model_state_dict", sd_raw.get("hyper_parameters", {}))
        sd = sd_raw["state_dict"]
    elif isinstance(sd_raw, dict) and any(k.startswith("module.") or "encoder" in k for k in sd_raw):
        sd = sd_raw
        attr = {}
        arch = {}
    else:
        sd = sd_raw
        attr = {}
        arch = {}

    # Strip "module." prefix
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    return {"sd": sd, "attr": attr, "arch": arch}


def run_eval(adata, cfg: dict) -> dict:
    """Run CPA evaluation.

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
    import pandas as pd
    import scanpy as sc
    from scipy.spatial.distance import cdist
    from sklearn.metrics import r2_score
    from tqdm import tqdm

    t_start = time.time()
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    top_k = cfg.get("TOP_K_DE", config.TOP_K_DE)
    device_str = cfg.get("DEVICE", config.DEVICE)
    rng = np.random.default_rng(seed)

    # --- Install & patch ---------------------------------------------------
    _install_dependencies()
    _clear_module_cache()
    _patch_scvi()
    _install_jax_stub()
    _patch_cpa_source()

    if CPA_DIR not in sys.path:
        sys.path.insert(0, CPA_DIR)

    import cpa

    # --- Download checkpoint -----------------------------------------------
    ckpt = _download_checkpoint(seed)
    sd, attr, arch = ckpt["sd"], ckpt["attr"], ckpt["arch"]

    # Detect checkpoint gene names for mapping
    ckpt_var_names = attr.get("var_names", None)
    if ckpt_var_names is not None and not isinstance(ckpt_var_names, list):
        ckpt_var_names = list(ckpt_var_names)

    # --- Prepare adata -----------------------------------------------------
    if "perturbation" not in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs[pert_col].astype(str)

    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "K562"

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Gene name mapping (Ensembl → symbol if needed)
    _ensembl_to_symbol = None
    if ckpt_var_names:
        h5_names = set(adata.var_names)
        ckpt_set = set(ckpt_var_names)
        if len(h5_names & ckpt_set) < 50:
            # Try mapping via var columns
            for col in ["gene_name", "gene_names", "gene_symbols", "symbol"]:
                if col in adata.var.columns:
                    _ensembl_to_symbol = dict(
                        zip(adata.var_names, adata.var[col].astype(str))
                    )
                    break
            if _ensembl_to_symbol:
                mapped = set(_ensembl_to_symbol.values()) & ckpt_set
                if len(mapped) > 50:
                    adata.var_names = pd.Index(
                        [_ensembl_to_symbol.get(g, g) for g in adata.var_names]
                    )
                    adata.var_names_make_unique()

    # Filter to shared genes with checkpoint if possible
    if ckpt_var_names:
        shared = sorted(set(adata.var_names) & set(ckpt_var_names))
        if shared:
            adata = adata[:, shared].copy()

    adata.obs["cov_cond"] = "K562_" + adata.obs["perturbation"].astype(str)

    # --- DEGs --------------------------------------------------------------
    if sp.issparse(adata.X):
        _mean = np.asarray(adata.X.mean(axis=0)).flatten()
        _sq = np.asarray(adata.X.power(2).mean(axis=0)).flatten()
        real_var = _sq - _mean ** 2
    else:
        real_var = np.asarray(adata.X.var(axis=0)).flatten()

    real_mask = real_var > 1e-8
    real_gene_names = adata.var_names[real_mask]
    real_gene_idx = np.where(real_mask)[0]

    # Drop tiny groups
    grp_counts = adata.obs["perturbation"].value_counts()
    small = [g for g in grp_counts[grp_counts < 3].index if g != ctrl_label]
    if small:
        adata = adata[~adata.obs["perturbation"].isin(small)].copy()

    adata_rgg = adata[:, real_gene_names].copy()
    sc.pp.normalize_total(adata_rgg, target_sum=1e4)
    sc.pp.log1p(adata_rgg)

    n_deg = min(200, len(real_gene_names))
    for method in ("t-test", "wilcoxon"):
        try:
            sc.tl.rank_genes_groups(
                adata_rgg, groupby="perturbation", reference=ctrl_label,
                method=method, n_genes=n_deg, key_added="rank_genes_groups",
                use_raw=False,
            )
            rgg = adata_rgg.uns.get("rank_genes_groups")
            if rgg and "names" in rgg and len(rgg["names"]) > 0:
                break
        except Exception:
            continue

    adata.uns["rank_genes_groups"] = adata_rgg.uns["rank_genes_groups"]
    del adata_rgg

    adata.uns["rank_genes_groups_cov"] = {
        f"K562_{g}": list(adata.uns["rank_genes_groups"]["names"][g])
        for g in adata.uns["rank_genes_groups"]["names"].dtype.names
        if g != ctrl_label
    }

    # Splits
    all_perts = [p for p in adata.obs["perturbation"].unique() if p != ctrl_label]
    ood = rng.choice(all_perts, max(1, int(len(all_perts) * 0.1)),
                     replace=False).tolist() if all_perts else []
    adata.obs["split"] = rng.choice(["train", "valid"], adata.n_obs, p=[0.85, 0.15])
    if ood:
        adata.obs.loc[adata.obs["perturbation"].isin(ood), "split"] = "ood"

    # --- Setup CPA model ---------------------------------------------------
    cpa.CPA.setup_anndata(
        adata, perturbation_key="perturbation", control_group=ctrl_label,
        categorical_covariate_keys=["cell_type"], is_count_data=True,
        deg_uns_key="rank_genes_groups_cov", deg_uns_cat_key="cov_cond",
        max_comb_len=1,
    )

    # Extract architecture from checkpoint
    enc_hidden, dec_hidden = None, None
    enc_layers, dec_layers = 0, 0
    for k, v in sd.items():
        if "encoder" in k and "fc_layers.Layer" in k and k.endswith("0.bias"):
            enc_hidden = v.shape[0]
            m = re.search(r"Layer (\d+)", k)
            if m:
                enc_layers = max(enc_layers, int(m.group(1)) + 1)
        if "decoder" in k and "fc_layers.Layer" in k and k.endswith("0.bias"):
            dec_hidden = v.shape[0]
            m = re.search(r"Layer (\d+)", k)
            if m:
                dec_layers = max(dec_layers, int(m.group(1)) + 1)

    cpa_kw = {}
    for k in ("n_latent", "n_layers_encoder", "n_hidden_encoder",
              "dropout_rate", "use_batch_norm",
              "n_hidden_decoder", "n_layers_decoder"):
        if k in attr:
            cpa_kw[k] = attr[k]

    if "n_latent" not in cpa_kw and "n_latent" in arch:
        cpa_kw["n_latent"] = arch["n_latent"]
    if enc_hidden and "n_hidden_encoder" not in cpa_kw:
        cpa_kw["n_hidden_encoder"] = enc_hidden
    if enc_layers and "n_layers_encoder" not in cpa_kw:
        cpa_kw["n_layers_encoder"] = enc_layers
    if dec_hidden and "n_hidden_decoder" not in cpa_kw:
        cpa_kw["n_hidden_decoder"] = dec_hidden
    if dec_layers and "n_layers_decoder" not in cpa_kw:
        cpa_kw["n_layers_decoder"] = dec_layers

    # Remove unsupported kwargs
    mod_sig = set(inspect.signature(cpa.CPAModule.__init__).parameters.keys()) - {"self"}
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(cpa.CPA.__init__).parameters.values()
    )
    cpa_kw.pop("n_hidden", None)
    if not has_kwargs:
        for k in list(cpa_kw):
            if k not in mod_sig:
                cpa_kw.pop(k)

    model = cpa.CPA(adata=adata, **cpa_kw)

    # Shape-filtered state_dict loading
    model_sd = model.module.state_dict()
    filtered_sd = {
        k: v for k, v in sd.items()
        if k in model_sd and model_sd[k].shape == v.shape
    }
    model.module.load_state_dict(filtered_sd, strict=False)
    logger.info("Loaded %d/%d parameter tensors", len(filtered_sd), len(model_sd))

    try:
        model.to_device(device_str)
    except TypeError:
        model.module.to(torch.device(device_str))

    # --- Predictions -------------------------------------------------------
    adata.layers["X_true"] = adata.X.copy()

    ctrl_mask = adata.obs["perturbation"] == ctrl_label
    ctrl_sub = adata[ctrl_mask].copy()
    if ctrl_sub.n_obs > 0:
        cX = ctrl_sub.X.toarray() if sp.issparse(ctrl_sub.X) else np.array(ctrl_sub.X)
        samp = cX[rng.choice(ctrl_sub.n_obs, size=adata.n_obs, replace=True)]
    else:
        cX = adata.layers["counts"]
        cX = cX.toarray() if sp.issparse(cX) else np.array(cX)
        samp = np.tile(cX.mean(0, keepdims=True), (adata.n_obs, 1))

    s32 = samp.astype(np.float32)
    adata.X = sp.csr_matrix(s32) if sp.issparse(adata.layers["X_true"]) else s32

    try:
        model.to_device(device_str)
    except TypeError:
        model.module.to(torch.device(device_str))

    model.predict(adata, batch_size=512)

    # Find predictions
    pred_found = False
    for loc, d in [("obsm", adata.obsm), ("layers", adata.layers)]:
        for k in list(d.keys()):
            if "pred" in k.lower() or k == "CPA_pred":
                adata.layers["CPA_pred"] = np.array(d[k])
                pred_found = True
                break
        if pred_found:
            break

    if not pred_found:
        raise KeyError("CPA predictions not found in adata.obsm or adata.layers.")

    adata.X = adata.layers["X_true"].copy()

    # --- Normalize to log1p space ------------------------------------------
    true_raw = adata.layers["counts"]
    # Convert sparse to dense before element-wise division; scipy sparse
    # matrices don't support broadcasting division by a 2D array.
    if sp.issparse(true_raw):
        true_raw = true_raw.toarray()
    true_raw = np.asarray(true_raw, dtype=np.float32)
    lib_true = true_raw.sum(axis=1, keepdims=True)
    lib_true = np.maximum(lib_true, 1.0)
    true_log = np.log1p((true_raw / lib_true) * 1e4)

    pred_raw = adata.layers["CPA_pred"]
    lib_pred = np.asarray(pred_raw.sum(axis=1)).flatten()
    lib_pred = np.maximum(lib_pred, 1.0)
    pred_norm = (pred_raw / lib_pred[:, None]) * 1e4
    if not isinstance(pred_norm, np.ndarray):
        pred_norm = np.asarray(pred_norm)
    pred_log = np.log1p(pred_norm).astype(np.float32)

    adata.layers["counts_log"] = true_log
    adata.layers["CPA_pred_log"] = pred_log

    ctrl_for_mu = adata[adata.obs["perturbation"] == ctrl_label]
    ctrl_mu = (ctrl_for_mu.layers["counts_log"].mean(0)
               if ctrl_for_mu.n_obs > 0 else true_log.mean(0))

    # --- 3-tier metrics ----------------------------------------------------
    eval_perts = [
        c for c in adata.obs["perturbation"].unique()
        if c != ctrl_label
        and f"K562_{c}" in adata.uns.get("rank_genes_groups_cov", {})
    ]
    if len(eval_perts) > MAX_EVAL_PERTS:
        eval_counts = [(c, (adata.obs["perturbation"] == c).sum()) for c in eval_perts]
        eval_counts.sort(key=lambda x: x[1], reverse=True)
        eval_perts = [c for c, _ in eval_counts[:MAX_EVAL_PERTS]]

    if not eval_perts:
        return {
            "model": "CPA",
            "metrics": {},
            "pert_names": [],
            "runtime_seconds": time.time() - t_start,
        }

    # Build centroids
    PC, TC, CN = [], [], []
    for c in eval_perts:
        m = adata.obs["perturbation"] == c
        xt = adata[m].layers["counts_log"][:, real_gene_idx]
        xp = adata[m].layers["CPA_pred_log"][:, real_gene_idx]
        PC.append(xp.mean(0))
        TC.append(xt.mean(0))
        CN.append(c)

    P = len(CN)
    PM = np.stack(PC)
    TM = np.stack(TC)

    # Centroid Accuracy & PDS
    D = np.linalg.norm(PM[:, None] - TM[None], axis=-1)
    ca_v = (D.argmin(1) == np.arange(P)).astype(float)
    d_self = D[np.arange(P), np.arange(P)]
    off = ~np.eye(P, dtype=bool)
    d_cross = (D * off).sum(1) / off.sum(1) if P > 1 else np.full(P, np.nan)
    pds_v = d_self / (d_self + d_cross + 1e-8)

    # Per-perturbation T1/T2/T3
    def _pearson(a, b):
        if a.std() < 1e-8 or b.std() < 1e-8:
            return np.nan
        r = np.corrcoef(a, b)[0, 1]
        return float(r) if not np.isnan(r) else np.nan

    def _da(pm, tm, cm, thr=0.1):
        dp, dt = pm - cm, tm - cm
        m = np.abs(dt) > thr
        return float(np.mean(np.sign(dp[m]) == np.sign(dt[m]))) if m.any() else np.nan

    def _jaccard(pm, tm, cm, k=50):
        pt = set(np.argsort(np.abs(pm - cm))[-k:].tolist())
        tt = set(np.argsort(np.abs(tm - cm))[-k:].tolist())
        u = pt | tt
        return len(pt & tt) / len(u) if u else 0.0

    def _energy(p, q, n=MAX_CELLS_SAMPLE):
        r = np.random.default_rng(0)
        if len(p) < 2 or len(q) < 2:
            return np.nan
        p = p[r.choice(len(p), min(n, len(p)), replace=False)].astype(np.float32)
        q = q[r.choice(len(q), min(n, len(q)), replace=False)].astype(np.float32)
        return max(float(2 * cdist(p, q).mean() - cdist(p, p).mean()
                         - cdist(q, q).mean()), 0.0)

    def _mmd(p, q, n=MAX_CELLS_SAMPLE):
        r = np.random.default_rng(0)
        if len(p) < 2 or len(q) < 2:
            return np.nan
        p = p[r.choice(len(p), min(n, len(p)), replace=False)].astype(np.float32)
        q = q[r.choice(len(q), min(n, len(q)), replace=False)].astype(np.float32)
        dqq = cdist(q, q, "sqeuclidean")
        off_d = dqq[~np.eye(len(q), dtype=bool)]
        s2 = max(float(np.median(off_d)) / 2.0 if len(off_d) else 1.0, 1e-6)
        Kpp = np.exp(-cdist(p, p, "sqeuclidean") / (2 * s2)).mean()
        Kqq = np.exp(-dqq / (2 * s2)).mean()
        Kpq = np.exp(-cdist(p, q, "sqeuclidean") / (2 * s2)).mean()
        return max(float(Kpp - 2 * Kpq + Kqq), 0.0)

    res = defaultdict(list)
    ctrl_real = ctrl_mu[real_gene_idx]
    for i, c in enumerate(tqdm(CN, desc="CPA 3-tier", ncols=70)):
        m = adata.obs["perturbation"] == c
        xt = adata[m].layers["counts_log"][:, real_gene_idx]
        xp = adata[m].layers["CPA_pred_log"][:, real_gene_idx]
        dk = f"K562_{c}"
        deg_list = adata.uns["rank_genes_groups_cov"].get(dk, [])
        de = np.where(np.isin(adata.var_names[real_gene_idx], deg_list[:top_k]))[0]

        pm, tm = xp.mean(0), xt.mean(0)
        res["condition"].append(c)
        res["T1_CA"].append(ca_v[i])
        res["T1_PDS"].append(pds_v[i])
        res["T1_PR"].append(_pearson(pm - ctrl_real, tm - ctrl_real))
        res["T2_DA"].append(_da(pm, tm, ctrl_real))
        res["T2_PRde"].append(
            _pearson(pm[de] - ctrl_real[de], tm[de] - ctrl_real[de]) if len(de) > 0 else np.nan
        )
        res["T2_JC"].append(_jaccard(pm, tm, ctrl_real, top_k))
        res["T3_EN"].append(_energy(xp, xt))
        res["T3_MM"].append(_mmd(xp, xt))

    df = pd.DataFrame(res)

    metrics = {
        "T1_Centroid_Accuracy": float(df["T1_CA"].mean()),
        "T1_Profile_Distance_Score": float(df["T1_PDS"].mean()),
        "T1_Systema_Pearson_Delta": float(df["T1_PR"].mean()),
        "T2_Directional_Accuracy": float(df["T2_DA"].mean()),
        "T2_Pearson_Delta_TopK": float(df["T2_PRde"].mean()),
        "T2_Jaccard_TopK": float(df["T2_JC"].mean()),
        "T3_Energy_Distance": float(df["T3_EN"].mean()),
        "T3_MMD_RBF": float(df["T3_MM"].mean()),
    }

    return {
        "model": "CPA",
        "metrics": metrics,
        "pert_names": CN,
        "runtime_seconds": time.time() - t_start,
    }

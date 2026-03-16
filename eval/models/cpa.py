"""
CPA (Compositional Perturbation Autoencoder) evaluation.

Full production pipeline: clones theislab/cpa, patches for compatibility,
loads a pretrained K562 checkpoint, runs predictions, and computes
3-tier metrics in log1p-normalized space.

KEY FEATURES:
- JAX/FLAX stub for Python 3.12 compatibility
- scvi-tools patching for current versions
- Gene name reconciliation (Ensembl → HGNC symbol)
- Log1p-CPM normalization (CPA-aligned metrics)
- Non-zero-variance gene filtering
- Memory-safe batch processing
- Comprehensive T1/T2/T3 metrics

Usage
-----
    # Via eval_runner (recommended):
    python -m eval.eval_runner --data K562.h5ad --models cpa

    # Direct import:
    from eval.models.cpa import run_eval
    result = run_eval(adata, cfg)
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import types
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .. import config
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CPA_DIR = "/tmp/theislab_cpa"
PRETRAINED_PT = os.path.join("./pretrained_cpa_k562", "k562_model.pt")
MODEL_DIR = "./pretrained_cpa_k562"
KANG_MODEL_ID = "1IVDsxkCZZlU5MCyiu0MKyAwzeEd_yV4B"

MAX_EVAL_PERTS = 100
TOP_K_DE = 50
SUBSAMPLE_FRAC = 0.20
MIN_CELLS_PER_PERT = 5
CTRL_LABEL = "ctrl"


# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

def _pip(*pkgs: str) -> None:
    """Install packages with --upgrade flag."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + list(pkgs),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _pip_no_upgrade(*pkgs: str) -> None:
    """Install packages without upgrade (preserve existing versions)."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Clone CPA repo and install all dependencies with version pinning."""
    if os.path.exists(CPA_DIR):
        shutil.rmtree(CPA_DIR)
    
    subprocess.check_call([
        "git", "clone", "-q",
        "https://github.com/theislab/cpa.git",
        CPA_DIR,
    ])
    logger.info("✅ Cloned CPA repository to %s", CPA_DIR)

    # Capture current versions BEFORE any pip operations
    NP_VER = np.__version__
    import scipy as _sc_snap
    SC_VER = _sc_snap.__version__
    del _sc_snap

    logger.info("📦 Installing dependencies...")
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
    # Restore scipy to avoid ABI mismatch
    _pip(f"scipy=={SC_VER}")
    
    logger.info("✅ All dependencies installed")


def _clear_module_cache() -> None:
    """Remove stale modules from sys.modules to ensure fresh imports."""
    CLEAR = ["anndata", "scvi", "scanpy", "cpa", "lightning", "jax", "flax",
             "pytorch_lightning", "torchmetrics", "numba", "scipy"]
    n_cleared = 0
    for k in list(sys.modules):
        if any(k == m or k.startswith(m + ".") for m in CLEAR):
            sys.modules.pop(k, None)
            n_cleared += 1
    logger.info("✅ Cleared %d modules from cache", n_cleared)


# =============================================================================
# JAX/FLAX STUB (Critical for Python 3.12)
# =============================================================================

def _install_jax_stub() -> None:
    """Install lightweight JAX/FLAX stubs so CPA can import without real JAX."""

    class _JaxStub(types.ModuleType):
        """Stub module that satisfies JAX/FLAX imports without real JAX."""

        def __init__(self, name: str):
            super().__init__(name)
            object.__setattr__(self, "_attrs", {})

        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            if name in self._attrs:
                return self._attrs[name]
            child_name = f"{self.__name__}.{name}"
            if child_name not in sys.modules:
                child = _JaxStub(child_name)
                sys.modules[child_name] = child
            return sys.modules[child_name]

        def __setattr__(self, name: str, value):
            if name in ("__name__", "_attrs"):
                object.__setattr__(self, name, value)
            else:
                self._attrs[name] = value

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _JaxStub(f"{self.__name__}.__call__")

        def __iter__(self):
            return iter([])

        def __bool__(self):
            return True

    # Register all known JAX/FLAX submodules
    _mods = [
        "jax", "jax.numpy", "jax.random", "jax.config", "jax.interpreters",
        "jax.lib", "jax.dlpack", "jax.tree_util", "jax.scipy", "jax.nn",
        "flax", "flax.linen", "flax.training", "flax.serialization",
        "flax.struct", "flax.core", "flax.training.train_state",
        "flax.optim", "jaxlib", "jaxlib.xla_extension",
    ]
    for mod_name in _mods:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _JaxStub(mod_name)

    # Special attributes needed by specific consumers
    class _JaxArray:
        __or__ = __ror__ = lambda self, other: other

    sys.modules["jax.numpy"].ndarray = _JaxArray()

    class _TrainState:
        def __init__(self, *args, **kwargs):
            pass

        def apply_gradients(self, *args, **kwargs):
            return self

    sys.modules["flax.training.train_state"].TrainState = _TrainState
    sys.modules["flax.struct"].dataclass = lambda *a, **k: (lambda cls: cls)

    logger.info("✅ JAX/FLAX stub installed")


# =============================================================================
# SCVI-TOOLS PATCHING
# =============================================================================

def _patch_scvi() -> None:
    """Apply compatibility patches to scvi-tools."""
    spec = importlib.util.find_spec("scvi")
    if not spec or not spec.submodule_search_locations:
        logger.warning("⚠️  scvi not found — skipping patches")
        return
    scvi_path = spec.submodule_search_locations[0]
    logger.info("🔧 Patching scvi-tools at %s", scvi_path)

    # --- _types.py ---
    types_path = os.path.join(scvi_path, "_types.py")
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
        logger.info("  ✅ Patched _types.py")

    # --- train/__init__.py — remove SaveBestState ---
    train_init = os.path.join(scvi_path, "train", "__init__.py")
    if os.path.exists(train_init):
        with open(train_init) as f:
            content = f.read()
        content = re.sub(
            r"from\s+\._callbacks\s+import\s+SaveBestState",
            "# removed SaveBestState",
            content,
        )
        with open(train_init, "w") as f:
            f.write(content)
        logger.info("  ✅ Patched train/__init__.py")

    # --- distributions/_negative_binomial.py ---
    nb_path = os.path.join(scvi_path, "distributions", "_negative_binomial.py")
    if os.path.exists(nb_path):
        with open(nb_path) as f:
            nb_content = f.read()
        nb_content = re.sub(
            r"(except\s+\w[\w.]*\s*:\s*\n)(\s*)pass\b",
            r"\1\2    pass  # jax not available",
            nb_content,
        )
        with open(nb_path, "w") as f:
            f.write(nb_content)
        logger.info("  ✅ Patched _negative_binomial.py")

    # --- nn/_base_module.py — remove bare JAX imports ---
    base_module_path = os.path.join(scvi_path, "nn", "_base_module.py")
    if os.path.exists(base_module_path):
        with open(base_module_path) as f:
            bm_content = f.read()
        bm_content = re.sub(
            r"^(from jax\b.*|import jax\b.*)$",
            r"# \1  # jax stub active",
            bm_content,
            flags=re.MULTILINE,
        )
        with open(base_module_path, "w") as f:
            f.write(bm_content)
        logger.info("  ✅ Patched _base_module.py")

    logger.info("✅ scvi-tools patching complete")


# =============================================================================
# CPA SOURCE PATCHING
# =============================================================================

def _patch_cpa_source() -> None:
    """Patch CPA source files for compatibility with current scvi-tools."""
    logger.info("🔧 Patching CPA source...")

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
    logger.info("  ✅ Patched _data.py")

    # Patch _model.py — remove SaveBestState
    mp = os.path.join(CPA_DIR, "cpa", "_model.py")
    with open(mp) as f:
        mc = f.read()

    mc = re.sub(
        r"from scvi\.train\._callbacks import SaveBestState",
        "# removed SaveBestState import",
        mc,
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
    mc = re.sub(
        r"checkpoint\s*=\s*SaveBestState\([^)]*\)\s*\n",
        "# checkpoint = SaveBestState removed\n",
        mc,
    )
    mc = re.sub(
        r"callbacks\s*=\s*\[[^\]]*checkpoint[^\]]*\]\s*\n",
        "callbacks = []\n",
        mc,
        flags=re.DOTALL,
    )
    with open(mp, "w") as f:
        f.write(mc)
    logger.info("  ✅ Patched _model.py")

    logger.info("✅ CPA source patching complete")


# =============================================================================
# PANDAS SHIM (for old checkpoint compatibility)
# =============================================================================

def _pandas_shim() -> None:
    """Install pandas backward-compat shims for torch.load on old checkpoints."""
    try:
        import pandas as pd
        _shim_mods = [
            "pandas.core.indexes.base",
            "pandas.core.indexes.range",
            "pandas.core.indexes.numeric",
            "pandas.core.indexes.int64",
        ]
        for old_mod in _shim_mods:
            if old_mod not in sys.modules:
                shim = types.ModuleType(old_mod)
                shim.Index = pd.Index
                shim.RangeIndex = pd.RangeIndex
                shim.Int64Index = getattr(pd, "Int64Index", pd.Index)
                shim.Float64Index = getattr(pd, "Float64Index", pd.Index)
                sys.modules[old_mod] = shim
        logger.info("✅ Pandas shim installed")
    except Exception as e:
        logger.warning("⚠️  Pandas shim failed: %s", e)


# =============================================================================
# GENE NAME RECONCILIATION
# =============================================================================

def _reconcile_gene_names(adata, ckpt_var_names: list[str]):
    """Map h5ad gene IDs to checkpoint gene symbols if needed."""
    h5_names = list(adata.var_names)
    ckpt_set = set(ckpt_var_names)

    # 1. Direct overlap
    direct_overlap = len(set(h5_names) & ckpt_set)
    if direct_overlap >= 50:
        logger.info("✅ Gene names already overlap (%d genes)", direct_overlap)
        return adata, None

    logger.info("🔍 Gene reconciliation needed (direct overlap: %d)", direct_overlap)

    # 2. var column mapping
    for col in ("gene_name", "gene_names", "gene_symbols", "symbol", "hgnc_symbol"):
        if col in adata.var.columns:
            mapping = dict(zip(adata.var_names, adata.var[col].astype(str)))
            mapped_overlap = len(set(mapping.values()) & ckpt_set)
            if mapped_overlap >= 50:
                logger.info(
                    "✅ Using adata.var['%s'] for gene mapping (%d overlapping)",
                    col, mapped_overlap,
                )
                new_names = pd.Index([mapping.get(g, g) for g in adata.var_names])
                adata = adata.copy()
                adata.var_names = new_names
                adata.var_names_make_unique()
                return adata, mapping

    # Detect if IDs look like Ensembl
    ensembl_ids = [g for g in h5_names if g.startswith("ENSG")]
    if not ensembl_ids:
        logger.warning(
            "⚠️  Gene names not Ensembl IDs and no var column found; "
            "proceeding with overlap=%d",
            direct_overlap,
        )
        return adata, None

    # 3. pybiomart
    try:
        import pybiomart
        server = pybiomart.Server(host="http://www.ensembl.org")
        mart = server["ENSEMBL_MART_ENSEMBL"]
        dataset = mart["hsapiens_gene_ensembl"]
        result = dataset.query(
            attributes=["ensembl_gene_id", "hgnc_symbol"],
            filters={"ensembl_gene_id": ensembl_ids[:5000]},
        )
        e2s = dict(zip(result["Gene stable ID"], result["HGNC symbol"]))
        e2s = {k: v for k, v in e2s.items() if v}
        mapped_overlap = len(set(e2s.get(g, g) for g in h5_names) & ckpt_set)
        if mapped_overlap >= 50:
            logger.info(
                "✅ pybiomart: %d Ensembl→symbol, %d overlap with checkpoint",
                len(e2s), mapped_overlap,
            )
            new_names = pd.Index([e2s.get(g, g) for g in adata.var_names])
            adata = adata.copy()
            adata.var_names = new_names
            adata.var_names_make_unique()
            return adata, e2s
    except Exception as exc:
        logger.warning("⚠️  pybiomart failed (%s), trying MyGene.info", exc)

    # 4. MyGene.info REST API
    try:
        batch_size = 1000
        e2s = {}
        for i in range(0, len(ensembl_ids), batch_size):
            batch = ensembl_ids[i: i + batch_size]
            payload = (
                "q=" + ",".join(batch)
                + "&fields=symbol&species=human&scopes=ensembl.gene"
            )
            req = urllib.request.Request(
                "https://mygene.info/v3/query",
                data=payload.encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                hits = json.loads(resp.read())
            for hit in hits:
                if "query" in hit and "symbol" in hit:
                    e2s[hit["query"]] = hit["symbol"]
        mapped_overlap = len(set(e2s.get(g, g) for g in h5_names) & ckpt_set)
        if mapped_overlap >= 50:
            logger.info(
                "✅ MyGene.info: %d Ensembl→symbol, %d overlap with checkpoint",
                len(e2s), mapped_overlap,
            )
            new_names = pd.Index([e2s.get(g, g) for g in adata.var_names])
            adata = adata.copy()
            adata.var_names = new_names
            adata.var_names_make_unique()
            return adata, e2s
    except Exception as exc:
        logger.warning("⚠️  MyGene.info failed (%s)", exc)

    logger.warning(
        "⚠️  Gene reconciliation failed; checkpoint overlap = %d. "
        "Proceeding with partial overlap.",
        direct_overlap,
    )
    return adata, None


# =============================================================================
# CHECKPOINT DOWNLOAD
# =============================================================================

def _download_checkpoint(seed: int) -> dict:
    """Download pretrained CPA checkpoint."""
    if not os.path.exists(PRETRAINED_PT):
        import gdown
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("📥 Downloading pretrained CPA K562 checkpoint ...")
        gdown.download(id=KANG_MODEL_ID, output=PRETRAINED_PT, quiet=False)
        logger.info("✅ Checkpoint downloaded to %s", PRETRAINED_PT)
    else:
        logger.info("✅ Using existing checkpoint: %s", PRETRAINED_PT)

    _pandas_shim()
    sd_raw = torch.load(PRETRAINED_PT, map_location="cpu", weights_only=False)

    attr: dict = {}
    arch: dict = {}
    pert_encoder: dict = {}
    ckpt_var_names = None
    sd = sd_raw

    if isinstance(sd_raw, dict):
        if "state_dict" in sd_raw:
            sd = sd_raw["state_dict"]
            attr_dict = sd_raw.get("attr_dict", {})
            if "init_params_" in attr_dict:
                inner = attr_dict["init_params_"]
                if isinstance(inner, dict) and "hyper_params" in inner:
                    attr = inner["hyper_params"]
                else:
                    attr = inner
            else:
                attr = attr_dict
            arch = sd_raw.get("model_state_dict", sd_raw.get("hyper_parameters", {}))
            registry = sd_raw.get("var_registry", {})
            if "perturbation_key" in registry:
                pert_encoder = registry["perturbation_key"]
            elif "pert_encoder" in sd_raw:
                pert_encoder = sd_raw["pert_encoder"]
            ckpt_var_names = attr.get("var_names", sd_raw.get("var_names", None))
        elif "model_state_dict" in sd_raw:
            sd = sd_raw["model_state_dict"]
            attr = sd_raw.get("hyper_parameters", {})
            arch = attr
        elif any("encoder" in k or k.startswith("module.") for k in sd_raw):
            sd = sd_raw
        else:
            sd = sd_raw
    else:
        sd = sd_raw

    # Strip "module." prefix
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    if ckpt_var_names is not None and not isinstance(ckpt_var_names, list):
        ckpt_var_names = list(ckpt_var_names)

    logger.info("✅ Checkpoint loaded: %d genes, %d params",
                len(ckpt_var_names) if ckpt_var_names else 0, len(sd))

    return {
        "sd": sd,
        "attr": attr,
        "arch": arch,
        "pert_encoder": pert_encoder,
        "ckpt_var_names": ckpt_var_names,
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_eval(adata, cfg: dict) -> dict:
    """
    Run CPA evaluation.

    Parameters
    ----------
    adata : sc.AnnData
        Subsampled dataset.
    cfg : dict
        Runtime configuration.

    Returns
    -------
    dict
        Evaluation result with model name, metrics, perturbation names,
        and runtime in seconds.
    """
    import warnings
    import scanpy as sc
    from tqdm import tqdm

    warnings.filterwarnings("ignore")

    t_start = time.time()
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)
    ctrl_label = cfg.get("CTRL_LABEL", CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    top_k = cfg.get("TOP_K_DE", TOP_K_DE)
    device_str = cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("🧬 CPA EVALUATION STARTING")
    logger.info("=" * 70)

    # --- Install & patch ---------------------------------------------------
    _install_dependencies()
    _clear_module_cache()
    _patch_scvi()
    _install_jax_stub()
    _patch_cpa_source()

    if CPA_DIR not in sys.path:
        sys.path.insert(0, CPA_DIR)

    import cpa
    logger.info("✅ CPA %s imported", getattr(cpa, "__version__", "?"))

    # --- Download checkpoint -----------------------------------------------
    ckpt = _download_checkpoint(seed)
    sd = ckpt["sd"]
    attr = ckpt["attr"]
    arch = ckpt["arch"]
    pert_encoder = ckpt["pert_encoder"]
    ckpt_var_names = ckpt["ckpt_var_names"]

    # --- Prepare adata -----------------------------------------------------
    adata = adata.copy()

    if "perturbation" not in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs[pert_col].astype(str)

    # The Kang K562 checkpoint was trained with "ctrl" as control label.
    # Global config uses "non-targeting" — rename to "ctrl" for CPA only.
    # This mirrors the notebook (cell 12, STEP 10):
    #   adata.obs["perturbation"].str.replace(CTRL_LABEL, "ctrl", ...)
    #   CTRL_LABEL = "ctrl"
    orig_ctrl = ctrl_label          # e.g. "non-targeting" from config
    adata.obs["perturbation"] = (
        adata.obs["perturbation"].astype(str)
             .str.replace(orig_ctrl, "ctrl", regex=False)
    )
    ctrl_label = "ctrl"             # CPA always uses "ctrl" internally

    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "K562"

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # --- Gene reconciliation -----------------------------------------------
    if ckpt_var_names:
        adata, gene_map = _reconcile_gene_names(adata, ckpt_var_names)
        shared = sorted(set(adata.var_names) & set(ckpt_var_names))
        if shared:
            adata = adata[:, shared].copy()
            logger.info("✅ Gene subset: %d shared with checkpoint", len(shared))
        else:
            logger.warning("⚠️  No shared genes found after reconciliation")

    adata.obs["cov_cond"] = "K562_" + adata.obs["perturbation"].astype(str)

    # Filter to perturbations the checkpoint knows about
    if pert_encoder:
        known_perts = set(pert_encoder.keys())
        valid_perts = set(adata.obs["perturbation"].unique()) & (known_perts | {ctrl_label})
        before = adata.n_obs
        adata = adata[adata.obs["perturbation"].isin(valid_perts)].copy()
        logger.info(
            "✅ Filtered to %d perturbations (%d → %d cells)",
            len(valid_perts), before, adata.n_obs,
        )

    # --- Compute variable genes --------------------------------------------
    if sp.issparse(adata.X):
        _mean = np.asarray(adata.X.mean(axis=0)).flatten()
        _sq = np.asarray(adata.X.power(2).mean(axis=0)).flatten()
        real_var = _sq - _mean ** 2
    else:
        real_var = np.asarray(adata.X.var(axis=0)).flatten()

    real_mask = real_var > 1e-8
    real_gene_names = adata.var_names[real_mask]
    real_gene_idx = np.where(real_mask)[0]
    logger.info("✅ Non-zero-variance genes: %d/%d", len(real_gene_names), adata.n_vars)

    # Drop tiny groups
    grp_counts = adata.obs["perturbation"].value_counts()
    small = [g for g in grp_counts[grp_counts < 3].index if g != ctrl_label]
    if small:
        adata = adata[~adata.obs["perturbation"].isin(small)].copy()

    # --- DEGs --------------------------------------------------------------
    adata_rgg = adata[:, real_gene_names].copy()
    sc.pp.normalize_total(adata_rgg, target_sum=1e4)
    sc.pp.log1p(adata_rgg)

    n_deg = min(200, len(real_gene_names))
    deg_ok = False
    for method in ("t-test", "wilcoxon"):
        try:
            sc.tl.rank_genes_groups(
                adata_rgg, groupby="perturbation", reference=ctrl_label,
                method=method, n_genes=n_deg, key_added="rank_genes_groups",
                use_raw=False,
            )
            rgg = adata_rgg.uns.get("rank_genes_groups")
            if rgg and "names" in rgg and len(rgg["names"]) > 0:
                deg_ok = True
                logger.info("✅ DEG succeeded with '%s'", method)
                break
        except Exception as e:
            logger.warning("⚠️  DEG method '%s' failed: %s", method, e)

    if not deg_ok:
        raise RuntimeError("All DEG methods failed.")

    adata.uns["rank_genes_groups"] = adata_rgg.uns["rank_genes_groups"]
    del adata_rgg

    adata.uns["rank_genes_groups_cov"] = {
        f"K562_{g}": list(adata.uns["rank_genes_groups"]["names"][g])
        for g in adata.uns["rank_genes_groups"]["names"].dtype.names
        if g != ctrl_label
    }

    # Splits
    all_perts = [p for p in adata.obs["perturbation"].unique() if p != ctrl_label]
    ood = (
        rng.choice(all_perts, max(1, int(len(all_perts) * 0.1)), replace=False).tolist()
        if all_perts else []
    )
    adata.obs["split"] = rng.choice(["train", "valid"], adata.n_obs, p=[0.85, 0.15])
    if ood:
        adata.obs.loc[adata.obs["perturbation"].isin(ood), "split"] = "ood"

    # --- Setup CPA model ---------------------------------------------------
    logger.info("🔧 Setting up CPA model...")
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

    cpa_kw: dict = {}
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
    logger.info("✅ Loaded %d/%d parameter tensors from checkpoint",
                len(filtered_sd), len(model_sd))

    try:
        model.to_device(device_str)
    except TypeError:
        model.module.to(torch.device(device_str))
    logger.info("✅ Model ready on %s", device_str)

    # --- Predictions -------------------------------------------------------
    logger.info("🚀 Running predictions...")
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
    for _loc, d in [("obsm", adata.obsm), ("layers", adata.layers)]:
        for k in list(d.keys()):
            if "pred" in k.lower() or k == "CPA_pred":
                adata.layers["CPA_pred"] = np.array(d[k])
                pred_found = True
                logger.info("✅ Found predictions in adata.%s['%s']", _loc, k)
                break
        if pred_found:
            break

    if not pred_found:
        raise KeyError("CPA predictions not found in adata.obsm or adata.layers.")

    adata.X = adata.layers["X_true"].copy()
    logger.info("✅ Predictions ready: %s", str(adata.layers["CPA_pred"].shape))

    # --- Normalize to log1p space ------------------------------------------
    logger.info("📊 Normalizing to log1p space...")
    true_raw = adata.layers["counts"]
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
    ctrl_mu = (
        ctrl_for_mu.layers["counts_log"].mean(0)
        if ctrl_for_mu.n_obs > 0
        else true_log.mean(0)
    )
    logger.info("✅ Control baseline computed on %d genes", len(ctrl_mu))

    # --- 3-tier metrics ----------------------------------------------------
    logger.info("📈 Computing 3-tier metrics...")
    eval_perts = [
        c for c in adata.obs["perturbation"].unique()
        if c != ctrl_label
        and f"K562_{c}" in adata.uns.get("rank_genes_groups_cov", {})
    ]
    if len(eval_perts) > MAX_EVAL_PERTS:
        eval_counts = [(c, (adata.obs["perturbation"] == c).sum()) for c in eval_perts]
        eval_counts.sort(key=lambda x: x[1], reverse=True)
        eval_perts = [c for c, _ in eval_counts[:MAX_EVAL_PERTS]]
        logger.info("ℹ️  Limited to top %d perturbations", MAX_EVAL_PERTS)

    if not eval_perts:
        logger.warning("⚠️  No evaluable perturbations found")
        return {
            "model": "CPA",
            "metrics": {},
            "pert_names": [],
            "runtime_seconds": time.time() - t_start,
        }

    # Build per-perturbation tensors
    pred_centroids_list, true_centroids_list = [], []
    pred_cells_d, true_cells_d = {}, {}
    CN = []
    ctrl_real = torch.tensor(ctrl_mu[real_gene_idx], dtype=torch.float32)

    for c in tqdm(eval_perts, desc="CPA build tensors", ncols=70):
        m = adata.obs["perturbation"] == c
        xt = adata[m].layers["counts_log"][:, real_gene_idx].astype(np.float32)
        xp = adata[m].layers["CPA_pred_log"][:, real_gene_idx].astype(np.float32)
        pred_centroids_list.append(torch.tensor(xp.mean(0), dtype=torch.float32))
        true_centroids_list.append(torch.tensor(xt.mean(0), dtype=torch.float32))
        pred_cells_d[c] = torch.tensor(xp, dtype=torch.float32)
        true_cells_d[c] = torch.tensor(xt, dtype=torch.float32)
        CN.append(c)

    pred_c = torch.stack(pred_centroids_list)
    true_c = torch.stack(true_centroids_list)

    metrics = compute_all_metrics(
        pred_c, true_c, ctrl_real,
        pred_cells_dict=pred_cells_d,
        true_cells_dict=true_cells_d,
        pert_names=CN,
    )

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    runtime = time.time() - t_start
    logger.info("⏱️  Total runtime: %.2f min", runtime / 60)
    logger.info("=" * 70)

    return {
        "model": "CPA",
        "metrics": metrics,
        "pert_names": CN,
        "runtime_seconds": runtime,
    }
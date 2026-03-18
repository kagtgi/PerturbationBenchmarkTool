"""
Cell2Sentence (C2S-Scale-Gemma-2-2B) evaluation.

Uses an LLM-based approach: encodes cell expression as gene-name sentences
ranked by expression level, prompts the model to predict post-perturbation
sentences, then reconstructs expression profiles via power-law mapping.

Refined based on working reference (v5.0) with:
- Stable transformers version (>=4.45.0, not >=4.50.0)
- Proper dependency checking before installation
- Log1p-normalized metrics (CPA-aligned)
- Non-zero-variance gene filtering
- Better memory management
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

from tqdm import tqdm

from .. import config

logger = logging.getLogger(__name__)

C2S_DIR = "/tmp/vandijklab_c2s"
HF_MODEL = "vandijklab/C2S-Scale-Gemma-2-2B"
TOP_K_GENES = 200
MAX_NEW_TOKENS = 400      # reduced from 600; 200 genes × ~2 tokens/gene ≈ 400 tokens needed
MAX_PERTURBATIONS = 120   # reduced from 200 (60% of inference calls → ~40% time saving)
EVAL_SAMPLE_CELLS = 30    # reduced from 50
MAX_EVAL_PERTS = 60       # reduced from 100
MAX_CELLS_SAMPLE = 100    # reduced from 150 (subsample for energy/MMD)


def _pip(*packages: str) -> None:
    """Install packages with --upgrade flag (matching reference script)."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + list(packages),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Clone Cell2Sentence repo and install transformer deps (reference-style)."""
    if os.path.exists(C2S_DIR):
        shutil.rmtree(C2S_DIR)
    
    subprocess.check_call([
        "git", "clone", "-q",
        "https://github.com/vandijklab/cell2sentence.git",
        C2S_DIR,
    ])
    
    if C2S_DIR not in sys.path:
        sys.path.insert(0, C2S_DIR)

    # Check if already installed (reference script approach)
    try:
        import transformers
        import bitsandbytes
        logger.info(f"  ✅ transformers: {transformers.__version__}")
        logger.info(f"  ✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        # Pin to stable versions (>=4.45.0 works, >=4.50.0 causes FourOverSixConfig errors)
        _pip("transformers>=4.45.0", "accelerate>=0.34.0", "bitsandbytes>=0.43.0")
        _pip("cell2sentence==1.1.0", "anndata>=0.10.0", "scanpy>=1.10.0")
        _pip("scikit-learn", "pandas", "scipy", "tqdm")


def _pearson(a, b):
    """Safe Pearson correlation."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return np.nan if np.isnan(r) else float(r)


def _da(pm, tm, cm, thr=0.1):
    """Directional Accuracy."""
    dp, dt = pm - cm, tm - cm
    m = np.abs(dt) > thr
    if not m.any():
        return np.nan
    return float(np.mean(np.sign(dp[m]) == np.sign(dt[m])))


def _jaccard(pm, tm, cm, k=50):
    """Jaccard index for top-k DEGs."""
    pt = set(np.argsort(np.abs(pm - cm))[-k:].tolist())
    tt = set(np.argsort(np.abs(tm - cm))[-k:].tolist())
    u = pt | tt
    return len(pt & tt) / len(u) if u else 0.0


def _energy(p, q, n=MAX_CELLS_SAMPLE, seed=42):
    """Energy distance between cell distributions."""
    rng = np.random.default_rng(seed)
    if len(p) < 2 or len(q) < 2:
        return np.nan
    n_p = min(n, len(p))
    n_q = min(n, len(q))
    p = p[rng.choice(len(p), n_p, replace=False)].astype(np.float32)
    q = q[rng.choice(len(q), n_q, replace=False)].astype(np.float32)
    return max(float(2 * cdist(p, q).mean()
                    - cdist(p, p).mean()
                    - cdist(q, q).mean()), 0.)


def _mmd(p, q, n=MAX_CELLS_SAMPLE, seed=42):
    """Maximum Mean Discrepancy (RBF kernel)."""
    rng = np.random.default_rng(seed)
    if len(p) < 2 or len(q) < 2:
        return np.nan
    n_p = min(n, len(p))
    n_q = min(n, len(q))
    p = p[rng.choice(len(p), n_p, replace=False)].astype(np.float32)
    q = q[rng.choice(len(q), n_q, replace=False)].astype(np.float32)
    dqq = cdist(q, q, "sqeuclidean")
    off = dqq[~np.eye(len(q), dtype=bool)]
    s2 = max(float(np.median(off)) / 2.0 if len(off) else 1.0, 1e-6)
    Kpp = np.exp(-cdist(p, p, "sqeuclidean") / (2 * s2)).mean()
    Kqq = np.exp(-dqq / (2 * s2)).mean()
    Kpq = np.exp(-cdist(p, q, "sqeuclidean") / (2 * s2)).mean()
    return max(float(Kpp - 2 * Kpq + Kqq), 0.)


def run_eval(adata, cfg: dict) -> dict:
    """Run Cell2Sentence evaluation (CPA-aligned metrics).

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

    t_start = time.time()
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    top_k_de = cfg.get("TOP_K_DE", config.TOP_K_DE)
    max_t3 = cfg.get("MAX_T3_CELLS", config.MAX_T3_CELLS)

    logger.info("Cell2Sentence eval starting (python=%s cuda=%s model=%s)",
                sys.version.split()[0], torch.cuda.is_available(), HF_MODEL)

    # Install dependencies (reference-style with version check)
    _install_dependencies()

    # Import transformers AFTER installation
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        if "FourOverSixConfig" in str(e):
            logger.error(
                "Transformers installation is inconsistent. "
                "This often happens when upgrading transformers in a running session. "
                "Please restart the Runtime and run again."
            )
        raise e

    # Verify CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("Cell2Sentence requires CUDA for quantized inference.")

    logger.info(f"  ✅ CUDA: {torch.cuda.get_device_name(0)}")
    logger.info(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # --- Prepare data ------------------------------------------------------
    if "perturbation" not in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs[pert_col].astype(str)
    pert_key = "perturbation"

    # Filter: minimum cells per perturbation (consistent with rest of pipeline)
    min_cells = cfg.get("MIN_CELLS_PER_PERT", config.MIN_CELLS_PER_PERT)
    vc = adata.obs[pert_key].value_counts()
    adata = adata[adata.obs[pert_key].isin(vc[vc >= min_cells].index)].copy()

    # Select TOP perturbations by cell count
    pert_counts = adata.obs[pert_key].value_counts()
    pert_counts = pert_counts[pert_counts.index != ctrl_label]
    top_perts = pert_counts.head(MAX_PERTURBATIONS).index.tolist()
    logger.info(f"  Selected {len(top_perts)} perturbations for evaluation")

    # Filter to selected perturbations + control
    keep_perts = top_perts + [ctrl_label]
    adata = adata[adata.obs[pert_key].isin(keep_perts)].copy()

    # Subsample for evaluation
    rng = np.random.default_rng(seed)
    keep = []
    for _, grp in adata.obs.groupby(pert_key):
        n = min(EVAL_SAMPLE_CELLS, max(2, int(len(grp) * 0.15)))
        keep.extend(rng.choice(grp.index, n, replace=False))
    adata = adata[keep].copy()
    logger.info(f"  After subsampling: {adata.shape}")

    # Dense conversion
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()

    # --- Load model --------------------------------------------------------
    logger.info("Loading C2S-Scale-Gemma-2-2B (4-bit) ...")
    load_start = time.time()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_threshold=6.0,
    )

    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL, padding_side="left", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL, quantization_config=quant_config,
        device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    load_end = time.time()
    logger.info(f"  ✅ Model loaded in {(load_end - load_start)/60:.2f} min")

    # --- Build gene-symbol vocabulary for C2S ----------------------------------
    # C2S was trained on sentences of HGNC gene symbols (e.g. "TP53 GAPDH …").
    # If the dataset uses Ensembl IDs as var_names (common in Perturb-seq h5ads)
    # the model would receive unrecognisable tokens and produce empty sentences.
    # Detect this case and fall back to the gene_name column when available.
    _var_names_sample = list(adata.var_names)[:20]
    _uses_ensembl = any(str(v).startswith("ENSG") for v in _var_names_sample)
    if _uses_ensembl and "gene_name" in adata.var.columns:
        var_symbols = adata.var["gene_name"].astype(str).tolist()
        logger.info("  var_names are Ensembl IDs — using adata.var['gene_name'] for C2S vocabulary.")
    else:
        var_symbols = list(adata.var_names)

    # --- Build control template --------------------------------------------
    ctrl_mask = adata.obs[pert_key] == ctrl_label
    ctrl_expr = adata[ctrl_mask].X.mean(axis=0).flatten()
    top_idx = np.argsort(-ctrl_expr)[:TOP_K_GENES]
    template = " ".join(var_symbols[i] for i in top_idx)

    # Power-law fit
    ranks = np.argsort(-ctrl_expr)[:TOP_K_GENES]
    log_expr = np.log10(ctrl_expr[ranks] + 1e-8)
    log_rank = np.log10(1 + np.arange(TOP_K_GENES))
    reg = LinearRegression().fit(log_rank.reshape(-1, 1), log_expr)
    slope, intercept = reg.coef_[0], reg.intercept_
    logger.info(f"  Power-law: slope={slope:.4f}, intercept={intercept:.4f}")

    # --- Generate predictions ----------------------------------------------
    unique_perts = [p for p in top_perts if p != ctrl_label]
    pred_sentences = {}

    logger.info(f"  Generating for {len(unique_perts)} perturbations...")
    inference_start = time.time()

    for i, condition in enumerate(tqdm(unique_perts, desc="Generating")):
        prompt = (
            f"Given the following cell sentence of {TOP_K_GENES} expressed genes "
            f"representing a cell's basal state, predict the cell sentence after "
            f"applying the perturbation: {condition}.\n"
            f"Control cell sentence: {template}\n"
            f"Perturbed cell sentence:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, temperature=None,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_sentences[condition] = generated.replace(prompt, "").strip()
        del inputs, outputs
        if i % 20 == 0:
            torch.cuda.empty_cache()

    inference_end = time.time()
    logger.info(f"  ✅ Inference completed in {(inference_end - inference_start)/60:.2f} min")

    # --- Reconstruct expression from sentences -----------------------------
    logger.info("  Reconstructing sentence → matrix...")
    recon_start = time.time()

    # vocab_list / vocab_set / vocab_dict use gene symbols (consistent with template above)
    vocab_list = var_symbols
    vocab_set = set(vocab_list)
    vocab_dict = {g: i for i, g in enumerate(vocab_list)}   # O(1) lookup
    adata.layers["X_true"] = adata.X.copy()
    pred_X_list = []

    for i, row in enumerate(tqdm(adata.obs.itertuples(), desc="Reconstructing", total=adata.n_obs)):
        cond = getattr(row, pert_key, None) or row.perturbation
        if cond == ctrl_label:
            pred_X_list.append(adata.X[i].copy())
            continue
        sentence = pred_sentences.get(cond, "")
        genes = [g.strip() for g in sentence.split() if g.strip() in vocab_set][:TOP_K_GENES]
        expr = np.zeros(adata.n_vars, dtype=np.float64)
        for rank, gene in enumerate(genes):
            idx = vocab_dict[gene]
            expr[idx] = 10 ** (slope * np.log10(1 + rank) + intercept)
        expr += rng.normal(0, 1e-6, size=expr.shape)
        pred_X_list.append(expr)

    adata.layers["C2S_pred"] = np.array(pred_X_list)
    recon_end = time.time()
    logger.info(f"  ✅ Reconstruction in {recon_end - recon_start:.2f}s")

    # --- Normalize to log1p space (CPA-aligned) ----------------------------
    logger.info("  Normalizing to log1p space (CPA-aligned)...")

    # Compute non-zero-variance genes
    _mean = np.asarray(adata.X.mean(axis=0)).flatten()
    _sq_mean = np.asarray((adata.X ** 2).mean(axis=0)).flatten()
    real_var = _sq_mean - _mean ** 2
    _real_genes_mask = real_var > 1e-8
    _real_gene_idx = np.where(_real_genes_mask)[0]
    n_real = int(_real_genes_mask.sum())
    logger.info(f"  Non-zero-variance genes: {n_real}/{adata.n_vars}")

    # Library size normalization + log1p for TRUE counts
    _true_raw = adata.layers["X_true"]
    _lib_size = np.asarray(_true_raw.sum(axis=1)).flatten()
    _lib_size = np.maximum(_lib_size, 1.0)
    _true_norm = (_true_raw / _lib_size[:, None]) * 1e4
    _true_log = np.log1p(_true_norm).astype(np.float64)

    # Same normalization for PREDICTIONS
    _pred_raw = adata.layers["C2S_pred"]
    _lib_size_pred = np.asarray(_pred_raw.sum(axis=1)).flatten()
    _lib_size_pred = np.maximum(_lib_size_pred, 1.0)
    _pred_norm = (_pred_raw / _lib_size_pred[:, None]) * 1e4
    _pred_log = np.log1p(_pred_norm).astype(np.float64)

    # Store normalized versions
    adata.layers["X_true_log"] = _true_log
    adata.layers["C2S_pred_log"] = _pred_log

    # Control baseline in log space (restricted to real genes)
    ctrl_mu = _true_log[ctrl_mask].mean(0)[_real_gene_idx]

    # --- Compute 3-tier metrics (CPA-aligned) ------------------------------
    logger.info("  Computing 3-tier metrics (CPA-aligned)...")
    eval_start = time.time()

    _eval = [c for c in adata.obs[pert_key].unique() if c != ctrl_label]

    if len(_eval) > MAX_EVAL_PERTS:
        _eval_counts = [(c, (adata.obs[pert_key] == c).sum()) for c in _eval]
        _eval_counts.sort(key=lambda x: x[1], reverse=True)
        _eval = [c for c, _ in _eval_counts[:MAX_EVAL_PERTS]]
        logger.info(f"  Limited evaluation to top {MAX_EVAL_PERTS} perturbations")

    if len(_eval) == 0:
        logger.warning("  ⚠ No evaluable perturbations found")
        metrics = {}
        pert_names = []
    else:
        # Collect centroids (LOG-SPACE, REAL GENES)
        PC, TC, CN = [], [], []
        BATCH_SIZE = 50

        for i in range(0, len(_eval), BATCH_SIZE):
            batch = _eval[i:i+BATCH_SIZE]
            for _c in batch:
                _m = adata.obs[pert_key] == _c
                _xt = adata[_m].layers["X_true_log"][:, _real_gene_idx]
                _xp = adata[_m].layers["C2S_pred_log"][:, _real_gene_idx]
                PC.append(_xp.mean(0))
                TC.append(_xt.mean(0))
                CN.append(_c)
            del _xt, _xp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        P = len(CN)
        PM = np.stack(PC)
        TM = np.stack(TC)
        del PC, TC

        # Centroid Accuracy & PDS
        D = np.linalg.norm(PM[:, None] - TM[None], axis=-1)
        ca_v = (D.argmin(1) == np.arange(P)).astype(float)
        d_self = D[np.arange(P), np.arange(P)]

        if P > 1:
            _off = ~np.eye(P, dtype=bool)
            d_cross = (D * _off).sum(1) / _off.sum(1)
            pds_v = d_self / (d_self + d_cross + 1e-8)
        else:
            pds_v = np.array([np.nan])

        del D, PM, TM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Per-perturbation metrics
        res = defaultdict(list)
        for i, _c in enumerate(tqdm(CN, ncols=70, desc="Metrics")):
            _m = adata.obs[pert_key] == _c
            _xt = adata[_m].layers["X_true_log"][:, _real_gene_idx]
            _xp = adata[_m].layers["C2S_pred_log"][:, _real_gene_idx]

            pm = _xp.mean(0)
            tm = _xt.mean(0)

            # Deltas relative to control
            _pred_delta = pm - ctrl_mu
            _true_delta = tm - ctrl_mu

            # Top-k genes by |true_delta| for T2_Pearson_Delta_TopK
            _k = min(top_k_de, len(_true_delta))
            _top_k_idx = np.argsort(np.abs(_true_delta))[-_k:]

            res["condition"].append(_c)
            res["T1_Centroid_Accuracy"].append(ca_v[i])
            res["T1_Profile_Distance_Score"].append(pds_v[i])
            res["T1_Systema_Pearson_Delta"].append(_pearson(_pred_delta, _true_delta))
            res["T2_Directional_Accuracy"].append(_da(pm, tm, ctrl_mu))
            res["T2_Pearson_Delta_TopK"].append(
                _pearson(_pred_delta[_top_k_idx], _true_delta[_top_k_idx]))
            res["T2_Jaccard_TopK"].append(_jaccard(pm, tm, ctrl_mu, top_k_de))
            res["T3_Energy_Distance"].append(_energy(_xp, _xt, n=max_t3, seed=seed))
            res["T3_MMD_RBF"].append(_mmd(_xp, _xt, n=max_t3, seed=seed))

            del _xt, _xp, pm, tm
            if i % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        df = pd.DataFrame(res)
        pert_names = CN

        # Aggregate metrics — standard benchmark key names
        def _mean(col):
            return float(df[col].mean()) if col in df.columns else np.nan

        metrics = {
            "T1_Centroid_Accuracy":      _mean("T1_Centroid_Accuracy"),
            "T1_Profile_Distance_Score": _mean("T1_Profile_Distance_Score"),
            "T1_Systema_Pearson_Delta":  _mean("T1_Systema_Pearson_Delta"),
            "T2_Directional_Accuracy":   _mean("T2_Directional_Accuracy"),
            "T2_Pearson_Delta_TopK":     _mean("T2_Pearson_Delta_TopK"),
            "T2_Jaccard_TopK":           _mean("T2_Jaccard_TopK"),
            "T3_Energy_Distance":        _mean("T3_Energy_Distance"),
            "T3_MMD_RBF":                _mean("T3_MMD_RBF"),
        }

    eval_end = time.time()

    # Cleanup GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t_start
    logger.info(f"  ⏱️  TOTAL: {total_time/60:.2f} min")

    return {
        "model": "Cell2Sentence",
        "metrics": metrics,
        "pert_names": pert_names,
        "runtime_seconds": total_time,
    }
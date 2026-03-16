"""
Cell2Sentence (C2S-Scale-Gemma-2-2B) evaluation.

Uses an LLM-based approach: encodes cell expression as gene-name sentences
ranked by expression level, prompts the model to predict post-perturbation
sentences, then reconstructs expression profiles via power-law mapping.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .. import config
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)

C2S_DIR = "/tmp/vandijklab_c2s"
HF_MODEL = "vandijklab/C2S-Scale-Gemma-2-2B"
TOP_K_GENES = 200
MAX_NEW_TOKENS = 600
MAX_PERTURBATIONS = 200
EVAL_SAMPLE_CELLS = 50


def _pip(*packages: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + list(packages),
        stdout=subprocess.DEVNULL,
    )


def _install_dependencies() -> None:
    """Clone Cell2Sentence repo and install transformer deps."""
    if os.path.exists(C2S_DIR):
        shutil.rmtree(C2S_DIR)
    subprocess.check_call([
        "git", "clone", "-q",
        "https://github.com/vandijklab/cell2sentence.git", C2S_DIR,
    ])
    if C2S_DIR not in sys.path:
        sys.path.insert(0, C2S_DIR)

    # Always upgrade — checking importability is not enough.  FourOverSixConfig
    # was added in transformers 4.50; older versions import fine but fail later
    # inside AutoModelForCausalLM.from_pretrained when loading Gemma-2.
    _pip("transformers>=4.50.0", "accelerate>=0.34.0", "bitsandbytes>=0.43.0")
    _pip("cell2sentence==1.1.0", "anndata>=0.10.0", "scanpy>=1.10.0")
    _pip("scikit-learn", "pandas", "scipy", "tqdm")


def run_eval(adata, cfg: dict) -> dict:
    """Run Cell2Sentence evaluation.

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
    from sklearn.linear_model import LinearRegression
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t_start = time.time()
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    top_k = cfg.get("TOP_K_DE", config.TOP_K_DE)

    _install_dependencies()

    if not torch.cuda.is_available():
        raise RuntimeError("Cell2Sentence requires CUDA for quantized inference.")

    # --- Prepare data ------------------------------------------------------
    # Ensure perturbation column
    if "perturbation" not in adata.obs.columns:
        adata.obs["perturbation"] = adata.obs[pert_col].astype(str)
    pert_key = "perturbation"

    # Select top perturbations by cell count
    pert_counts = adata.obs[pert_key].value_counts()
    pert_counts = pert_counts[pert_counts.index != ctrl_label]
    top_perts = pert_counts.head(MAX_PERTURBATIONS).index.tolist()
    keep_perts = top_perts + [ctrl_label]
    adata = adata[adata.obs[pert_key].isin(keep_perts)].copy()

    # Dense conversion
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()

    # DEGs
    adata_log = adata.copy()
    # Apply log1p directly to the matrix rather than via sc.pp.log1p to avoid
    # singledispatch failures caused by sys.path manipulation in
    # _install_dependencies (inserting C2S_DIR at front can shadow anndata,
    # so AnnData is no longer the registered type in scanpy's dispatch table).
    adata_log.X = np.log1p(adata_log.X)
    sc.tl.rank_genes_groups(
        adata_log, groupby=pert_key, reference=ctrl_label,
        method="t-test", n_genes=adata.n_vars,
    )
    rgg = adata_log.uns["rank_genes_groups"]
    adata.uns["rank_genes_groups_cov"] = {
        f"K562_{g}": list(rgg["names"][g])
        for g in rgg["names"].dtype.names if g != ctrl_label
    }

    # --- Load model --------------------------------------------------------
    logger.info("Loading C2S-Scale-Gemma-2-2B (4-bit) ...")
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

    # --- Build control template --------------------------------------------
    ctrl_mask = adata.obs[pert_key] == ctrl_label
    ctrl_expr = adata[ctrl_mask].X.mean(axis=0).flatten()
    top_idx = np.argsort(-ctrl_expr)[:TOP_K_GENES]
    template = " ".join(adata.var_names[i] for i in top_idx)

    # Power-law fit
    ranks = np.argsort(-ctrl_expr)[:TOP_K_GENES]
    log_expr = np.log10(ctrl_expr[ranks] + 1e-8)
    log_rank = np.log10(1 + np.arange(TOP_K_GENES))
    reg = LinearRegression().fit(log_rank.reshape(-1, 1), log_expr)
    slope, intercept = reg.coef_[0], reg.intercept_

    # --- Generate predictions ----------------------------------------------
    rng = np.random.default_rng(seed)
    unique_perts = [p for p in top_perts if p != ctrl_label]
    pred_sentences = {}

    for i, condition in enumerate(tqdm(unique_perts, desc="C2S generating")):
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

    # --- Reconstruct expression from sentences -----------------------------
    vocab_list = list(adata.var_names)
    vocab_set = set(vocab_list)
    adata.layers["X_true"] = adata.X.copy()
    # Pre-allocate the prediction matrix to avoid building a Python list of
    # n_cells arrays and then calling np.array() on it (which doubles peak RAM).
    pred_X = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)

    for i, row in enumerate(adata.obs.itertuples()):
        cond = row.perturbation
        if cond == ctrl_label:
            pred_X[i] = adata.X[i]
            continue
        sentence = pred_sentences.get(cond, "")
        genes = [g.strip() for g in sentence.split() if g.strip() in vocab_set][:TOP_K_GENES]
        expr = np.zeros(adata.n_vars, dtype=np.float32)
        for rank, gene in enumerate(genes):
            idx = vocab_list.index(gene)
            expr[idx] = 10 ** (slope * np.log10(1 + rank) + intercept)
        expr += rng.normal(0, 1e-6, size=expr.shape).astype(np.float32)
        pred_X[i] = expr

    adata.layers["C2S_pred"] = pred_X

    # --- Metrics -----------------------------------------------------------
    ctrl_mu = adata[ctrl_mask].X.mean(axis=0).flatten()
    pred_centroids_list, true_centroids_list, pert_names = [], [], []
    pred_cells_d, true_cells_d = {}, {}

    for pert in unique_perts:
        idx = adata.obs[pert_key] == pert
        X_true = adata.X[idx]
        X_pred = adata.layers["C2S_pred"][idx]
        if X_true.shape[0] < 2:
            continue

        pred_centroids_list.append(torch.tensor(X_pred.mean(0), dtype=torch.float32))
        true_centroids_list.append(torch.tensor(X_true.mean(0), dtype=torch.float32))
        pred_cells_d[pert] = torch.tensor(X_pred, dtype=torch.float32)
        true_cells_d[pert] = torch.tensor(X_true, dtype=torch.float32)
        pert_names.append(pert)

    if not pert_names:
        raise RuntimeError("No perturbations evaluated for Cell2Sentence.")

    pred_c = torch.stack(pred_centroids_list)
    true_c = torch.stack(true_centroids_list)
    ctrl_t = torch.tensor(ctrl_mu, dtype=torch.float32)

    metrics = compute_all_metrics(
        pred_c, true_c, ctrl_t,
        pred_cells_dict=pred_cells_d,
        true_cells_dict=true_cells_d,
        pert_names=pert_names,
    )

    # Cleanup GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": "Cell2Sentence",
        "metrics": metrics,
        "pert_names": pert_names,
        "runtime_seconds": time.time() - t_start,
    }

"""
scGPT evaluation — zero-shot perturbation prediction using scGPT_human.

For each knockout gene, feeds control cells through the pre-trained
TransformerGenerator with ``pert_flag=1`` at the target gene position,
then compares predicted post-KO expression to observed perturbed cells.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import types as _types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .. import config
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)

SCGPT_DIR = "./scGPT_human"
GDRIVE_FOLDER_ID = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
LOAD_PREFIXES = ["encoder", "value_encoder", "transformer_encoder"]
MAX_EVAL_GENES = 1500
N_CTRL_CELLS = 200
INFER_BATCH = 32


# ═══════════════════════════════════════════════════════════════════════════
# torchtext stub (scGPT depends on torchtext but its .so is ABI-locked)
# ═══════════════════════════════════════════════════════════════════════════

class _VocabStub:
    """Minimal torchtext.vocab.Vocab substitute for scGPT internals."""
    def __init__(self, stoi=None):
        self._stoi = stoi or {}
        self._itos = {v: k for k, v in self._stoi.items()}
        self._default_index = 0
    def __contains__(self, t):      return t in self._stoi
    def __getitem__(self, t):       return self._stoi.get(t, self._default_index)
    def __len__(self):              return len(self._stoi)
    def set_default_index(self, i): self._default_index = i
    def append_token(self, t):
        if t not in self._stoi:
            i = len(self._stoi)
            self._stoi[t] = i
            self._itos[i] = t
    def get_stoi(self): return self._stoi
    def get_itos(self): return [self._itos.get(i, "<unk>") for i in range(len(self._itos))]
    @staticmethod
    def build_vocab_from_iterator(it, **kw):
        s = {}
        for toks in it:
            for t in toks:
                if t not in s:
                    s[t] = len(s)
        return _VocabStub(s)


def _inject_torchtext_stub() -> None:
    """Inject a lightweight torchtext stub into sys.modules."""
    for k in list(sys.modules.keys()):
        if "torchtext" in k:
            del sys.modules[k]

    def _stub(name):
        return _types.ModuleType(name)

    tt_voc = _stub("torchtext.vocab")
    tt_voc.Vocab = _VocabStub
    tt_voc.build_vocab_from_iterator = _VocabStub.build_vocab_from_iterator
    tt_txt = _stub("torchtext._torchtext")
    tt_txt.Vocab = _VocabStub

    for name, mod in [
        ("torchtext",            _stub("torchtext")),
        ("torchtext.vocab",      tt_voc),
        ("torchtext._torchtext", tt_txt),
        ("torchtext.data",       _stub("torchtext.data")),
        ("torchtext.utils",      _stub("torchtext.utils")),
        ("torchtext.datasets",   _stub("torchtext.datasets")),
        ("torchtext.functional", _stub("torchtext.functional")),
    ]:
        sys.modules[name] = mod


# ═══════════════════════════════════════════════════════════════════════════
# Standalone GeneVocab (no torchtext dependency at runtime)
# ═══════════════════════════════════════════════════════════════════════════

class GeneVocab:
    """Reads vocab.json directly; zero torchtext dependency."""

    def __init__(self, stoi: dict):
        self._stoi = dict(stoi)
        self._itos = {v: k for k, v in stoi.items()}
        self._default_index = 0

    @classmethod
    def from_file(cls, path: str) -> GeneVocab:
        with open(path) as f:
            data = json.load(f)
        stoi = data if isinstance(data, dict) else {t: i for i, t in enumerate(data)}
        return cls(stoi)

    def __contains__(self, t): return t in self._stoi
    def __getitem__(self, t):  return self._stoi.get(t, self._default_index)
    def __len__(self):         return len(self._stoi)
    def set_default_index(self, i): self._default_index = i
    def append_token(self, t):
        if t not in self._stoi:
            i = len(self._stoi)
            self._stoi[t] = i
            self._itos[i] = t
    def get_stoi(self): return self._stoi
    def get_itos(self): return [self._itos.get(i, "<unk>") for i in range(len(self._itos))]
    def lookup_indices(self, toks): return [self[t] for t in toks]
    def lookup_tokens(self, idxs):  return [self._itos.get(i, "<unk>") for i in idxs]


# ═══════════════════════════════════════════════════════════════════════════
# Install & download
# ═══════════════════════════════════════════════════════════════════════════

def _pip(*pkgs: str, quiet: bool = True) -> None:
    args = [sys.executable, "-m", "pip", "install", "--break-system-packages"]
    if quiet:
        args += ["-q"]
    subprocess.check_call(args + list(pkgs))


def _pip_uninstall(*pkgs: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *pkgs],
        capture_output=True,
    )


def _install_dependencies() -> None:
    """Install scGPT and remove torchtext."""
    _pip("git+https://github.com/bowang-lab/scGPT.git", quiet=False)
    _pip("huggingface_hub", "scanpy", "anndata")
    _pip_uninstall("torchtext")
    _inject_torchtext_stub()


def _download_checkpoint() -> str:
    """Download scGPT_human checkpoint from Google Drive if needed.

    Returns the resolved directory containing best_model.pt.
    """
    scgpt_dir = SCGPT_DIR
    if not Path(scgpt_dir).exists():
        _pip("gdown")
        import gdown
        logger.info("Downloading scGPT_human from Google Drive (~500 MB) ...")
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID, output=scgpt_dir,
            quiet=False, use_cookies=False,
        )

    model_file = Path(scgpt_dir) / "best_model.pt"
    if not model_file.exists():
        hits = list(Path(scgpt_dir).rglob("best_model.pt"))
        if hits:
            scgpt_dir = str(hits[0].parent)
        else:
            raise FileNotFoundError(
                f"best_model.pt not found under {scgpt_dir}."
            )
    return scgpt_dir


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(adata, cfg: dict) -> dict:
    """Run scGPT zero-shot evaluation.

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
    import scanpy as sc
    from tqdm import tqdm

    t_start = time.time()
    device_str = cfg.get("DEVICE", config.DEVICE)
    seed = cfg.get("RANDOM_SEED", config.RANDOM_SEED)
    ctrl_label = cfg.get("CTRL_LABEL", config.CTRL_LABEL)
    pert_col = cfg.get("PERT_COL", config.PERT_COL)
    min_cells = cfg.get("MIN_CELLS_PER_PERT", config.MIN_CELLS_PER_PERT)
    device = torch.device(device_str)

    # --- Install & download ------------------------------------------------
    _install_dependencies()
    scgpt_dir = _download_checkpoint()
    model_file = str(Path(scgpt_dir) / "best_model.pt")
    vocab_file = str(Path(scgpt_dir) / "vocab.json")

    from scgpt.model import TransformerGenerator
    from scgpt.utils import set_seed
    set_seed(seed)

    # AMP compatibility
    if hasattr(torch.amp, "GradScaler"):
        def _autocast():
            return torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))
    else:
        def _autocast():
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda"))

    # flash-attn detection
    try:
        import flash_attn  # noqa: F401
        use_fast = True
    except ImportError:
        use_fast = False

    # --- Preprocess h5ad ---------------------------------------------------
    rng = np.random.default_rng(seed)

    # Auto-detect raw counts and normalize
    X_check = adata.X[:1000]
    if sp.issparse(X_check):
        X_check = X_check.toarray()
    looks_raw = bool(
        np.nanmax(X_check) > 50
        or np.array_equal(X_check, X_check.astype(int))
    )
    if looks_raw:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    X_dense = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)).astype(np.float32)
    ctrl_mask_obs = (adata.obs[pert_col] == ctrl_label).values

    # --- Vocab & eval gene set ---------------------------------------------
    PAD_TOKEN = "<pad>"
    vocab = GeneVocab.from_file(vocab_file)
    for s in [PAD_TOKEN, "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])

    # Resolve gene symbols
    _SYMBOL_COL_CANDIDATES = [
        "gene_name", "gene_names", "gene_symbols", "symbol", "Gene",
        "feature_name", "gene_short_name", "name", "var_gene_name",
    ]
    gene_symbol_col = None
    for col in _SYMBOL_COL_CANDIDATES:
        if col in adata.var.columns:
            test = adata.var[col].astype(str).tolist()
            hits = sum(1 for g in test[:200] if g in vocab)
            if hits > 10:
                gene_symbol_col = col
                break

    if gene_symbol_col is not None:
        all_gene_names = adata.var[gene_symbol_col].astype(str).tolist()
    else:
        all_gene_names = list(adata.var_names)

    in_vocab = np.array([g in vocab for g in all_gene_names])

    # Select top genes by ctrl-cell variance from intersection
    ctrl_X = X_dense[ctrl_mask_obs]
    inter_idx = np.where(in_vocab)[0]
    ctrl_var = ctrl_X[:, inter_idx].var(axis=0)
    top_idx = np.argsort(ctrl_var)[::-1][:min(MAX_EVAL_GENES, len(inter_idx))]
    eval_h5ad_idx = np.sort(inter_idx[top_idx])
    eval_gene_names = [all_gene_names[i] for i in eval_h5ad_idx]
    eval_vocab_ids = np.array([vocab[g] for g in eval_gene_names], dtype=int)
    n_eval = len(eval_gene_names)
    eval_gene_pos = {g: i for i, g in enumerate(eval_gene_names)}

    if n_eval == 0:
        raise RuntimeError(
            "Zero genes in h5ad / vocab intersection. "
            "Check that h5ad var_names are gene symbols."
        )

    # Obs-level gene name mapping
    if gene_symbol_col is not None:
        ensg2sym = dict(zip(list(adata.var_names), all_gene_names))
        sample_obs = [g for g in adata.obs[pert_col].unique()[:20]
                      if g != ctrl_label and "+" not in g]
        direct = sum(1 for g in sample_obs if g in eval_gene_pos)
        via_map = sum(1 for g in sample_obs if ensg2sym.get(g, g) in eval_gene_pos)
        obs_to_eval = (lambda g: g) if direct >= via_map else (lambda g: ensg2sym.get(g, g))
    else:
        obs_to_eval = lambda g: g

    ctrl_mean_eval = ctrl_X[:, eval_h5ad_idx].mean(axis=0).astype(np.float32)

    # --- Build model -------------------------------------------------------
    cfg_path = Path(scgpt_dir) / "args.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            mcfg = json.load(f)
        embsize = mcfg.get("embsize", 512)
        nhead = mcfg.get("nheads", 8)
        d_hid = mcfg.get("d_hid", 512)
        nlayers = mcfg.get("nlayers", 12)
        n_cls_lyr = mcfg.get("n_layers_cls", 3)
    else:
        embsize, nhead, d_hid, nlayers, n_cls_lyr = 512, 8, 512, 12, 3

    model = TransformerGenerator(
        len(vocab), embsize, nhead, d_hid, nlayers,
        nlayers_cls=n_cls_lyr, n_cls=1, vocab=vocab,
        dropout=0.0, pad_token=PAD_TOKEN, pad_value=0, pert_pad_id=2,
        do_mvc=False, cell_emb_style="cls",
        mvc_decoder_style="inner product",
        use_fast_transformer=use_fast,
    )

    md = model.state_dict()
    pd_ = torch.load(model_file, map_location="cpu", weights_only=False)
    pd_ = {k: v for k, v in pd_.items()
           if any(k.startswith(p) for p in LOAD_PREFIXES) and k in md}
    md.update(pd_)
    model.load_state_dict(md)
    model.eval().to(device)

    # Pre-built tensors
    gene_ids_t = torch.tensor(eval_vocab_ids, dtype=torch.long).unsqueeze(0)
    pad_mask_t = torch.zeros(1, n_eval, dtype=torch.bool)

    n_ctrl_total = int(ctrl_mask_obs.sum())
    n_ctrl_use = min(N_CTRL_CELLS, n_ctrl_total)
    ctrl_sample_idx = rng.choice(n_ctrl_total, size=n_ctrl_use, replace=False)
    ctrl_eval_full = ctrl_X[:, eval_h5ad_idx]
    ctrl_expr_used = ctrl_eval_full[ctrl_sample_idx]

    # --- Inference ---------------------------------------------------------
    @torch.no_grad()
    def _run_pert(pert_pos: int) -> np.ndarray:
        chunks = []
        for start in range(0, n_ctrl_use, INFER_BATCH):
            end = min(start + INFER_BATCH, n_ctrl_use)
            B = end - start
            expr = torch.tensor(ctrl_expr_used[start:end], dtype=torch.float32)
            pflg = torch.zeros(B, n_eval, dtype=torch.long)
            pflg[:, pert_pos] = 1
            with _autocast():
                out = model(
                    gene_ids_t.expand(B, -1).to(device),
                    expr.to(device),
                    pflg.to(device),
                    src_key_padding_mask=pad_mask_t.expand(B, -1).to(device),
                    CLS=False, CCE=False, MVC=False, ECS=False,
                )
            chunks.append(out["mlm_output"].float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    unique_perts = [
        g for g in adata.obs[pert_col].unique().tolist()
        if g != ctrl_label and "+" not in g
    ]

    pred_centroids, true_centroids, pert_names = [], [], []
    energy_list, mmd_list = [], []
    max_t3 = cfg.get("MAX_T3_CELLS", config.MAX_T3_CELLS)

    def _t(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    for gene in tqdm(unique_perts, desc="scGPT inference", ncols=72):
        pert_mask = (adata.obs[pert_col] == gene).values
        if pert_mask.sum() < min_cells:
            continue
        pert_pos = eval_gene_pos.get(obs_to_eval(gene), -1)
        if pert_pos < 0:
            continue

        pred_cells = _run_pert(pert_pos)
        true_cells = X_dense[pert_mask][:, eval_h5ad_idx]

        pred_centroids.append(pred_cells.mean(axis=0))
        true_centroids.append(true_cells.mean(axis=0))
        pert_names.append(gene)

        # T3: energy + MMD inline
        p_arr = pred_cells.copy()
        t_arr = true_cells
        if p_arr.shape[0] >= 2 and t_arr.shape[0] >= 2:
            if p_arr.shape[0] > max_t3:
                p_arr = p_arr[rng.choice(p_arr.shape[0], max_t3, replace=False)]
            if t_arr.shape[0] > max_t3:
                t_arr = t_arr[rng.choice(t_arr.shape[0], max_t3, replace=False)]
            pt, tt = _t(p_arr), _t(t_arr)
            dpp = torch.cdist(pt, pt)
            dtt = torch.cdist(tt, tt)
            dpt = torch.cdist(pt, tt)
            mask_pp = ~torch.eye(pt.size(0), dtype=torch.bool, device=device)
            mask_tt = ~torch.eye(tt.size(0), dtype=torch.bool, device=device)
            energy_list.append(
                max((2 * dpt.mean() - dpp[mask_pp].mean() - dtt[mask_tt].mean()).item(), 0.0)
            )
            sigma2 = dtt.pow(2)[mask_tt].median().clamp(1e-6).item() / 2.0
            Kpp = torch.exp(-dpp.pow(2) / (2 * sigma2)).mean()
            Ktt = torch.exp(-dtt.pow(2) / (2 * sigma2)).mean()
            Kpt = torch.exp(-dpt.pow(2) / (2 * sigma2)).mean()
            mmd_list.append(max((Kpp - 2 * Kpt + Ktt).item(), 0.0))
            del pt, tt, dpp, dtt, dpt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            energy_list.append(0.0)
            mmd_list.append(0.0)

    if not pert_names:
        raise RuntimeError("Zero perturbations evaluated for scGPT.")

    pred_mat = np.stack(pred_centroids).astype(np.float32)
    true_mat = np.stack(true_centroids).astype(np.float32)

    pred_c = _t(pred_mat)
    true_c = _t(true_mat)
    ctrl_t = _t(ctrl_mean_eval)

    # T1 + T2 via unified metrics (no T3 — we computed it inline)
    from ..metrics import (
        centroid_accuracy_and_pds, systema_pearson_delta,
        directional_accuracy, pearson_delta_topk, jaccard_topk,
    )

    pred_delta = pred_c - ctrl_t.unsqueeze(0)
    true_delta = true_c - ctrl_t.unsqueeze(0)

    ca, pds = centroid_accuracy_and_pds(pred_c, true_c)
    spd = systema_pearson_delta(pred_c, true_c)
    da = directional_accuracy(pred_delta, true_delta)
    pde = pearson_delta_topk(pred_delta, true_delta)
    jac = jaccard_topk(pred_delta, true_delta)

    metrics = {
        "T1_Centroid_Accuracy": ca,
        "T1_Profile_Distance_Score": pds,
        "T1_Systema_Pearson_Delta": spd,
        "T2_Directional_Accuracy": da,
        "T2_Pearson_Delta_TopK": pde,
        "T2_Jaccard_TopK": jac,
        "T3_Energy_Distance": float(np.mean(energy_list)),
        "T3_MMD_RBF": float(np.mean(mmd_list)),
    }

    return {
        "model": "scGPT",
        "metrics": metrics,
        "pert_names": pert_names,
        "n_eval_genes": n_eval,
        "runtime_seconds": time.time() - t_start,
    }

"""
Unified evaluation metrics for perturbation prediction benchmarking.

Three tiers of metrics:
  Tier 1 — Whole-profile identity (centroid-level)
  Tier 2 — Differentially-expressed gene accuracy
  Tier 3 — Distributional fidelity (single-cell level)

All functions accept torch tensors on any device and return plain floats.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Whole-Profile Identity
# ═══════════════════════════════════════════════════════════════════════════

def centroid_accuracy_and_pds(
    pred_centroids: torch.Tensor,
    true_centroids: torch.Tensor,
) -> tuple[float, float]:
    """Centroid Accuracy (CA) and Profile Distance Score (PDS).

    Parameters
    ----------
    pred_centroids : (P, G) tensor of predicted perturbation centroids.
    true_centroids : (P, G) tensor of ground-truth perturbation centroids.

    Returns
    -------
    ca : float  — higher is better
    pds : float — lower is better
    """
    n = pred_centroids.size(0)
    D = torch.cdist(pred_centroids, true_centroids, p=2.0)
    d_self = D.diagonal()
    off_mask = ~torch.eye(n, dtype=torch.bool, device=D.device)

    # CA: fraction of other perts that are farther from this pred than the true
    ca = (d_self.unsqueeze(1) < D).float().sum(1) / max(n - 1, 1)

    # PDS: d_self / (d_self + mean_off_diagonal)
    d_cross = (D * off_mask).sum(1) / max(n - 1, 1)
    pds = d_self / (d_self + d_cross + 1e-8)

    return ca.mean().item(), pds.mean().item()


def systema_pearson_delta(
    pred_centroids: torch.Tensor,
    true_centroids: torch.Tensor,
) -> float:
    """Genome-wide Pearson correlation of expression deltas re-centered by
    the true mean across perturbations.

    Higher is better.
    """
    origin = true_centroids.mean(dim=0, keepdim=True)
    p = pred_centroids - origin
    t = true_centroids - origin
    p = p - p.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    cov = (p * t).sum(-1)
    r = cov / (p.pow(2).sum(-1).clamp(1e-8).sqrt()
               * t.pow(2).sum(-1).clamp(1e-8).sqrt())
    return r.mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Differentially Expressed Gene Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def directional_accuracy(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    threshold: float | None = None,
) -> float:
    """Fraction of DE genes where the predicted direction matches truth.

    Parameters
    ----------
    pred_delta : (P, G) predicted expression deltas (pred − ctrl).
    true_delta : (P, G) true expression deltas (true − ctrl).
    threshold  : minimum |true_delta| to count a gene as DE.

    Returns
    -------
    float — higher is better
    """
    threshold = threshold if threshold is not None else config.DIR_ACC_THRESHOLD
    active = true_delta.abs() > threshold
    matches = (torch.sign(pred_delta) == torch.sign(true_delta)) & active
    per_pert = matches.sum(-1).float() / (active.sum(-1).float() + 1e-8)
    return per_pert.mean().item()


def pearson_delta_topk(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    k: int | None = None,
) -> float:
    """Pearson correlation of expression deltas on the top-k DE genes.

    Higher is better.
    """
    k = k or config.TOP_K_DE
    k = min(k, pred_delta.size(1))
    _, idx = torch.topk(true_delta.abs(), k=k, dim=-1)
    p = torch.gather(pred_delta, 1, idx)
    t = torch.gather(true_delta, 1, idx)
    p = p - p.mean(-1, keepdim=True)
    t = t - t.mean(-1, keepdim=True)
    cov = (p * t).sum(-1)
    r = cov / (p.pow(2).sum(-1).clamp(1e-8).sqrt()
               * t.pow(2).sum(-1).clamp(1e-8).sqrt())
    return r.mean().item()


def jaccard_topk(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    k: int | None = None,
) -> float:
    """Jaccard similarity of predicted vs true top-k DE gene sets.

    Higher is better.
    """
    k = k or config.TOP_K_DE
    k = min(k, pred_delta.size(1))
    _, ti = torch.topk(true_delta.abs(), k=k, dim=-1)
    _, pi = torch.topk(pred_delta.abs(), k=k, dim=-1)
    scores = []
    for t_row, p_row in zip(ti, pi):
        ts = set(t_row.tolist())
        ps = set(p_row.tolist())
        scores.append(len(ts & ps) / len(ts | ps) if (ts | ps) else 0.0)
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — Distributional Fidelity
# ═══════════════════════════════════════════════════════════════════════════

def energy_distance(
    pred_cells: torch.Tensor,
    true_cells: torch.Tensor,
    max_cells: int | None = None,
    seed: int | None = None,
) -> float:
    """Energy distance between predicted and observed cell distributions.

    Lower is better.

    Parameters
    ----------
    pred_cells : (Np, G) predicted single-cell profiles.
    true_cells : (Nt, G) observed single-cell profiles.
    max_cells  : subsample both sets if larger.
    """
    max_cells = max_cells or config.MAX_T3_CELLS
    pred_cells = _maybe_subsample(pred_cells, max_cells, seed)
    true_cells = _maybe_subsample(true_cells, max_cells, seed)

    d_pt = torch.cdist(pred_cells, true_cells, p=2.0).mean()
    # Use off-diagonal means to exclude self-distance (=0)
    n_p, n_t = pred_cells.size(0), true_cells.size(0)
    dpp = torch.cdist(pred_cells, pred_cells, p=2.0)
    dtt = torch.cdist(true_cells, true_cells, p=2.0)
    if n_p > 1:
        mask_pp = ~torch.eye(n_p, dtype=torch.bool, device=dpp.device)
        d_pp = dpp[mask_pp].mean()
    else:
        d_pp = torch.tensor(0.0, device=dpp.device)
    if n_t > 1:
        mask_tt = ~torch.eye(n_t, dtype=torch.bool, device=dtt.device)
        d_tt = dtt[mask_tt].mean()
    else:
        d_tt = torch.tensor(0.0, device=dtt.device)
    return max((2 * d_pt - d_pp - d_tt).item(), 0.0)


def mmd_rbf(
    pred_cells: torch.Tensor,
    true_cells: torch.Tensor,
    max_cells: int | None = None,
    seed: int | None = None,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel (median bandwidth heuristic).

    Lower is better.
    """
    max_cells = max_cells or config.MAX_T3_CELLS
    pred_cells = _maybe_subsample(pred_cells, max_cells, seed)
    true_cells = _maybe_subsample(true_cells, max_cells, seed)

    with torch.no_grad():
        d_tt = torch.cdist(true_cells, true_cells)
        # Exclude diagonal (self-distances = 0) to avoid biasing the median
        n_t = true_cells.size(0)
        off_diag = ~torch.eye(n_t, dtype=torch.bool, device=d_tt.device)
        sigma2 = d_tt.pow(2)[off_diag].median().clamp(1e-6).item() / 2.0

    def rbf(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * sigma2))

    kxx = rbf(pred_cells, pred_cells).mean()
    kyy = rbf(true_cells, true_cells).mean()
    kxy = rbf(pred_cells, true_cells).mean()
    return max((kxx - 2 * kxy + kyy).item(), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: compute all tiers at once
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    pred_centroids: torch.Tensor,
    true_centroids: torch.Tensor,
    ctrl_centroid: torch.Tensor,
    pred_cells_dict: dict[str, torch.Tensor] | None = None,
    true_cells_dict: dict[str, torch.Tensor] | None = None,
    pert_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute all T1/T2/T3 metrics and return a flat dictionary.

    Parameters
    ----------
    pred_centroids : (P, G)
    true_centroids : (P, G)
    ctrl_centroid  : (G,) control mean expression
    pred_cells_dict : {pert_name: (Np, G)} per-pert predicted cells (for T3)
    true_cells_dict : {pert_name: (Nt, G)} per-pert observed cells (for T3)
    pert_names : list of perturbation names matching centroid rows
    """
    pred_delta = pred_centroids - ctrl_centroid.unsqueeze(0)
    true_delta = true_centroids - ctrl_centroid.unsqueeze(0)

    # Tier 1
    ca, pds = centroid_accuracy_and_pds(pred_centroids, true_centroids)
    spd = systema_pearson_delta(pred_centroids, true_centroids)

    # Tier 2
    da = directional_accuracy(pred_delta, true_delta)
    pde = pearson_delta_topk(pred_delta, true_delta)
    jac = jaccard_topk(pred_delta, true_delta)

    results = {
        "T1_Centroid_Accuracy": ca,
        "T1_Profile_Distance_Score": pds,
        "T1_Systema_Pearson_Delta": spd,
        "T2_Directional_Accuracy": da,
        "T2_Pearson_Delta_TopK": pde,
        "T2_Jaccard_TopK": jac,
    }

    # Tier 3 (optional — requires per-cell data)
    if pred_cells_dict and true_cells_dict and pert_names:
        e_scores, m_scores = [], []
        for name in pert_names:
            if name in pred_cells_dict and name in true_cells_dict:
                e_scores.append(
                    energy_distance(pred_cells_dict[name], true_cells_dict[name])
                )
                m_scores.append(
                    mmd_rbf(pred_cells_dict[name], true_cells_dict[name])
                )
        if e_scores:
            results["T3_Energy_Distance"] = float(np.mean(e_scores))
            results["T3_MMD_RBF"] = float(np.mean(m_scores))

    return results


# ───────────────────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────────────────

def _maybe_subsample(
    x: torch.Tensor, max_n: int, seed: int | None = None,
) -> torch.Tensor:
    if x.size(0) <= max_n:
        return x
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    idx = torch.randperm(x.size(0), generator=gen, device=x.device)[:max_n]
    return x[idx]

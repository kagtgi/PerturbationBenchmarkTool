"""
Model registry for perturbation prediction evaluators.

Each model module exposes a ``run_eval(adata, config) -> dict`` function.
The registry maps model names to their evaluation entry points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scanpy as sc
    from types import ModuleType

# Lazy imports — each model has heavy dependencies that should only
# be loaded when that specific model is evaluated.
_MODEL_MODULES: dict[str, str] = {
    "gears": ".gears",
    "scgpt": ".scgpt",
    "state": ".state",
    "cell2sentence": ".cell2sentence",
    "cpa": ".cpa",
}

AVAILABLE_MODELS: list[str] = sorted(_MODEL_MODULES.keys())


def get_model_module(name: str) -> "ModuleType":
    """Import and return the model module by name."""
    import importlib
    if name not in _MODEL_MODULES:
        raise ValueError(
            f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}"
        )
    return importlib.import_module(_MODEL_MODULES[name], package=__name__)


def run_model_eval(name: str, adata: "sc.AnnData", cfg: dict) -> dict:
    """Run evaluation for a single model.

    Parameters
    ----------
    name : str
        Model name (one of ``AVAILABLE_MODELS``).
    adata : sc.AnnData
        Subsampled dataset (output of ``sampling.stratified_subsample``).
    cfg : dict
        Runtime configuration (typically ``vars(config)``).

    Returns
    -------
    dict with keys: ``model``, ``metrics`` (dict of metric values),
    ``pert_names`` (list), ``runtime_seconds`` (float).
    """
    mod = get_model_module(name)
    return mod.run_eval(adata, cfg)

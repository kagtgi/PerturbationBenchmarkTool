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

# Per-model pip requirements used by eval_runner's subprocess installer.
# Each model's run_eval() handles its own deps internally; this dict lets
# the subprocess wrapper pre-install packages before any top-level import.
MODEL_REQUIREMENTS: dict[str, list[str]] = {
    "gears": [
        "torch_geometric",
        "cell-gears",
        "scanpy",
    ],
    "state": [
        "uv",
        "huggingface_hub",
    ],
    "scgpt": [
        "huggingface_hub",
        "scanpy",
        "anndata",
    ],
    "cell2sentence": [
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "cell2sentence==1.1.0",
    ],
    "cpa": [
        "anndata>=0.10.0,<0.13.0",
        "scanpy>=1.10.0,<1.11.0",
        "scvi-tools>=1.0.0,<1.5.0",
        "lightning>=2.2.0,<2.4.0",
        "pytorch-lightning>=2.2.0,<2.4.0",
        "gdown",
        "pybiomart",
    ],
}


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

"""
Main entry point for running perturbation model evaluations.

Orchestrates: dataset loading → sampling → per-model evaluation → result saving.

Usage
-----
    python -m eval.eval_runner --data path/to/dataset.h5ad --models gears scgpt state
    python -m eval.eval_runner --data K562.h5ad --models all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from . import config
from .dataset import load_h5ad, ensure_raw_counts
from .sampling import stratified_subsample
from .models import AVAILABLE_MODELS, run_model_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_runtime_config(args: argparse.Namespace) -> dict:
    """Build a runtime config dict from CLI args + config.py defaults."""
    cfg = {
        "DATA_PATH": args.data or config.DATA_PATH,
        "RANDOM_SEED": args.seed if args.seed is not None else config.RANDOM_SEED,
        "DEVICE": args.device or config.DEVICE,
        "SUBSAMPLE_FRAC": config.SUBSAMPLE_FRAC,
        "MIN_CELLS_PER_PERT": config.MIN_CELLS_PER_PERT,
        "MAX_T3_CELLS": config.MAX_T3_CELLS,
        "CTRL_LABEL": config.CTRL_LABEL,
        "PERT_COL": config.PERT_COL,
        "TOP_K_DE": config.TOP_K_DE,
        "DIR_ACC_THRESHOLD": config.DIR_ACC_THRESHOLD,
        "OUTPUT_DIR": args.output or config.OUTPUT_DIR,
    }
    return cfg


def run(
    data_path: str | None = None,
    models: list[str] | None = None,
    device: str | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """Programmatic entry point for running evaluations.

    Parameters
    ----------
    data_path : path to .h5ad file (default: config.DATA_PATH)
    models : list of model names to evaluate (default: all)
    device : "cuda" or "cpu"
    seed : random seed
    output_dir : where to save results

    Returns
    -------
    List of result dicts, one per model.
    """
    data_path = data_path or config.DATA_PATH
    models = models or AVAILABLE_MODELS
    device = device or config.DEVICE
    seed = seed if seed is not None else config.RANDOM_SEED
    output_dir = output_dir or config.OUTPUT_DIR

    cfg = {
        "DATA_PATH": data_path,
        "RANDOM_SEED": seed,
        "DEVICE": device,
        "SUBSAMPLE_FRAC": config.SUBSAMPLE_FRAC,
        "MIN_CELLS_PER_PERT": config.MIN_CELLS_PER_PERT,
        "MAX_T3_CELLS": config.MAX_T3_CELLS,
        "CTRL_LABEL": config.CTRL_LABEL,
        "PERT_COL": config.PERT_COL,
        "TOP_K_DE": config.TOP_K_DE,
        "DIR_ACC_THRESHOLD": config.DIR_ACC_THRESHOLD,
        "OUTPUT_DIR": output_dir,
    }

    os.makedirs(output_dir, exist_ok=True)

    # Load & subsample once
    logger.info("Loading dataset: %s", data_path)
    adata = load_h5ad(data_path)
    adata = ensure_raw_counts(adata)

    logger.info("Stratified subsampling (%d%%) ...", int(config.SUBSAMPLE_FRAC * 100))
    adata_sub = stratified_subsample(adata)
    del adata  # free full dataset; only the subsampled copy is needed from here

    results: list[dict] = []
    for model_name in models:
        logger.info("=" * 60)
        logger.info("Evaluating: %s", model_name)
        logger.info("=" * 60)
        try:
            # Each model gets a fresh copy to avoid cross-contamination
            adata_copy = adata_sub.copy()
            result = run_model_eval(model_name, adata_copy, cfg)
            results.append(result)

            # Save individual result
            out_path = os.path.join(output_dir, f"{model_name}_results.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("Saved: %s", out_path)

        except Exception as e:
            logger.error("Model %s failed: %s", model_name, e, exc_info=True)
            results.append({
                "model": model_name,
                "metrics": {},
                "pert_names": [],
                "runtime_seconds": 0.0,
                "error": str(e),
            })

    # Save combined summary
    _save_summary(results, output_dir)
    return results


def _save_summary(results: list[dict], output_dir: str) -> None:
    """Save a combined CSV/JSON summary table."""
    rows = []
    for r in results:
        row = {"model": r["model"], "runtime_seconds": r.get("runtime_seconds", 0)}
        row.update(r.get("metrics", {}))
        if "error" in r:
            row["error"] = r["error"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "summary.csv")
    json_path = os.path.join(output_dir, "summary.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    logger.info("\n%s", df.to_string(index=False))
    logger.info("Summary saved to %s and %s", csv_path, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturbation Prediction Benchmark — Evaluation Runner",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help=f"Path to .h5ad dataset (default: {config.DATA_PATH})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to evaluate (default: all). Choices: {AVAILABLE_MODELS}",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help=f"Device: cuda or cpu (default: auto-detect → {config.DEVICE})",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=f"Random seed (default: {config.RANDOM_SEED})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output directory (default: {config.OUTPUT_DIR})",
    )
    args = parser.parse_args()

    model_list = args.models
    if model_list and "all" in model_list:
        model_list = AVAILABLE_MODELS

    run(
        data_path=args.data,
        models=model_list,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

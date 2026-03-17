"""
Main entry point for running perturbation model evaluations.

Orchestrates: dataset loading → sampling → per-model evaluation → result saving.

KEY IMPROVEMENTS:
- Each model runs in isolated subprocess (no library conflicts)
- Model-specific dependencies installed per-model
- No runtime restarts needed between models
- Timeout protection per model
- Comprehensive logging and error tracking

Usage
-----
    python -m eval.eval_runner --data path/to/dataset.h5ad --models gears scgpt c2s
    python -m eval.eval_runner --data K562.h5ad --models all
    python -m eval_runner --data K562.h5ad --models all --no-isolate  # Legacy mode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .dataset import load_h5ad, ensure_raw_counts
from .sampling import stratified_subsample
from .models import AVAILABLE_MODELS, MODEL_REQUIREMENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# =============================================================================

# Accurate pre-install requirements for each model, derived directly from
# Eval_.ipynb (cells 3, 6, 8, 10, 12).  These are the packages installed
# by each notebook cell *before* the model's own run_eval() is invoked.
# Each model's _install_dependencies() handles the full install at runtime;
# this list lets the subprocess wrapper pre-seed the heaviest packages so
# the first import succeeds quickly.
#
# Values from eval.models.MODEL_REQUIREMENTS (models/__init__.py) override
# these defaults — keep them in sync.
DEFAULT_MODEL_REQUIREMENTS: dict[str, list[str]] = {
    # Notebook cell 3: pip("torch_geometric"); pip("cell-gears", "scanpy")
    "gears": [
        "torch_geometric",
        "cell-gears",
        "scanpy",
    ],
    # Notebook cell 6: pip("uv", "huggingface_hub"); uv tool install arc-state
    "state": [
        "uv",
        "huggingface_hub",
    ],
    # Notebook cell 8: pip(scGPT git HEAD); pip("huggingface_hub","scanpy","anndata")
    # NOTE: scGPT is NOT on PyPI as a stable release — installed from git HEAD
    # inside run_eval().  Pre-installing aux deps only here.
    "scgpt": [
        "huggingface_hub",
        "scanpy",
        "anndata",
    ],
    # Notebook cell 10: pip("transformers>=4.45.0","accelerate>=0.34.0","bitsandbytes>=0.43.0")
    #                   pip("cell2sentence==1.1.0","anndata>=0.10.0","scanpy>=1.10.0")
    "cell2sentence": [
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "cell2sentence==1.1.0",
    ],
    # Notebook cell 12: pip("anndata>=0.10.0,<0.13.0"); pip("scanpy>=1.10.0,<1.11.0");
    #                   pip("scvi-tools>=1.0.0,<1.5.0"); pip("lightning>=2.2.0,<2.4.0");
    #                   pip("pytorch-lightning>=2.2.0,<2.4.0"); pip("gdown"); pip("pybiomart")
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

# Merge: models/__init__.py values win (they are also derived from the notebook)
MODEL_REQUIREMENTS = {**DEFAULT_MODEL_REQUIREMENTS, **MODEL_REQUIREMENTS}

# ---------------------------------------------------------------------------
# Known library conflicts — relevant ONLY for non-isolated (--no-isolate) mode.
# In subprocess isolation mode (default) every model runs in its own Python
# process, so conflicts are automatically avoided.
# ---------------------------------------------------------------------------
KNOWN_CONFLICTS: dict[tuple[str, str], str] = {
    # STATE: arc-state install evicts scipy/scanpy/anndata from sys.modules
    ("state", "gears"):         "arc-state install evicts scipy/scanpy/anndata from sys.modules",
    ("state", "scgpt"):         "arc-state install evicts scipy/scanpy/anndata from sys.modules",
    ("state", "cell2sentence"): "arc-state install evicts scipy/scanpy/anndata from sys.modules",
    ("state", "cpa"):           "arc-state install evicts scipy/scanpy/anndata from sys.modules",
    # GEARS ↔ CPA: torch_geometric vs. scvi-tools anndata version pin (<0.13.0)
    ("gears", "cpa"):           "torch_geometric conflicts with scvi-tools anndata<0.13.0 pin",
    # scGPT ↔ C2S: scGPT torchtext stub may conflict with C2S transformers
    ("scgpt", "cell2sentence"): "scGPT uses torchtext stub; C2S needs transformers>=4.45.0",
}

# Safe run order for non-isolated (in-process) mode:
#   1. GEARS  — minimal deps; no sys.modules side-effects
#   2. scGPT  — injects torchtext stub (harmless for subsequent models)
#   3. C2S    — upgrades transformers; OK after scGPT
#   4. CPA    — pins anndata/scanpy; clears its own module cache internally
#   5. STATE  — MUST be last: arc-state evicts scipy/anndata/scanpy on install
SAFE_RUN_ORDER: list[str] = ["gears", "scgpt", "cell2sentence", "cpa", "state"]

# Timeout per model in minutes
MODEL_TIMEOUT: dict[str, int] = {
    "gears":         20,
    "state":         25,
    "scgpt":         30,
    "cell2sentence": 90,   # LLM 2B-parameter inference is the slowest step
    "cpa":           30,
}


# =============================================================================
# CONFIGURATION BUILDERS
# =============================================================================

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
        "ISOLATE_MODELS": args.isolate,  # NEW: subprocess isolation flag
    }
    return cfg


# =============================================================================
# SUBPROCESS MODEL RUNNER (KEY FEATURE)
# =============================================================================

def run_model_in_subprocess(
    model_name: str,
    data_path: str,
    cfg: dict,
    output_dir: str,
    timeout_minutes: int = 30,
) -> dict:
    """
    Run a single model evaluation in an isolated subprocess.
    
    This prevents library version conflicts between models (e.g., transformers
    4.45.0 for C2S vs different versions for scGPT).
    
    Parameters
    ----------
    model_name : str
        Name of the model to evaluate.
    data_path : str
        Path to the .h5ad dataset.
    cfg : dict
        Runtime configuration.
    output_dir : str
        Directory to save results.
    timeout_minutes : int
        Maximum runtime before killing the subprocess.
    
    Returns
    -------
    dict
        Evaluation result with metrics, runtime, and status.
    """
    logger.info(f"🧬 Starting subprocess for: {model_name}")

    # Use absolute paths so the subprocess (which os.chdir to project_root)
    # and the parent process always agree on file locations.
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    result_file = os.path.join(output_dir, f"{model_name}_results.json")
    log_file = os.path.join(output_dir, f"{model_name}_log.txt")
    
    # Get model-specific requirements
    requirements = MODEL_REQUIREMENTS.get(model_name, [])
    
    # Project root = parent of the `eval/` package directory
    project_root = str(Path(__file__).parent.parent.resolve())

    # Create isolated runner script
    runner_script = f'''
import sys, os, warnings, json, time, logging, subprocess
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Ensure the project root is importable (eval/ package lives there)
_project_root = {repr(project_root)}
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)  # working directory for relative file paths

# ---------------------------------------------------------------------------
# STEP 1 — Pre-install model-specific packages in a single pip call.
#
# We do NOT use __import__ to skip "already installed" packages because
# pip-package names != Python module names for several deps:
#   scvi-tools  -> module: scvi       (not scvi_tools)
#   cell-gears  -> module: gears      (not cell_gears)
# Passing a version-ranged spec to pip is idempotent: pip exits immediately
# if the constraint is already satisfied, so there is no performance penalty.
# ---------------------------------------------------------------------------
requirements = {json.dumps(requirements)}
if requirements:
    logger.info(f"Pre-installing {{len(requirements)}} package(s) for {model_name} ...")
    try:
        result_pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + requirements,
            capture_output=True, text=True,
        )
        if result_pip.returncode != 0:
            # Show stderr so the user can debug pip failures
            # Use \\n so the generated script has a literal \\n (not a raw newline)
            logger.warning("Pre-install non-zero exit for {model_name}:\\n%s",
                           result_pip.stderr[-2000:])
        else:
            logger.info("Pre-install done.")
    except Exception as _e:
        # Non-fatal: each model's run_eval() installs its own deps anyway.
        logger.warning(f"Pre-install warning (non-fatal): {{_e}}")

# ---------------------------------------------------------------------------
# STEP 2 — Run model evaluation
# ---------------------------------------------------------------------------
start = time.time()
result = {{
    "model": "{model_name}",
    "status": "unknown",
    "runtime_seconds": 0,
    "metrics": {{}},
    "pert_names": [],
}}

try:
    import scanpy as sc
    from eval.models import run_model_eval

    logger.info("Loading data: {data_path}")
    adata = sc.read_h5ad("{data_path}")

    cfg = {json.dumps(cfg)}

    logger.info("Running {model_name} evaluation ...")
    result = run_model_eval("{model_name}", adata, cfg)
    result["status"] = "success"

except Exception as e:
    import traceback
    result["status"] = "failed"
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()
    logger.error(f"Evaluation failed: {{e}}")
    logger.error(traceback.format_exc())

finally:
    result["runtime_seconds"] = time.time() - start
    with open("{result_file}", "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"{model_name} completed in {{result['runtime_seconds']:.2f}}s")
'''
    
    # Write temporary runner script
    temp_script = os.path.join(tempfile.gettempdir(), f"run_{model_name}_{int(time.time())}.py")
    with open(temp_script, 'w') as f:
        f.write(runner_script)
    
    # Run in subprocess with timeout — stream output live to console/notebook
    timeout_seconds = timeout_minutes * 60
    log_lines: list[str] = [
        f"=== {model_name} Evaluation Log ===\n",
        f"Timestamp: {datetime.now().isoformat()}\n",
        f"Timeout: {timeout_minutes} minutes\n\n",
    ]
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout
            text=True,
            bufsize=1,                  # line-buffered
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            cwd=project_root,
        )

        # Stream lines live; collect for log file
        # Handle tqdm progress bars: when a line contains \r, only the
        # last segment is the current bar state.  We print it with \r so
        # the terminal overwrites the previous bar in-place, and only
        # log the final state to avoid bloating the log file.
        deadline = start_time + timeout_seconds
        timed_out = False
        prev_was_progress = False
        for line in proc.stdout:                         # type: ignore[union-attr]
            if '\r' in line:
                # tqdm-style progress: take only the last \r segment
                segments = line.rsplit('\r', 1)
                latest = segments[-1]
                if latest.strip():
                    # Overwrite current terminal line with latest bar
                    print(f"\r[{model_name}] {latest}", end="", flush=True)
                    prev_was_progress = True
                # Log only the final progress state (strip \r chars)
                log_lines.append(latest)
            else:
                if prev_was_progress:
                    # End the progress line before printing a normal line
                    print(flush=True)
                    prev_was_progress = False
                print(f"[{model_name}] {line}", end="", flush=True)
                log_lines.append(line)
            if time.time() > deadline:
                proc.kill()
                timed_out = True
                break
        if prev_was_progress:
            print(flush=True)  # ensure terminal moves past last bar
        proc.wait()

        # Save all collected output to log file
        with open(log_file, "w") as f:
            f.writelines(log_lines)

        if timed_out:
            raise subprocess.TimeoutExpired(proc.args, timeout_seconds)

        # Load results written by the subprocess
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                model_result = json.load(f)
        else:
            model_result = {
                "model": model_name,
                "status": "failed",
                "error": "No results file generated (subprocess may have crashed at import)",
                "runtime_seconds": time.time() - start_time,
                "metrics": {},
                "pert_names": [],
            }

    except subprocess.TimeoutExpired:
        logger.warning(f"⚠️  {model_name} timed out after {timeout_minutes} minutes")
        model_result = {
            "model": model_name,
            "status": "timeout",
            "error": f"Exceeded {timeout_minutes} minutes",
            "runtime_seconds": timeout_seconds,
            "metrics": {},
            "pert_names": [],
        }

    except Exception as e:
        logger.error(f"❌ Subprocess failed for {model_name}: {e}")
        model_result = {
            "model": model_name,
            "status": "failed",
            "error": str(e),
            "runtime_seconds": time.time() - start_time,
            "metrics": {},
            "pert_names": [],
        }
    
    finally:
        # Cleanup temp script
        if os.path.exists(temp_script):
            try:
                os.remove(temp_script)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    status_icon = "✅" if model_result.get('status') == 'success' else "❌"
    logger.info(f"{status_icon} {model_name}: {elapsed/60:.2f} min (status: {model_result.get('status')})")
    
    return model_result


# =============================================================================
# LEGACY IN-PROCESS RUNNER (for compatible models)
# =============================================================================

def run_model_in_process(
    model_name: str,
    adata_copy,
    cfg: dict,
    output_dir: str,
) -> dict:
    """
    Run model evaluation in the current process (legacy mode).
    
    Use this only when all models share compatible dependencies.
    """
    from .models import run_model_eval
    
    logger.info(f"🧬 Evaluating (in-process): {model_name}")
    start_time = time.time()
    
    try:
        result = run_model_eval(model_name, adata_copy, cfg)
        result["status"] = "success"
        
    except Exception as e:
        logger.error(f"❌ Model {model_name} failed: {e}", exc_info=True)
        result = {
            "model": model_name,
            "metrics": {},
            "pert_names": [],
            "runtime_seconds": time.time() - start_time,
            "error": str(e),
            "status": "failed",
        }
    
    return result


# =============================================================================
# LOG / DEBUG HELPERS
# =============================================================================

def print_logs(output_dir: str | None = None, model: str | None = None) -> None:
    """Print all evaluation logs from a previous run.

    Call this after a failed run to see what went wrong::

        from eval.eval_runner import print_logs
        print_logs()           # all models
        print_logs(model="gears")  # single model

    Parameters
    ----------
    output_dir : str, optional
        Directory where logs were saved (default: config.OUTPUT_DIR).
    model : str, optional
        If given, print only that model's log; otherwise print all.
    """
    import glob as _glob

    output_dir = output_dir or config.OUTPUT_DIR
    pattern = (
        os.path.join(output_dir, f"{model}_log.txt")
        if model
        else os.path.join(output_dir, "*_log.txt")
    )
    files = sorted(_glob.glob(pattern))
    if not files:
        print(f"No log files found in {output_dir!r}. "
              "Make sure output_dir matches what was passed to run().")
        return
    for path in files:
        print(f"\n{'='*70}")
        print(f"LOG: {path}")
        print('='*70)
        with open(path) as f:
            print(f.read())


def print_errors(output_dir: str | None = None) -> None:
    """Print error messages and tracebacks from failed model runs.

    Call this after a failed run::

        from eval.eval_runner import print_errors
        print_errors()
    """
    import glob as _glob

    output_dir = output_dir or config.OUTPUT_DIR
    for path in sorted(_glob.glob(os.path.join(output_dir, "*_results.json"))):
        with open(path) as f:
            res = json.load(f)
        if res.get("status") != "success":
            print(f"\n{'='*70}")
            print(f"MODEL: {res.get('model')}  status={res.get('status')}")
            print(f"Error: {res.get('error')}")
            tb = res.get("traceback")
            if tb:
                print("Traceback:")
                print(tb)


# =============================================================================
# MAIN EVALUATION ORCHESTRATOR
# =============================================================================

def run(
    data_path: str | None = None,
    models: list[str] | None = None,
    device: str | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
    isolate: bool = True,
) -> list[dict]:
    """
    Programmatic entry point for running evaluations.
    
    Parameters
    ----------
    data_path : str
        Path to .h5ad file.
    models : list[str]
        List of model names to evaluate.
    device : str
        "cuda" or "cpu".
    seed : int
        Random seed.
    output_dir : str
        Where to save results.
    isolate : bool
        If True, run each model in isolated subprocess (recommended).
    
    Returns
    -------
    list[dict]
        List of result dicts, one per model.
    """
    data_path = data_path or config.DATA_PATH
    models = list(models or AVAILABLE_MODELS)
    device = device or config.DEVICE
    seed = seed if seed is not None else config.RANDOM_SEED
    output_dir = output_dir or config.OUTPUT_DIR

    # Always use absolute paths so subprocesses (which chdir to project root)
    # and the parent process agree on file locations regardless of CWD.
    output_dir = str(Path(output_dir).resolve())
    data_path = str(Path(data_path).resolve())
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Non-isolated mode: enforce safe run order and warn about conflicts.
    # In subprocess isolation mode every model runs in its own Python process
    # so conflicts are automatically avoided — no reordering needed.
    # -----------------------------------------------------------------------
    if not isolate:
        # Reorder models to match SAFE_RUN_ORDER
        ordered = [m for m in SAFE_RUN_ORDER if m in models]
        ordered += [m for m in models if m not in SAFE_RUN_ORDER]
        if ordered != models:
            logger.warning(
                "Non-isolated mode: reordering models to safe execution order: %s",
                " → ".join(ordered),
            )
        models = ordered

        # Warn about known conflicts
        active = set(models)
        conflicts_found = []
        for (a, b), reason in KNOWN_CONFLICTS.items():
            if a in active and b in active:
                conflicts_found.append(f"  {a} ↔ {b}: {reason}")
        if conflicts_found:
            logger.warning(
                "⚠️  Non-isolated mode with conflicting models detected!\n"
                "   Use --isolate (default) to avoid these conflicts.\n"
                "   Detected conflicts:\n%s",
                "\n".join(conflicts_found),
            )

    # Log configuration
    logger.info("=" * 70)
    logger.info("🚀 PERTURBATION BENCHMARK - EVALUATION RUNNER")
    logger.info("=" * 70)
    logger.info(f"📁 Data: {data_path}")
    logger.info(f"📂 Output: {output_dir}")
    logger.info(f"🧬 Models: {models}")
    logger.info(f"🔒 Isolation: {'Enabled (subprocess)' if isolate else 'Disabled (in-process)'}")
    logger.info(f"🎲 Seed: {seed}")
    logger.info(f"🖥️  Device: {device}")
    logger.info("=" * 70)
    
    # Load & subsample ONCE (shared across all models)
    logger.info("📊 Loading dataset...")
    adata = load_h5ad(data_path)
    adata = ensure_raw_counts(adata)
    
    logger.info(f"📉 Stratified subsampling ({int(config.SUBSAMPLE_FRAC * 100)}%)...")
    adata_sub = stratified_subsample(adata)
    del adata  # Free full dataset
    
    logger.info(f"✅ Data ready: {adata_sub.shape}")
    
    # Save subsampled data for subprocess access
    temp_data_path = os.path.join(output_dir, "temp_subsampled.h5ad")
    adata_sub.write_h5ad(temp_data_path)
    logger.info(f"💾 Subsampled data saved to: {temp_data_path}")
    
    results: list[dict] = []
    summary_data = []
    
    for i, model_name in enumerate(models, 1):
        logger.info("\n" + "=" * 70)
        logger.info(f"[{i}/{len(models)}] Evaluating: {model_name.upper()}")
        logger.info("=" * 70)
        
        model_start = time.time()
        
        try:
            if isolate:
                # Run in isolated subprocess (recommended)
                timeout = MODEL_TIMEOUT.get(model_name, 30)
                result = run_model_in_subprocess(
                    model_name=model_name,
                    data_path=temp_data_path,
                    cfg={
                        "RANDOM_SEED": seed,
                        "DEVICE": device,
                        "CTRL_LABEL": config.CTRL_LABEL,
                        "PERT_COL": config.PERT_COL,
                        "TOP_K_DE": config.TOP_K_DE,
                        "MAX_T3_CELLS": config.MAX_T3_CELLS,
                        "MIN_CELLS_PER_PERT": config.MIN_CELLS_PER_PERT,
                        "DIR_ACC_THRESHOLD": config.DIR_ACC_THRESHOLD,
                        "OUTPUT_DIR": output_dir,
                    },
                    output_dir=output_dir,
                    timeout_minutes=timeout,
                )
            else:
                # Run in current process (legacy mode)
                adata_copy = adata_sub.copy()
                result = run_model_in_process(
                    model_name=model_name,
                    adata_copy=adata_copy,
                    cfg={
                        "RANDOM_SEED": seed,
                        "DEVICE": device,
                        "CTRL_LABEL": config.CTRL_LABEL,
                        "PERT_COL": config.PERT_COL,
                        "TOP_K_DE": config.TOP_K_DE,
                        "MAX_T3_CELLS": config.MAX_T3_CELLS,
                        "MIN_CELLS_PER_PERT": config.MIN_CELLS_PER_PERT,
                        "DIR_ACC_THRESHOLD": config.DIR_ACC_THRESHOLD,
                        "OUTPUT_DIR": output_dir,
                    },
                    output_dir=output_dir,
                )
            
            results.append(result)
            summary_data.append({
                "model": model_name,
                "status": result.get("status", "unknown"),
                "runtime_seconds": result.get("runtime_seconds", 0),
                "error": result.get("error", None),
            })
            
        except Exception as e:
            logger.error(f"❌ Unexpected error for {model_name}: {e}", exc_info=True)
            results.append({
                "model": model_name,
                "metrics": {},
                "pert_names": [],
                "runtime_seconds": time.time() - model_start,
                "error": str(e),
                "status": "exception",
            })
            summary_data.append({
                "model": model_name,
                "status": "exception",
                "runtime_seconds": time.time() - model_start,
                "error": str(e),
            })
        
        # Garbage collection between models
        import gc
        gc.collect()
        
        # Small delay to allow GPU memory to clear
        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
    
    # Cleanup temp data
    if os.path.exists(temp_data_path):
        try:
            os.remove(temp_data_path)
        except Exception:
            pass
    
    # Save combined summary
    _save_summary(results, output_dir, summary_data)

    # Print final report
    _print_final_report(summary_data)

    # If any models failed, automatically show their errors so the user
    # doesn't have to hunt for log files manually.
    failed = [s for s in summary_data if s["status"] not in ("success", "timeout")]
    if failed:
        print(f"\n{'='*70}")
        print("🔍 AUTO-DIAGNOSE — error details for failed models:")
        print(f"{'='*70}")
        print_errors(output_dir)
        print(f"\n💡 For full subprocess output run:  print_logs('{output_dir}')")
        print(f"   Log files:  {output_dir}/<model>_log.txt")
        print(f"{'='*70}\n")

    return results


# =============================================================================
# SUMMARY & REPORTING
# =============================================================================

def _save_summary(results: list[dict], output_dir: str, summary_data: list[dict]) -> None:
    """Save a combined CSV/JSON summary table."""
    # Detailed results
    rows = []
    for r in results:
        row = {
            "model": r["model"],
            "status": r.get("status", "unknown"),
            "runtime_seconds": r.get("runtime_seconds", 0),
        }
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
    
    # Summary table (lighter)
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 DETAILED RESULTS")
    logger.info("=" * 70)
    logger.info("\n%s", df.to_string(index=False))
    logger.info("\n📁 Saved: %s, %s, %s", csv_path, json_path, summary_csv)


def _print_final_report(summary_data: list[dict]) -> None:
    """Print a final summary report."""
    total = len(summary_data)
    success = sum(1 for s in summary_data if s["status"] == "success")
    failed = sum(1 for s in summary_data if s["status"] in ["failed", "exception"])
    timeout = sum(1 for s in summary_data if s["status"] == "timeout")
    total_time = sum(s["runtime_seconds"] for s in summary_data)
    
    print("\n" + "=" * 70)
    print("🏁 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Total Models:     {total}")
    print(f"  ✅ Successful:    {success}")
    print(f"  ❌ Failed:        {failed}")
    print(f"  ⏱️  Timeout:       {timeout}")
    print(f"  ⏱️  Total Time:    {total_time/60:.2f} min")
    print("=" * 70)
    
    if success == total:
        print("  🎉 All models completed successfully!")
    elif success > 0:
        print(f"  ⚠️  {success}/{total} models succeeded")
    else:
        print("  ❌ All models failed - check logs for details")
    
    print("=" * 70)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturbation Prediction Benchmark — Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval.eval_runner --data K562.h5ad --models all
  python -m eval.eval_runner --data K562.h5ad --models cell2sentence scgpt
  python -m eval.eval_runner --data K562.h5ad --models all --no-isolate  # Legacy mode
  python -m eval.eval_runner --data K562.h5ad --models all --timeout 60  # 60 min per model
        """,
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help=f"Path to .h5ad dataset (default: {config.DATA_PATH})",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to evaluate. Choices: {AVAILABLE_MODELS}. Use 'all' for all models.",
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
    parser.add_argument(
        "--isolate", action="store_true", default=True,
        help="Run each model in isolated subprocess (default: True, recommended)",
    )
    parser.add_argument(
        "--no-isolate", action="store_false", dest="isolate",
        help="Run all models in same process (faster but may cause conflicts)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Timeout per model in minutes (default: model-specific)",
    )
    
    args = parser.parse_args()
    
    model_list = args.models
    if model_list and "all" in model_list:
        model_list = AVAILABLE_MODELS
    
    # Override timeout if specified
    if args.timeout:
        for model in MODEL_TIMEOUT:
            MODEL_TIMEOUT[model] = args.timeout
    
    run(
        data_path=args.data,
        models=model_list,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
        isolate=args.isolate,
    )


if __name__ == "__main__":
    main()
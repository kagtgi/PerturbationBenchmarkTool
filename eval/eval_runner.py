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

# Define requirements for each model (prevents version conflicts)
DEFAULT_MODEL_REQUIREMENTS = {
    "cell2sentence": [
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0",
        "cell2sentence==1.1.0",
    ],
    "scgpt": [
        "scgpt==0.1.2",
        "transformers>=4.45.0",
        "anndata>=0.10.0",
    ],
    "cpa": [
        "scvi-tools==1.0.0",
        "pyro-ppl==1.8.4",
        "anndata>=0.10.0",
        "scanpy>=1.10.0",
    ],
    "gears": [
        "torch>=2.0.0",
        "anndata>=0.10.0",
        "scanpy>=1.10.0",
    ],
    "scgen": [
        "scgen==2.1.0",
        "anndata>=0.10.0",
        "scanpy>=1.10.0",
    ],
}

# Merge with any model-specific overrides
MODEL_REQUIREMENTS = {**DEFAULT_MODEL_REQUIREMENTS, **MODEL_REQUIREMENTS}

# Timeout per model (minutes)
MODEL_TIMEOUT = {
    "cell2sentence": 30,
    "scgpt": 30,
    "cpa": 30,
    "gears": 20,
    "scgen": 20,
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
    
    start_time = time.time()
    result_file = os.path.join(output_dir, f"{model_name}_results.json")
    log_file = os.path.join(output_dir, f"{model_name}_log.txt")
    
    # Get model-specific requirements
    requirements = MODEL_REQUIREMENTS.get(model_name, [])
    
    # Create isolated runner script
    runner_script = f'''
import sys
import os
import warnings
import json
import time
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# STEP 1: Install model-specific dependencies FIRST
# =============================================================================
requirements = {json.dumps(requirements)}
installed = []

for pkg in requirements:
    pkg_name = pkg.split("==")[0].split(">=")[0].split("<=")[0]
    try:
        __import__(pkg_name.replace("-", "_"))
        logger.info(f"✅ Already installed: {{pkg_name}}")
    except ImportError:
        logger.info(f"📦 Installing: {{pkg}}")
        import subprocess
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            installed.append(pkg)
            logger.info(f"✅ Installed: {{pkg}}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to install {{pkg}}: {{e}}")

# Force reload of any cached modules
if installed:
    logger.info("🔄 Restarting imports after package installation...")
    import importlib
    for mod in list(sys.modules.keys()):
        if mod not in ["sys", "os", "json", "time", "logging", "warnings"]:
            del sys.modules[mod]

# =============================================================================
# STEP 2: Run model evaluation
# =============================================================================
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
    
    # Load data
    logger.info(f"📁 Loading data: {data_path}")
    adata = sc.read_h5ad("{data_path}")
    
    # Load config
    cfg = {json.dumps(cfg)}
    
    # Run evaluation
    logger.info(f"🚀 Running evaluation for: {model_name}")
    result = run_model_eval("{model_name}", adata, cfg)
    result["status"] = "success"
    
except Exception as e:
    import traceback
    result["status"] = "failed"
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()
    logger.error(f"❌ Evaluation failed: {{e}}")
    logger.error(traceback.format_exc())

finally:
    result["runtime_seconds"] = time.time() - start
    
    # Save results
    with open("{result_file}", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"✅ {{'{model_name}'}} completed in {{result['runtime_seconds']:.2f}}s")
'''
    
    # Write temporary runner script
    temp_script = os.path.join(tempfile.gettempdir(), f"run_{model_name}_{int(time.time())}.py")
    with open(temp_script, 'w') as f:
        f.write(runner_script)
    
    # Run in subprocess with timeout
    timeout_seconds = timeout_minutes * 60
    try:
        process_result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            cwd=os.getcwd(),
        )
        
        # Save logs
        with open(log_file, 'w') as f:
            f.write(f"=== {model_name} Evaluation Log ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Timeout: {timeout_minutes} minutes\n\n")
            f.write(f"STDOUT:\n{process_result.stdout}\n\n")
            f.write(f"STDERR:\n{process_result.stderr}\n")
        
        # Load results
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                model_result = json.load(f)
        else:
            model_result = {
                'model': model_name,
                'status': 'failed',
                'error': 'No results file generated',
                'runtime_seconds': time.time() - start_time,
                'metrics': {},
                'pert_names': [],
            }
            
    except subprocess.TimeoutExpired:
        logger.warning(f"⚠️  {model_name} timed out after {timeout_minutes} minutes")
        model_result = {
            'model': model_name,
            'status': 'timeout',
            'error': f'Exceeded {timeout_minutes} minutes',
            'runtime_seconds': timeout_seconds,
            'metrics': {},
            'pert_names': [],
        }
        
    except Exception as e:
        logger.error(f"❌ Subprocess failed for {model_name}: {e}")
        model_result = {
            'model': model_name,
            'status': 'failed',
            'error': str(e),
            'runtime_seconds': time.time() - start_time,
            'metrics': {},
            'pert_names': [],
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
    models = models or AVAILABLE_MODELS
    device = device or config.DEVICE
    seed = seed if seed is not None else config.RANDOM_SEED
    output_dir = output_dir or config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
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
"""
Aggregate results from all model evaluations into a single table.

Usage
-----
    python -m eval.collect_results --results-dir results/
    python -m eval.collect_results --results-dir results/ --format csv json table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Metric display info: (column_name, display_name, direction)
METRIC_DISPLAY = [
    ("T1_Centroid_Accuracy",      "Centroid Accuracy (CA)",       "higher"),
    ("T1_Profile_Distance_Score", "Profile Distance Score (PDS)", "lower"),
    ("T1_Systema_Pearson_Delta",  "Systema Pearson Delta",        "higher"),
    ("T2_Directional_Accuracy",   "Directional Accuracy",         "higher"),
    ("T2_Pearson_Delta_TopK",     "Pearson Delta Top-K",          "higher"),
    ("T2_Jaccard_TopK",           "Jaccard DE Top-K",             "higher"),
    ("T3_Energy_Distance",        "Energy Distance",              "lower"),
    ("T3_MMD_RBF",                "MMD (RBF kernel)",             "lower"),
]


def collect(results_dir: str | None = None) -> pd.DataFrame:
    """Load all *_results.json files from the results directory.

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    results_dir = results_dir or config.OUTPUT_DIR
    result_files = sorted(Path(results_dir).glob("*_results.json"))

    if not result_files:
        # Try summary.json as fallback
        summary = Path(results_dir) / "summary.json"
        if summary.exists():
            with open(summary) as f:
                rows = json.load(f)
            return pd.DataFrame(rows)
        logger.warning("No result files found in %s", results_dir)
        return pd.DataFrame()

    rows = []
    for fp in result_files:
        with open(fp) as f:
            data = json.load(f)
        row = {"model": data.get("model", fp.stem)}
        row["runtime_seconds"] = data.get("runtime_seconds", 0)
        row["n_perturbations"] = len(data.get("pert_names", []))
        row.update(data.get("metrics", {}))
        if "error" in data:
            row["error"] = data["error"]
        rows.append(row)

    return pd.DataFrame(rows)


def pretty_table(df: pd.DataFrame) -> str:
    """Format the results as a readable comparison table."""
    if df.empty:
        return "No results to display."

    lines = []
    sep = "=" * 72
    lines.append(sep)
    lines.append("  PERTURBATION PREDICTION BENCHMARK — RESULTS")
    lines.append(sep)

    # Header
    models = df["model"].tolist()
    header = f"  {'Metric':<38}"
    for m in models:
        header += f" {m:>12}"
    header += "  Dir"
    lines.append(header)
    lines.append(f"  {'-' * 68}")

    # Rows
    for col, display, direction in METRIC_DISPLAY:
        row = f"  {display:<38}"
        for _, r in df.iterrows():
            val = r.get(col)
            if val is not None and not pd.isna(val):
                row += f" {val:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        arrow = "up" if direction == "higher" else "down"
        row += f"  {arrow}"
        lines.append(row)

    lines.append(f"  {'-' * 68}")

    # Runtime
    rt_row = f"  {'Runtime (seconds)':<38}"
    for _, r in df.iterrows():
        rt_row += f" {r.get('runtime_seconds', 0):>12.1f}"
    lines.append(rt_row)

    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and display perturbation benchmark results",
    )
    parser.add_argument(
        "--results-dir", type=str, default=config.OUTPUT_DIR,
        help=f"Directory containing result JSON files (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format", nargs="+", default=["table", "csv"],
        choices=["table", "csv", "json"],
        help="Output formats (default: table csv)",
    )
    args = parser.parse_args()

    df = collect(args.results_dir)
    if df.empty:
        logger.error("No results found in %s", args.results_dir)
        sys.exit(1)

    if "table" in args.format:
        print(pretty_table(df))

    if "csv" in args.format:
        out = os.path.join(args.results_dir, "summary.csv")
        df.to_csv(out, index=False)
        logger.info("CSV saved: %s", out)

    if "json" in args.format:
        out = os.path.join(args.results_dir, "summary.json")
        df.to_json(out, orient="records", indent=2)
        logger.info("JSON saved: %s", out)


if __name__ == "__main__":
    main()

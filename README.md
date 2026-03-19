# Perturbation Prediction Benchmark Tool

A reusable evaluation framework for benchmarking single-cell gene perturbation
prediction models on Perturb-seq datasets stored as `.h5ad` files.

---

## Supported Models

| Model | Approach | GPU | ~Runtime | ~Disk |
|-------|----------|-----|----------|-------|
| **GEARS** | Graph neural network | Recommended | 10–20 min | 200 MB |
| **scGPT** | Pre-trained Transformer | Recommended | 20–30 min | 500 MB |
| **STATE** | Foundation model (HVG) | Recommended | 15–25 min | 1 GB |
| **Cell2Sentence** | LLM-based (Gemma-2-2B) | **Required** | 60–90 min | 10 GB |
| **CPA** | Conditional Perturbation Autoencoder | Recommended | 20–30 min | 300 MB |

> **First run**: each model downloads pretrained weights automatically.
> Total disk for all five: ~12 GB. Internet access is required.

---

## Evaluation Metrics (3 Tiers)

### Tier 1 — Whole-Profile Identity
| Metric | Direction | Description |
|--------|-----------|-------------|
| Centroid Accuracy (CA) | ↑ higher | Predicted centroid closest to correct perturbation |
| Profile Distance Score (PDS) | ↓ lower | Relative distance ranking among all perturbations |
| Systema Pearson Delta | ↑ higher | Genome-wide correlation of expression changes |

### Tier 2 — Differentially Expressed Gene Accuracy
| Metric | Direction | Description |
|--------|-----------|-------------|
| Directional Accuracy | ↑ higher | Fraction of DE genes with correct up/down direction |
| Pearson Delta Top-K | ↑ higher | Correlation of top-K DE gene expression changes |
| Jaccard DE Top-K | ↑ higher | Overlap of predicted vs observed top-K DE gene sets |

### Tier 3 — Distributional Fidelity
| Metric | Direction | Description |
|--------|-----------|-------------|
| Energy Distance | ↓ lower | Statistical distance between predicted/observed distributions |
| MMD (RBF kernel) | ↓ lower | Kernel-based distributional divergence |

---

## Repository Structure

```
├── README.md
├── Eval_.ipynb                  # Original monolithic notebook (reference)
└── eval/
    ├── __init__.py
    ├── __main__.py              # python -m eval entry point
    ├── config.py                # All configuration parameters
    ├── dataset.py               # .h5ad loading and preprocessing
    ├── sampling.py              # Centralized stratified subsampling
    ├── metrics.py               # Unified T1/T2/T3 metric functions
    ├── eval_runner.py           # Main orchestrator (subprocess isolation)
    ├── collect_results.py       # Result aggregation and display
    ├── models/
    │   ├── __init__.py          # Model registry + per-model requirements
    │   ├── gears.py
    │   ├── scgpt.py
    │   ├── state.py
    │   ├── cell2sentence.py
    │   └── cpa.py
    ├── notebooks/
    │   └── tutorial.ipynb       # Step-by-step Colab tutorial
    └── results/                 # Output directory (created at runtime)
```

---

## Quick Start

### 1. Prerequisites

```
Python ≥ 3.10
PyTorch (any recent version, CUDA build recommended)
git  (required — CPA and Cell2Sentence clone their repos on first run)
```

### 2. Install base dependencies

```bash
pip install anndata scanpy scipy pandas matplotlib torch
```

Model-specific packages (transformers, scvi-tools, torch-geometric, etc.) are
**installed automatically** the first time each model runs.

### 3. Configure your dataset

Edit `eval/config.py`:

```python
DATA_PATH  = "/path/to/your/dataset.h5ad"
DEVICE     = "cuda"               # or "cpu"
CTRL_LABEL = "non-targeting"      # obs label for control cells
PERT_COL   = "gene"               # obs column holding perturbation names
```

Or pass `--data` on the CLI — no file edit required.

### 4. Run evaluation

**All models (recommended):**
```bash
python -m eval --data K562.h5ad --models all
```

**Specific models:**
```bash
python -m eval --data K562.h5ad --models gears scgpt state
```

**Custom timeout or output directory:**
```bash
python -m eval --data K562.h5ad --models all \
    --device cuda --seed 42 \
    --output my_results/ --timeout 120
```

### 5. Collect and compare results

```bash
python -m eval.collect_results --results-dir results/
```

Produces a formatted comparison table plus `results/summary.csv`.

---

## Library Isolation & Run Order

Each model runs in its own **subprocess** (the default). This means:

- No library conflicts between models — each subprocess installs its own
  version-pinned dependencies independently.
- No Python runtime restart needed between models.
- CPA pins strict versions (`anndata<0.13.0`, `scanpy<1.11.0`) that modify
  the shared site-packages. The runner always executes models in the safe order
  `gears → scgpt → cell2sentence → cpa → state`, so CPA's pins only affect
  STATE — which uses its own `uv`-isolated environment and is unaffected.

To run all models in the **same process** (faster, no subprocess overhead,
but may hit library conflicts):

```bash
python -m eval --data K562.h5ad --models all --no-isolate
```

---

## Dataset Requirements

Your `.h5ad` file must have:

| Field | Default key | Description |
|-------|------------|-------------|
| `adata.X` | — | Expression matrix (raw counts preferred) |
| `adata.obs[PERT_COL]` | `"gene"` | Perturbation label per cell |
| Control value | `"non-targeting"` | Value in `PERT_COL` for control cells |

Set `PERT_COL` and `CTRL_LABEL` in `config.py` to match your data.
No other changes are needed.

---

## Programmatic API

```python
from eval.eval_runner import run

results = run(
    data_path="K562.h5ad",
    models=["gears", "scgpt", "cpa"],   # or None for all
    device="cuda",
    seed=42,
    output_dir="my_results/",
    isolate=True,                        # default, recommended
)

for r in results:
    if r.get("status") == "success":
        ca  = r["metrics"]["T1_Centroid_Accuracy"]
        pds = r["metrics"]["T1_Profile_Distance_Score"]
        print(f"{r['model']:20s}  CA={ca:.4f}  PDS={pds:.4f}")
```

Collect results programmatically:

```python
from eval.collect_results import collect, pretty_table

df = collect("my_results/")
print(pretty_table(df))
df.to_csv("comparison.csv", index=False)
```

---

## Debugging Failed Runs

Errors and tracebacks are printed automatically at the end of a run.
For manual inspection:

```python
from eval.eval_runner import print_errors, print_logs

# Error message + traceback for every failed model:
print_errors("results/")

# Full subprocess stdout/stderr log for one model:
print_logs("results/", model="gears")

# Logs for all models:
print_logs("results/")
```

Log files: `results/{model}_log.txt`
Result JSON (including traceback on failure): `results/{model}_results.json`

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All models fail in < 5 s | Wrong working directory | `cd PerturbationBenchmarkTool` before running |
| `ModuleNotFoundError: No module named 'eval'` | Not in project root | See above |
| `CUDA not available` / Cell2Sentence fails | GPU required for C2S | Use a CUDA runtime or skip `cell2sentence` |
| `Gene overlap = 0` (CPA) | Control label mismatch | Set `CTRL_LABEL` in `config.py` to match your data |
| `No results file generated` | Subprocess crashed at import | Check `{model}_log.txt` for the full traceback |
| Model times out | Dataset too large or slow GPU | Increase `--timeout` or reduce `SUBSAMPLE_FRAC` in `config.py` |
| STATE fails with `uv` not found | `uv` not installed | `pip install uv` or run once — STATE installs it automatically |
| CPA fails with git error | `git` not installed | Install git: `apt install git` / `brew install git` |

---

## Tutorial

For an interactive walkthrough, visit the [tutorial notebook on Google Colab](https://colab.research.google.com/drive/1Ck5_uX4FRyrpno-fWF3vEX2dREcVFvOZ?usp=sharing).

---

## Example Output

```
Metric                                 Cell2Sentence          CPA        GEARS        scGPT        STATE  Dir
--------------------------------------------------------------------
Centroid Accuracy (CA)                       0.0167       0.0100       0.4998       0.5000       0.5033  up
Profile Distance Score (PDS)                 0.4994       0.4995       0.4997       0.4998       0.4999  down
Systema Pearson Delta                        0.1543       0.1546       0.0382       0.0578       0.0636  up
Directional Accuracy                         0.5020       0.5509       0.5235       0.5835       0.5251  up
Pearson Delta Top-K                          0.4057       0.2433       0.0613       0.1349       0.2595  up
Jaccard DE Top-K                             0.0148       0.0358       0.0190       0.0720       0.0317  up
Energy Distance                             87.2667      48.6083      24.6346      40.0196      34.4977  down
MMD (RBF kernel)                             1.1086       0.5717       0.4670       1.0282       0.7345  down
--------------------------------------------------------------------
Runtime (seconds)                            3304.5        140.3        445.4        321.7        348.7
========================================================================
```

---

## Requirements Summary

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | |
| PyTorch ≥ 2.0 | CUDA build strongly recommended |
| CUDA GPU | Required for Cell2Sentence; recommended for all others |
| `git` | CPA and Cell2Sentence clone repos on first run |
| Internet access | All models download pretrained weights on first run |
| ~12 GB disk | For all five model checkpoints combined |

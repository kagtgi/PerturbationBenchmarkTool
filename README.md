# Perturbation Prediction Benchmark Tool

A reusable evaluation framework for benchmarking single-cell gene perturbation prediction models on Perturb-seq datasets stored as `.h5ad` files.

## Supported Models

| Model | Type | Reference |
|-------|------|-----------|
| **GEARS** | Graph Auto-Regressive Learning | [Roohani et al. 2023](https://github.com/snap-stanford/GEARS) |
| **scGPT** | Pre-trained Transformer | [Cui et al. 2024](https://github.com/bowang-lab/scGPT) |
| **STATE** | Spatial Transcriptomics Transfer | [Arc Institute](https://huggingface.co/arcinstitute/ST-HVG-Replogle) |
| **Cell2Sentence** | LLM-based (Gemma-2-2B) | [Van Dijk Lab](https://github.com/vandijklab/cell2sentence) |
| **CPA** | Conditional Perturbation Autoencoder | [Lotfollahi et al. 2023](https://github.com/theislab/cpa) |

## Evaluation Metrics (3 Tiers)

### Tier 1 — Whole-Profile Identity
- **Centroid Accuracy (CA)** ↑ — Is the predicted profile closest to the correct perturbation?
- **Profile Distance Score (PDS)** ↓ — Relative distance ranking among all perturbations.
- **Systema Pearson Delta** ↑ — Genome-wide correlation of expression changes.

### Tier 2 — Differentially Expressed Gene Accuracy
- **Directional Accuracy** ↑ — Proportion of DE genes with correct up/down direction.
- **Pearson Delta Top-K** ↑ — Correlation of top-K DE gene expression changes.
- **Jaccard DE Top-K** ↑ — Overlap of predicted vs observed top-K DE gene sets.

### Tier 3 — Distributional Fidelity
- **Energy Distance** ↓ — Statistical distance between predicted and observed cell distributions.
- **MMD (RBF kernel)** ↓ — Kernel-based distributional divergence.

## Repository Structure

```
├── README.md
├── Eval_.ipynb                  # Original monolithic notebook (reference)
└── eval/
    ├── __init__.py
    ├── __main__.py              # python -m eval
    ├── config.py                # All configuration parameters
    ├── dataset.py               # .h5ad loading and preprocessing
    ├── sampling.py              # Centralized stratified subsampling
    ├── metrics.py               # Unified T1/T2/T3 metric functions
    ├── eval_runner.py           # Main orchestrator
    ├── collect_results.py       # Result aggregation and display
    ├── models/
    │   ├── __init__.py          # Model registry
    │   ├── gears.py             # GEARS evaluator
    │   ├── scgpt.py             # scGPT evaluator
    │   ├── state.py             # STATE evaluator
    │   ├── cell2sentence.py     # Cell2Sentence evaluator
    │   └── cpa.py               # CPA evaluator
    ├── notebooks/
    │   └── tutorial.ipynb       # Step-by-step tutorial
    └── results/                 # Output directory (created at runtime)
```

## Quick Start

### 1. Install base dependencies

```bash
pip install anndata scanpy scipy pandas matplotlib torch
```

### 2. Set your dataset path

Edit `eval/config.py`:

```python
DATA_PATH = "/path/to/your/dataset.h5ad"
DEVICE = "cuda"   # or "cpu"
```

### 3. Run evaluation

**All models:**
```bash
python -m eval --data /path/to/dataset.h5ad --models all
```

**Specific models:**
```bash
python -m eval --data K562.h5ad --models gears scgpt state
```

**With custom settings:**
```bash
python -m eval --data K562.h5ad --models gears --device cpu --seed 42 --output my_results/
```

### 4. Collect results

```bash
python -m eval.collect_results --results-dir results/
```

## Tutorial Notebook

For a guided walkthrough, open `eval/notebooks/tutorial.ipynb` in Jupyter or Google Colab.

The tutorial covers:
1. Installing dependencies
2. Configuring dataset path and device
3. Loading and subsampling data
4. Running individual model evaluations
5. Collecting and comparing results
6. Interpreting the 3-tier metrics

## Fair Comparison Guarantees

The framework ensures fair comparison across all models through:

- **Identical sampling**: All models evaluate on the exact same subsampled cells via centralized `sampling.py` with a fixed random seed.
- **Identical preprocessing**: Raw counts are preserved and each model applies its own normalization consistently.
- **Identical metrics**: All models are scored with the same metric implementations from `metrics.py`.
- **Reproducibility**: Fixed random seeds at every stochastic step.

## Using Your Own Dataset

The only requirement is an `.h5ad` file with:
- **`adata.X`**: Expression matrix (raw counts or log-normalized)
- **`adata.obs['gene']`**: Perturbation labels (column name configurable via `PERT_COL`)
- A control label value (default: `"non-targeting"`, configurable via `CTRL_LABEL`)

Change `config.DATA_PATH` and run — no other code changes needed.

## Programmatic API

```python
from eval.eval_runner import run

results = run(
    data_path="my_data.h5ad",
    models=["gears", "scgpt"],
    device="cuda",
    seed=42,
    output_dir="my_results/",
)

for r in results:
    print(f"{r['model']}: CA={r['metrics']['T1_Centroid_Accuracy']:.4f}")
```

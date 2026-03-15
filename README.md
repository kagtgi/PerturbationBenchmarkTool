# Perturbation Prediction Benchmark Tool

A modular evaluation framework for benchmarking single-cell gene perturbation prediction models on Perturb-seq datasets (`.h5ad` format).

Refactored from a monolithic notebook into a clean, extensible toolkit with centralized configuration, fair sampling, unified metrics, and per-model evaluation modules.

## Supported Models

| Model | Type | GPU | Reference |
|-------|------|-----|-----------|
| **GEARS** | Graph Auto-Regressive | Yes | [Roohani et al. 2023](https://github.com/snap-stanford/GEARS) |
| **scGPT** | Pre-trained Transformer | Yes | [Cui et al. 2024](https://github.com/bowang-lab/scGPT) |
| **STATE** | Zero-shot HVG Transfer | Yes | [Arc Institute](https://huggingface.co/arcinstitute/ST-HVG-Replogle) |
| **Cell2Sentence** | LLM (Gemma-2-2B, 4-bit) | Yes (CUDA) | [Van Dijk Lab](https://github.com/vandijklab/cell2sentence) |
| **CPA** | Conditional Perturbation VAE | Optional | [Lotfollahi et al. 2023](https://github.com/theislab/cpa) |

## Evaluation Metrics (3 Tiers)

### Tier 1 — Whole-Profile Identity
- **Centroid Accuracy (CA)** ↑ — Is the predicted profile closest to the correct perturbation?
- **Profile Distance Score (PDS)** ↓ — Relative distance ranking among all perturbations.
- **Systema Pearson Delta** ↑ — Genome-wide correlation of expression changes vs. control.

### Tier 2 — Differentially Expressed Gene Accuracy
- **Directional Accuracy** ↑ — Proportion of DE genes with correct up/down direction.
- **Pearson Delta Top-K** ↑ — Correlation of top-K DE gene expression changes.
- **Jaccard DE Top-K** ↑ — Overlap of predicted vs observed top-K DE gene sets.

### Tier 3 — Distributional Fidelity
- **Energy Distance** ↓ — Statistical distance between predicted and observed cell distributions.
- **MMD (RBF kernel)** ↓ — Kernel-based distributional divergence with median bandwidth heuristic.

## Repository Structure

```
├── README.md
├── log.md                          # Development changelog
├── Eval_.ipynb                     # Original monolithic notebook (reference)
└── eval/
    ├── __init__.py
    ├── __main__.py                 # python -m eval
    ├── config.py                   # All configuration parameters
    ├── dataset.py                  # .h5ad loading and preprocessing
    ├── sampling.py                 # Centralized stratified subsampling
    ├── metrics.py                  # Unified T1/T2/T3 metric functions
    ├── eval_runner.py              # Main orchestrator (CLI + API)
    ├── collect_results.py          # Result aggregation and display
    ├── models/
    │   ├── __init__.py             # Model registry with lazy imports
    │   ├── gears.py                # GEARS evaluator
    │   ├── scgpt.py                # scGPT evaluator
    │   ├── state.py                # STATE evaluator
    │   ├── cell2sentence.py        # Cell2Sentence evaluator
    │   └── cpa.py                  # CPA evaluator
    ├── notebooks/
    │   └── tutorial.ipynb          # Step-by-step tutorial
    └── results/                    # Output directory (created at runtime)
```

## Quick Start

### 1. Install base dependencies

```bash
pip install anndata scanpy scipy pandas matplotlib torch
```

Each model installs its own specific dependencies automatically on first run.

### 2. Run evaluation

**All models:**
```bash
python -m eval --data K562.h5ad --models all
```

**Specific models:**
```bash
python -m eval --data K562.h5ad --models gears scgpt state
```

**With custom settings:**
```bash
python -m eval --data K562.h5ad --models gears --device cpu --seed 42 --output my_results/
```

### 3. Collect results

```bash
python -m eval.collect_results --results-dir results/
```

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

## Tutorial Notebook

For a guided walkthrough, open `eval/notebooks/tutorial.ipynb` in Jupyter or Google Colab.

The tutorial covers:
1. Environment setup and data download
2. Configuring dataset path and parameters
3. Loading and subsampling data
4. Running individual model evaluations (with Colab restart guidance)
5. Running all models via the orchestrator
6. Collecting and comparing results
7. Interpreting the 3-tier metrics

## Fair Comparison Guarantees

The framework ensures fair comparison across all models:

- **Identical sampling**: All models evaluate on the exact same subsampled cells via centralized `sampling.py` with a fixed seed (`numpy.random.default_rng(42)`).
- **Identical preprocessing**: Raw counts are preserved; each model applies its own normalization to a fresh `.copy()` of the data.
- **Identical metrics**: All models are scored with the same metric implementations from `metrics.py`.
- **Reproducibility**: Fixed random seeds at every stochastic step.

## Using Your Own Dataset

The only requirement is an `.h5ad` file with:
- **`adata.X`**: Expression matrix (raw counts preferred; log-normalized also accepted)
- **`adata.obs['gene']`**: Perturbation labels (column name configurable via `PERT_COL`)
- A control label value (default: `"non-targeting"`, configurable via `CTRL_LABEL`)

Update `config.py` or pass parameters at runtime — no other code changes needed.

## Known Limitations

- **CPA** computes metrics in log1p-normalized space on non-zero-variance genes, so its metric values are not directly comparable to other models.
- **GEARS** produces point predictions (single centroid per perturbation), which limits T3 distributional metrics.
- Models with different gene vocabularies (GEARS uses Norman genes, scGPT uses its own vocab) evaluate on their respective gene overlaps.
- Running all models sequentially may require significant GPU memory; Colab users should restart between models.

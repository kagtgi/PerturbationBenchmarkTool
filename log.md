# Development Log

## v1.0 — Initial Release

### Phase 1: Refactoring (Commit: `0ad25b8`)

Refactored a monolithic 2776-line Jupyter notebook (`Eval_.ipynb`) into a clean, modular evaluation toolkit.

**Created 16 new files:**

- `eval/__init__.py` — Package init
- `eval/__main__.py` — CLI entry point (`python -m eval`)
- `eval/config.py` — Centralized configuration (data path, seed, device, sampling params, metric params)
- `eval/dataset.py` — `.h5ad` loading, raw count extraction, log-normalization, dense conversion utilities
- `eval/sampling.py` — Stratified subsampling (`numpy.default_rng(42)`, 20% per perturbation group, min 5 cells)
- `eval/metrics.py` — Unified 3-tier metric functions (8 metrics total)
- `eval/eval_runner.py` — Main orchestrator with CLI and programmatic API
- `eval/collect_results.py` — Result aggregation from JSON files into comparison tables
- `eval/models/__init__.py` — Model registry with lazy imports to avoid dependency conflicts
- `eval/models/gears.py` — GEARS evaluator (Norman pretrained, zero-shot transfer)
- `eval/models/scgpt.py` — scGPT evaluator (transformer, torchtext stub, batched inference)
- `eval/models/state.py` — STATE evaluator (CLI-based, HuggingFace model, 2000 HVG space)
- `eval/models/cell2sentence.py` — Cell2Sentence evaluator (Gemma-2-2B, 4-bit quantized, power-law reconstruction)
- `eval/models/cpa.py` — CPA evaluator (VAE, extensive dependency patching, inline metrics)
- `eval/notebooks/tutorial.ipynb` — Step-by-step tutorial notebook
- `eval/results/.gitkeep` — Placeholder for output directory

**Design decisions:**
- Lazy imports in model registry prevent dependency conflicts between models
- Each model gets a `.copy()` of the subsampled AnnData to avoid cross-contamination
- scGPT computes T3 metrics inline during inference to manage memory
- CPA retains inline metric computation due to its different gene space (log1p-normalized, non-zero-variance)
- GEARS uses `seed=1` for Norman data split to match the pretrained model's expected configuration

### Phase 2: Correctness Audit & Bug Fixes (Commit: `13901a5`)

Reviewed all metric implementations and model evaluation code against the original notebook.

**Bugs found and fixed:**

1. **`metrics.py` — `mmd_rbf` diagonal zeros in sigma (HIGH)**
   - `d_tt.pow(2).median()` included N zero diagonal entries, biasing the RBF bandwidth downward
   - Fixed: Added off-diagonal mask before computing median

2. **`cell2sentence.py` — Wrong reconstruction indexing (HIGH)**
   - Used `adata.X[row.Index]` which fails when AnnData index is string-based
   - Fixed: Changed to `enumerate()` pattern: `for i, row in enumerate(adata.obs.itertuples())`

3. **`metrics.py` — `energy_distance` diagonal in pairwise means (MEDIUM)**
   - Full mean including diagonal (self-distance=0) differs from the correct off-diagonal approach
   - Fixed: Compute off-diagonal means using eye mask

4. **`gears.py` — Wrong seed for Norman data split (MEDIUM)**
   - Used unified `seed=42` but original notebook and pretrained model expect `seed=1`
   - Fixed: Hardcoded `seed=1` with explanatory comment

5. **Missing `warnings.filterwarnings("ignore")` in model modules (LOW)**
   - Original notebook cells all suppressed warnings; some model modules were missing this
   - Fixed: Added to all `run_eval()` functions

6. **`results/` directory not tracked in git (LOW)**
   - Added `eval/results/.gitkeep`

### Phase 3: Finalization (v1.0)

- Finalized tutorial notebook with data download step, Colab-specific guidance, metric interpretation, and CLI reference
- Polished README with known limitations, fair comparison guarantees, and complete API documentation
- Created this `log.md` for tracking development history

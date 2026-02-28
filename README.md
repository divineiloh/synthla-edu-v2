# SYNTHLA-EDU V2 — Synthetic Educational Data Benchmark

<div align="center">

**Privacy-aware benchmark for synthetic educational data generation with multi-seed reproducibility and publication-quality visualizations.**

![CI Tests](https://github.com/divineiloh/synthla-edu-v2/workflows/CI%20Tests/badge.svg)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

[Quick Start](#quick-start) • [Visualizations](#publication-visualizations) • [Metrics](#key-metrics) • [Multi-Seed Runs](#multi-seed-runs)

</div>

---

## Overview

Single-file Python benchmark for evaluating synthetic data generators on educational datasets:

- **Datasets**: OULAD (student-level) and ASSISTments 2012–2013 (student-level aggregated)
- **Synthesizers**: Gaussian Copula (SDV), CTGAN (SDV), TabDDPM (Synthcity)
- **Evaluation**: Quality (SDMetrics), Utility (TSTR/TRTR), Realism (C2ST), Privacy (Multi-attacker MIA), Statistical rigor (Paired permutation tests with Bonferroni correction)
- **Visualizations**: 16 publication-quality figures at 1 200 DPI (fig2–fig17), including SHAP beeswarm plots

## Quick Start

### 1. Install

```bash
git clone https://github.com/divineiloh/synthla-edu-v2.git
cd synthla-edu-v2

# Recommended: locked versions for exact reproducibility
pip install -r requirements-locked.txt

# Alternative: minimum versions (may differ from paper)
# pip install -r requirements.txt
```

### 2. Download Data

Download datasets and place in `data/raw/`:

- **OULAD**: [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset) → `data/raw/oulad/`
- **ASSISTments 2012–2013**: [sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) → `data/raw/assistments/`

### 3. Run

> **⚠️ Quick Mode vs Full Mode**
>
> `--quick` is for smoke testing only. Metrics are NOT representative (under-trained models).
> - CTGAN: 100 epochs (vs 300 full) · TabDDPM: 300 iterations (vs 1 200 full) · Bootstrap: 100 resamples (vs 1 000 full)
>
> **Always run WITHOUT `--quick` for evaluation or publication.**

```bash
# Full matrix (2 datasets × 3 synthesizers)
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs --seed 42

# Quick smoke test
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs --quick
```

**Docker** (optional):

```bash
docker build -t synthla-edu-v2 .
docker run -v $(pwd)/data/raw:/app/data/raw -v $(pwd)/runs:/app/runs \
  synthla-edu-v2 python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs
```

See [DOCKER.md](DOCKER.md) for GPU support and docker-compose.

### 4. View Results

```bash
cat runs/oulad/results.json
cat runs/assistments/results.json
ls runs/figures/              # Cross-dataset comparison figures
```

## Multi-Seed Runs

For publication-quality results with error bars across seeds:

```bash
python synthla_edu_v2.py \
  --run-all --raw-dir data/raw --out-dir runs_publication \
  --seeds 0,1,2,3,4
```

This creates `runs_publication/seed_0/` through `seed_4/`, each containing per-dataset `results.json`. The aggregated figure script then produces mean ± std plots across all seeds.

### Regenerate Figures

After multi-seed runs complete, regenerate all 16 publication figures:

```bash
python scripts/generate_all_figures.py                  # all 16 figures
python scripts/generate_all_figures.py --skip-beeswarm  # skip fig14/fig15 (faster)
```

Output: `runs_publication/figures_aggregated/fig2.png` – `fig17.png`

## Publication Visualizations

The figure pipeline produces **16 publication-quality figures** (1 200 DPI, color-blind friendly) numbered fig2–fig17:

| Figure | Description |
|--------|-------------|
| fig2 | OULAD classification utility (TSTR bar chart) |
| fig3 | ASSISTments classification utility |
| fig4 | OULAD regression utility (TSTR bar chart) |
| fig5 | ASSISTments regression utility |
| fig6 | SDMetrics quality scores (grouped bar) |
| fig7 | MIA worst-case effective AUC (grouped bar) |
| fig8 | OULAD MIA per-attacker breakdown |
| fig9 | ASSISTments MIA per-attacker breakdown |
| fig10 | OULAD SHAP feature importance – Classification (top 7) |
| fig11 | OULAD SHAP feature importance – Regression (top 7) |
| fig12 | ASSISTments SHAP feature importance – Classification |
| fig13 | ASSISTments SHAP feature importance – Regression |
| fig14 | SHAP beeswarm – OULAD Classification TRTR (top 10 features) |
| fig15 | SHAP beeswarm – OULAD Classification TSTR TabDDPM (top 10 features) |
| fig16 | OULAD multi-objective performance heatmap |
| fig17 | ASSISTments multi-objective performance heatmap |

### Quality Standards
- ✅ 1 200 DPI resolution (IEEE print-ready)
- ✅ Color-blind friendly palette
- ✅ All text legible at single-column width
- ✅ SHAP beeswarm feature names at 24 pt
- ✅ Professional styling with value annotations

## Outputs

Each dataset in `runs/<dataset>/` contains:
- **`results.json`** — Comprehensive metrics for all 3 synthesizers:
  - `synthesizers.<name>`: `{sdmetrics, c2st, mia, utility, timing}` with bootstrap CIs
  - `pairwise_tests`: Statistical significance tests

Cross-dataset visualizations in `runs/figures/` (single-seed) or `runs_publication/figures_aggregated/` (multi-seed).

## Key Metrics

> **C2ST**: 0.5 = ideal (indistinguishable), 1.0 = worst (lower is better)
> **MIA**: 0.5 = ideal (no leakage), 1.0 = worst (lower is better)

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Quality** (SDMetrics) | 0–1 | Statistical similarity to real data (higher = better) |
| **Utility** (TSTR/TRTR AUC) | 0–1 | Downstream predictive performance (higher = better) |
| **Realism** (C2ST effective AUC) | 0.5–1.0 | Detectability (lower = better) |
| **Privacy** (MIA worst-case) | 0.5–1.0 | Membership inference risk (lower = better) |
| **Bootstrap CI** | 95% | 1 000 resamples per metric |
| **Permutation p-value** | [0, 1] | Pairwise significance (Bonferroni-adjusted α = 0.0083) |

## Statistical Methods

### Multiple Testing Correction

Bonferroni correction across all pairwise comparisons:
- 6 tests (3 pairs × 2 metrics) · α = 0.05/6 = **0.0083**
- Paired permutation tests (10 000 permutations)

### Effect Sizes

Cohen's d: |d| < 0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · ≥ 0.8 large

### Evaluation Design

- **C2ST** excludes ID + target columns (measures distribution fidelity on features only)
- **MIA** excludes ID columns only, targets included (worst-case adversary)

### Hyperparameters

| Component | Standard | Quick |
|-----------|----------|-------|
| CTGAN | 300 epochs | 100 epochs |
| TabDDPM | 1 200 iterations | 300 iterations |
| Bootstrap | 1 000 resamples | 100 resamples |
| Random Forest | 300 trees | 300 trees |

## Repository Structure

```
synthla-edu-v2/
├── synthla_edu_v2.py            # Main pipeline (~2 150 lines)
├── scripts/
│   └── generate_all_figures.py  # Publication figure generator (fig2–fig17)
├── tests/                       # pytest test suite
│   ├── conftest.py
│   ├── test_pipeline.py
│   ├── test_data_loading.py
│   └── test_synthesizers.py
├── utils/                       # Optional utility modules
│   ├── timing.py
│   ├── effect_size.py
│   └── logging_config.py
├── data/
│   └── raw/                     # Place datasets here (gitignored)
│       ├── oulad/               # OULAD CSV files (7 tables)
│       └── assistments/         # ASSISTments CSV
├── runs_publication/            # Multi-seed experiment results
│   ├── seed_0/ – seed_4/       # Per-seed results
│   ├── figures_aggregated/      # 16 publication figures (1 200 DPI)
│   ├── seed_summary.json
│   └── seed_summary.csv
├── Dockerfile                   # Reproducible container
├── requirements.txt             # Minimum dependencies
├── requirements-locked.txt      # Pinned versions (exact reproducibility)
├── DATA_SCHEMA.md               # OULAD & ASSISTments data schemas
├── DOCKER.md                    # Docker usage guide
├── QUICKSTART.md                # Quick-start guide
├── pytest.ini                   # Test configuration
├── VERSION                      # Semantic version
└── LICENSE                      # MIT License
```

## Dependencies

**Core:** Python 3.11+ · PyTorch 2.x · pandas · numpy · scikit-learn · SDV · SDMetrics · Synthcity · matplotlib · SHAP

**Optional:** xgboost (multi-attacker MIA)

See [requirements-locked.txt](requirements-locked.txt) for exact pinned versions.

## Advanced Usage

```
python synthla_edu_v2.py --help

  --dataset {oulad,assistments}  Single dataset
  --raw-dir PATH                 Raw CSV folder
  --out-dir PATH                 Output directory
  --seed INT                     Random seed (default: 0)
  --seeds STR                    Comma-separated seeds for multi-seed runs
  --synthesizer {gaussian_copula,ctgan,tabddpm}
  --run-all                      Full 2×3 matrix
  --quick                        Reduced compute (smoke test only)
  --test-size FLOAT              Test split (default: 0.3)
```

### Extending the Code

**Add a synthesizer:** implement `fit(df)` + `sample(n)` → register in `run_single()`/`run_all()`

**Add a dataset:** implement `build_<name>_table()` returning `(df, schema)` → register in `build_dataset()`

## Known Limitations

- **Compute**: Full `--run-all` takes 3–6 hours on CPU. Use `--quick` for validation.
- **TabDDPM preprocessing**: Clips numeric outliers (0.5th–99.5th percentile) before training.
- **XGBoost**: Optional dependency; install separately for multi-attacker MIA.

## Citation

```bibtex
@software{synthla_edu_v2,
  title  = {SYNTHLA-EDU V2: Synthetic Educational Data Benchmark},
  author = {Divine Iloh},
  year   = {2025},
  url    = {https://github.com/divineiloh/synthla-edu-v2}
}
```

## License

[MIT License](LICENSE) — Copyright © 2025 Divine Iloh

## References

- OULAD: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open-dataset)
- ASSISTments: [ASSISTments 2012–2013 Dataset](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect)
- SDV: [Synthetic Data Vault](https://github.com/sdv-dev/SDV)
- SHAP: [SHapley Additive exPlanations](https://github.com/shap/shap)

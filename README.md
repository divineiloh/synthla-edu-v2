# SYNTHLA-EDU V2 — Synthetic Educational Data Benchmark

<div align="center">

**Benchmark for synthetic educational data generation with comprehensive privacy-aware evaluation and publication-quality visualizations.**

![CI Tests](https://github.com/divineiloh/synthla-edu-v2/workflows/CI%20Tests/badge.svg)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

[Quick Start](#quick-start) • [Features](#features) • [Visualizations](#publication-visualizations) • [Metrics](#key-metrics)

</div>

---

## Overview

Single-file Python benchmark for evaluating synthetic data generators on educational datasets:

- **Datasets**: OULAD (student-level) and ASSISTments (student-level aggregated)
- **Synthesizers**: Gaussian Copula (SDV), CTGAN (SDV), TabDDPM (Synthcity)
- **Evaluation**: Quality (SDMetrics), Utility (TSTR/TRTR), Realism (C2ST), Privacy (Multi-attacker MIA), Statistical rigor (Paired permutation tests)
- **Visualizations**: 12 publication-quality figures (300 DPI, color-blind friendly)

## Quick Start

### 1. Install

```bash
git clone https://github.com/divineiloh/synthla-edu-v2.git
cd synthla-edu-v2

# Recommended: Use locked versions for exact reproducibility
pip install -r requirements-locked.txt

# Alternative: Minimum versions (may differ from paper)
# pip install -r requirements.txt
```

### 2. Download Data

Download datasets and place in `data/raw/`:

- **OULAD**: [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset) → `data/raw/oulad/`
- **ASSISTments**: [assistments.org](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) → `data/raw/assistments/`

### 3. Run

> **⚠️ IMPORTANT: Quick Mode vs Full Mode**
> 
> **`--quick` mode is for smoke testing and pipeline validation only.** Metrics from quick mode are NOT representative and may show very poor realism (often C2ST close to 1.0). Note that some model+dataset combinations can also yield poor realism in full mode; do not assume `--quick` is the only reason.
> - CTGAN: 100 epochs (vs 300 full)
> - TabDDPM: 300 iterations (vs 1200 full)
> - Bootstrap: 100 resamples (vs 1000 full)
>
> **Always run WITHOUT `--quick` for evaluation or publication.**

**Option A: Native Python**

Full experimental matrix (2 datasets × 3 synthesizers):
```bash
# Consolidated outputs: data.parquet + results.json per dataset
python synthla_edu_v2.py \
  --run-all \
  --raw-dir data/raw \
  --out-dir runs
```

**Quick mode (smoke testing only - NOT for evaluation):**
```bash
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs --quick
```

**Option B: Docker (optional, fully reproducible environment)**

```bash
# Build image
docker build -t synthla-edu-v2 .

# Run with mounted data/output directories
docker run -v $(pwd)/data/raw:/app/data/raw \
           -v $(pwd)/runs:/app/runs \
           synthla-edu-v2 \
           python synthla_edu_v2.py --run-all --quick --raw-dir data/raw --out-dir runs
```

See [DOCKER.md](DOCKER.md) for detailed Docker instructions, GPU support, and docker-compose setup.

**Single synthesizer (legacy, per-synth outputs):**
```bash
# Gaussian Copula on OULAD
python synthla_edu_v2.py \
  --dataset oulad \
  --raw-dir data/raw/oulad \
  --out-dir runs/oulad_gc \
  --synthesizer gaussian_copula

# CTGAN on ASSISTments
python synthla_edu_v2.py \
  --dataset assistments \
  --raw-dir data/raw/assistments \
  --out-dir runs/assistments_ctgan \
  --synthesizer ctgan

# TabDDPM on OULAD
python synthla_edu_v2.py \
  --dataset oulad \
  --raw-dir data/raw/oulad \
  --out-dir runs/oulad_tabddpm \
  --synthesizer tabddpm
```

### 4. View Results

**Consolidated outputs per dataset:**
```bash
# View per-dataset metrics
cat runs/oulad/results.json
cat runs/assistments/results.json

# View 12 cross-dataset comparison visualizations
ls runs/figures/fig*_*.png
```

## Publication Visualizations

SYNTHLA-EDU V2 automatically generates **12 gold-standard cross-dataset comparison figures** (300 DPI, color-blind friendly) after both datasets complete:

### Cross-Dataset Comparisons (10 figures)
1. **Classification Utility** - OULAD vs ASSISTments side-by-side comparison
2. **Regression Utility** - Cross-dataset MAE comparison
3. **Data Quality** - SDMetrics scores across both datasets
4. **Privacy (MIA)** - Cross-dataset privacy preservation
5. **Performance Heatmap** - All metrics × datasets × synthesizers grid
6. **Radar Chart** - Multi-dimensional profiles for both datasets
7. **Classification CI** - Confidence intervals for both datasets
8. **Regression CI** - MAE confidence intervals comparison
9. **Computational Efficiency** - Training/sampling time comparison
10. **Per-Attacker Privacy** - Detailed MIA breakdown across datasets

### Dataset-Specific Deep Dives (2 figures)
11. **Distribution Fidelity** - Feature distributions for both datasets
12. **Correlation Matrices** - Real vs synthetic feature relationships

**All 12 figures** are automatically generated in `runs/figures/` when running `--run-all`.

### Quality Standards
- ✅ 300 DPI resolution (print-ready)
- ✅ Color-blind friendly palette
- ✅ All text legible (10-14pt fonts)
- ✅ No overlapping elements
- ✅ Professional styling with value annotations

## Outputs

Each dataset in `runs/<dataset>/` contains:
- **`data.parquet`** — Consolidated data with columns: `[..., split: {real_train, real_test, synthetic_train}, synthesizer: {real, gaussian_copula, ctgan, tabddpm}]`
- **`results.json`** — Comprehensive metrics for all 3 synthesizers:
  - `synthesizers.<name>`: `{sdmetrics, c2st, mia, utility, timing}` with bootstrap CIs
    - `utility.per_sample`: Individual errors for all test samples (nested under utility)
  - `pairwise_tests`: Statistical significance tests

Cross-dataset visualizations in `runs/figures/`:
- **`fig1-fig12.png`** — 12 publication-ready cross-dataset comparison figures

**Output size**: 
- Per dataset: ~10-15MB (data + results)
- Figures: ~5MB (12 high-resolution PNGs)

## Key Metrics

> **Note on C2ST and MIA Interpretation:**
> 
> **C2ST (Classifier Two-Sample Test)** measures how easily a classifier can distinguish synthetic from real data:
> - `effective_auc = max(auc, 1-auc)` ensures the metric is in [0.5, 1.0]
> - **0.5 = ideal** (indistinguishable; classifier performs at chance)
> - **1.0 = worst** (perfectly distinguishable; synthetic is easily detected)
> - **Lower is better** for realism
>
> **MIA (Membership Inference Attack)** measures privacy leakage:
> - **0.5 = ideal** (no leakage; attacker performs at chance)
> - **1.0 = worst** (total leakage; training records perfectly identified)
> - **Lower is better** for privacy

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Quality** (SDMetrics overall_score) | 0–1 (0–100% in figures) | Statistical similarity to real data (stored as 0–1; displayed as %; higher = better) |
| **Utility** (TSTR/TRTR AUC) | 0–1 | Predictive performance on downstream tasks (higher = better) |
| **Realism** (C2ST effective_auc) | 0.5–1.0 | Detectability by classifier: 0.5 = indistinguishable, 1.0 = fully distinguishable (lower = better) |
| **Privacy** (MIA worst_case_effective_auc) | 0.5–1.0 | Membership inference risk: 0.5 = no leakage, 1.0 = total leakage (lower = better) |
| **Bootstrap CI** | 95% | Confidence intervals from 1,000 resamples per metric |
| **Permutation test p-value** | [0, 1] | Significance of pairwise model differences (lower = significant) |

### Example Results

**⚠️ DISCLAIMER**: Results below are **illustrative examples** showing the output format. Actual values vary by dataset, synthesizer, and training mode.

**Quick Mode (--quick flag):**
- Quality: 60-70% (SDMetrics overall)
- Utility: 0.70-0.85 (RF/LR AUC on real data)
- **Realism: 0.95-1.0 (POOR - expected in quick mode due to undertrained models)**
- Privacy: 0.50-0.55 (good)

**Full Mode (no --quick flag):**
- Quality: 65-85% (SDMetrics overall)
- Utility: 0.75-0.90 (RF/LR AUC on real data)
- **Realism: 0.60-0.95 (varies by synthesizer and dataset complexity)**
- Privacy: 0.50-0.60 (good to moderate)

**Interpretation:**
- Realism (C2ST effective_auc): **0.5 = perfect**, 1.0 = fully distinguishable
- Privacy (MIA): **0.5 = no leakage**, 1.0 = total leakage
- Both metrics: **lower is better**

## Project Structure

```
synthla-edu-v2/
├── synthla_edu_v2.py         # Single-file runner (all-in-one)
├── requirements.txt         # Dependencies
├── requirements-locked.txt  # Pinned versions
├── README.md                # This file
├── data/
│   └── raw/                 # Place datasets here
│       ├── oulad/           # OULAD CSVs
│       └── assistments/     # ASSISTments CSV
└── runs/                    # Output directory (auto-created)
```

## How It Works

1. **Load & Preprocess**: Build student-level (OULAD) or student-level aggregated (ASSISTments) tables
2. **Split**: Train/test split (30% test) with group awareness (ASSISTments) or stratification (OULAD)
3. **Synthesize**: Fit Gaussian Copula on training data, sample synthetic records
4. **Evaluate**:
   - **Quality**: SDMetrics column shape + pair trends
   - **Realism**: C2ST with Random Forest classifier (single seed)
   - **Privacy**: MIA with KNN distance features + multi-attacker (LR, RF, XGBoost) worst-case

## Dependencies

**Core:**
- Python 3.11+
- PyTorch 2.9.1+ (TabDDPM with RMSNorm support)
- pandas>=2.0, numpy>=1.24, scikit-learn>=1.3
- SDV>=1.0 (Gaussian Copula, CTGAN)
- SDMetrics>=0.14 (Quality reports)
- Synthcity>=0.2.11 (TabDDPM diffusion model)
- matplotlib>=3.7 (Visualizations)

**Optional:**
- xgboost (Multi-attacker MIA)

See [requirements.txt](requirements.txt) for full pinned versions.

## Repository Structure

```
SYNTHLA-EDU V2/
├── synthla_edu_v2.py          # Main pipeline (single file, ~2150 lines)
├── requirements.txt            # Python dependencies
├── requirements-locked.txt     # Pinned versions
├── README.md                   # This file
├── data/
│   └── raw/
│       ├── oulad/             # OULAD CSV files
│       │   ├── studentInfo.csv
│       │   ├── studentAssessment.csv
│       │   ├── studentVle.csv
│       │   ├── vle.csv
│       │   ├── assessments.csv
│       │   ├── courses.csv
│       │   └── studentRegistration.csv
│       └── assistments/        # ASSISTments CSV file
│           └── assistments_2009_2010.csv
└── runs/
    ├── oulad/                  # OULAD results
    │   ├── data.parquet        # Consolidated data (real + 3 synthetic)
    │   └── results.json        # All metrics + statistical tests
    ├── assistments/            # ASSISTments results
    │   ├── data.parquet
    │   └── results.json
    └── figures/                # Cross-dataset visualizations
        └── fig1-12.png         # 12 publication-quality comparison figures
```

**Total**: 1 Python file + 2 dataset result directories + 1 figures directory
- `xgboost` (Multi-attacker MIA; enhances privacy evaluation)

See [requirements.txt](requirements.txt) for full pinned list.

## Advanced Usage

### Command-Line Options

```bash
python synthla_edu_v2.py --help

options:
  --dataset {oulad,assistments}       Dataset to run
  --raw-dir PATH                      Path to raw CSV folder
  --out-dir PATH                      Output directory
  --test-size FLOAT                   Test split (default: 0.3)
  --seed INT                          Random seed (default: 0)
  --synthesizer {gaussian_copula,ctgan,tabddpm}  Model (default: gaussian_copula)
  --run-all                           Run full 2×3 matrix; writes consolidated outputs
  --quick                             Reduce compute (fewer CTGAN epochs, TabDDPM iterations)
  --aggregate-assistments             Aggregate ASSISTments to student-level (if not already)
  --compare DIR                       Generate model comparison chart from results.json in DIR
```

### Extending the Code

The single-file `synthla_edu_v2.py` is modular:

**Core Functions:**
- **Dataset builders**: `build_oulad_student_table()`, `build_assistments_table()`
- **Splitting**: `group_train_test_split()`, `simple_train_test_split()`
- **Synthesizers**: `GaussianCopulaSynth`, `CTGANSynth`, `TabDDPMSynth` classes
- **Evaluations**: 
  - `sdmetrics_quality()` — Quality report
  - `tstr_utility()` — TSTR/TRTR with bootstrap CIs
  - `c2st_effective_auc()` — Realism (real_test vs synthetic_train)
  - `mia_worst_case_effective_auc()` — Multi-attacker MIA (LR, RF, XGB)
  - `bootstrap_ci()` — 95% CI from 1,000 resamples
  - `paired_permutation_test()` — Pairwise statistical testing (2,000 permutations)

**To add a new synthesizer:**
1. Implement a class with `fit(df)` and `sample(n)` methods
2. Update `run_single()` and `run_all()` to instantiate it
3. Reference via `--synthesizer <name>`

**To add a new dataset:**
1. Implement `build_<dataset>_table()` returning `(df, schema)` with `target_cols`
2. Add to `build_dataset()` dispatcher
3. Define targets in `run_all()`: set `class_target, reg_target`
4. Run with `--dataset <name>`

## Known Limitations & Future Work

- **TabDDPM CSV parsing**: Occasional issues on large OULAD splits; use `engine='python'` fallback if needed
- **TabDDPM stability preprocessing**: The TabDDPM path fills missing values and clips numeric outliers (0.5th–99.5th percentiles) before training to reduce NaNs/instability; this can slightly alter marginal distributions.
- **Compute**: Runtime varies by hardware; CTGAN/TabDDPM benefit from GPU acceleration. Use `--quick` for faster validation runs.
- **XGBoost MIA attacker**: Optional; requires `pip install xgboost` for multi-attacker MIA
- **Extensibility**: Easy to add new datasets, synthesizers, or attackers by following the modular structure
- **Containerization**: Docker support provided via `Dockerfile` and [DOCKER.md](DOCKER.md). Reproducibility baseline uses pinned dependencies (`requirements-locked.txt`) and CI validation.

For original research pipeline, see [synthla-edu](https://github.com/divineiloh/synthla-edu) (full V1 codebase).

## Citation

```bibtex
@software{synthla_edu_v2,
  title = {SYNTHLA-EDU V2: Minimal Synthetic Educational Data Benchmark},
  author = {Divine Iloh},
  year = {2025},
  url = {https://github.com/divineiloh/synthla-edu-v2}
}
```

## License

MIT License

## References

- OULAD: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open-dataset)
- ASSISTments: [ASSISTments 2009-2010 Dataset](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
- SDV: [Synthetic Data Vault](https://github.com/sdv-dev/SDV)
- SDMetrics: [SDV Quality Metrics](https://github.com/sdv-dev/SDMetrics)

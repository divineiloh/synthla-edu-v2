# SYNTHLA-EDU V2 — Synthetic Educational Data Benchmark

<div align="center">

**Minimal single-file benchmark for synthetic educational data generation with privacy-aware evaluation.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

[Quick Start](#quick-start) • [Features](#features) • [Metrics](#key-metrics)

</div>

---

## Overview

Single-file Python benchmark for evaluating synthetic data generators on educational datasets:

- **Datasets**: OULAD (student-level) and ASSISTments (interaction-level)
- **Synthesizer**: Gaussian Copula (SDV)
- **Evaluation**: Quality (SDMetrics), Realism (C2ST), Privacy (MIA)

## Quick Start

### 1. Install

```bash
git clone https://github.com/divineiloh/synthla-edu-v2.git
cd synthla-edu-v2
pip install -r requirements.txt
```

### 2. Download Data

Download datasets and place in `data/raw/`:

- **OULAD**: [analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset) → `data/raw/oulad/`
- **ASSISTments**: [assistments.org](https://www.assistments.org/) → `data/raw/assistments/`

### 3. Run

```bash
# OULAD
python synthla_edu_v2.py \
  --dataset oulad \
  --raw-dir data/raw/oulad \
  --out-dir runs/oulad_runs

# ASSISTments (with student-level aggregation for evaluation)
python synthla_edu_v2.py \
  --dataset assistments \
  --raw-dir data/raw/assistments \
  --out-dir runs/assistments_runs \
  --aggregate-assistments
```

### 4. View Results

**Windows:**
```powershell
Get-Content runs/oulad_runs/oulad/sdmetrics__gaussian_copula.json
Get-Content runs/oulad_runs/oulad/c2st__gaussian_copula.json
Get-Content runs/oulad_runs/oulad/mia__gaussian_copula.json
```

**Linux/Mac:**
```bash
cat runs/oulad_runs/oulad/sdmetrics__gaussian_copula.json
cat runs/oulad_runs/oulad/c2st__gaussian_copula.json
cat runs/oulad_runs/oulad/mia__gaussian_copula.json
```

## Outputs

Each run produces under `runs/<dataset>_runs/<dataset>/`:

- `real_full.parquet` — Complete real dataset
- `real_train.parquet` — Training split
- `real_test.parquet` — Test split
- `synthetic_train__gaussian_copula.parquet` — Synthetic data
- `schema.json` — Dataset metadata (id cols, targets, categoricals)
- `sdmetrics__gaussian_copula.json` — Quality scores
- `c2st__gaussian_copula.json` — Realism (classifier two-sample test)
- `mia__gaussian_copula.json` — Privacy (membership inference attack)

## Key Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **SDMetrics overall_score** | 0-100 | Statistical similarity (higher = better) |
| **C2ST effective_auc_mean** | 0.5-1.0 | Realism (0.5 = indistinguishable, 1.0 = fully distinguishable) |
| **MIA worst_case_effective_auc** | 0.5-1.0 | Privacy (0.5 = no leakage, 1.0 = total leakage) |

### Example Results

**OULAD** (32,593 students):
- Quality: 76.3%
- C2ST: 0.67 (moderate realism)
- MIA: 0.50 (excellent privacy)

**ASSISTments** (1,000 interactions):
- Quality: 81.2%
- C2ST: 0.54 (good realism)
- MIA: 0.52 (excellent privacy)

## Project Structure

```
synthla-edu-v2/
├── synthla_edu.py           # Single-file runner (all-in-one)
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

1. **Load & Preprocess**: Build student-level (OULAD) or interaction-level (ASSISTments) tables
2. **Split**: Train/test split (30% test) with group awareness (ASSISTments) or stratification (OULAD)
3. **Synthesize**: Fit Gaussian Copula on training data, sample synthetic records
4. **Evaluate**:
   - **Quality**: SDMetrics column shape + pair trends
   - **Realism**: C2ST with Random Forest classifier (5 seeds)
   - **Privacy**: MIA with KNN distance features + RF attacker

## Dependencies

Core:
- `pandas>=2.0`
- `numpy>=1.24`
- `scikit-learn>=1.3`
- `sdv>=1.0` (Gaussian Copula synthesizer)
- `sdmetrics>=0.14` (Quality reports)

See [requirements.txt](requirements.txt) for full list.

## Advanced Usage

### Command-Line Options

```bash
python synthla_edu.py --help

options:
  --dataset {oulad,assistments}
  --raw-dir PATH            Path to raw CSV folder
  --out-dir PATH            Output directory
  --test-size FLOAT         Test split fraction (default: 0.3)
  --seed INT                Random seed (default: 0)
  --aggregate-assistments   Aggregate ASSISTments to student-level for evaluation
```

### Extending the Code

The single-file `synthla_edu.py` is modular:

- **Dataset builders**: `build_oulad_student_table()`, `build_assistments_table()`
- **Splitting**: `group_train_test_split()`, `simple_train_test_split()`
- **Synthesizer**: `GaussianCopulaSynth` class (can swap for CTGAN/TabDDPM)
- **Evaluations**: `sdmetrics_quality()`, `c2st_effective_auc()`, `mia_worst_case_effective_auc()`

To add a new dataset:
1. Implement a `build_<dataset>_table()` function returning `(df, schema)`
2. Add to `build_dataset()` dispatcher
3. Run with `--dataset <name>`

## Known Limitations

- Single synthesizer only (Gaussian Copula)
- No TSTR utility evaluation
- No bootstrap CIs or permutation tests
- Designed for quick validation, not full research benchmarks

For advanced features (CTGAN, TabDDPM, statistical testing), see the original [synthla-edu](https://github.com/divineiloh/synthla-edu) repository.

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

- OULAD: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- ASSISTments: [ASSISTments 2009-2010](https://www.assistments.org/)
- SDV: [Synthetic Data Vault](https://github.com/sdv-dev/SDV)
- SDMetrics: [SDV Quality Metrics](https://github.com/sdv-dev/SDMetrics)

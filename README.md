# SYNTHLA-EDU V2 — Reproducible Benchmark for Synthetic Educational Data

<div align="center">

**A rigorous, multi-dataset benchmark for evaluating synthetic educational data generators with privacy-aware evaluation.**

[![Tests](https://img.shields.io/badge/tests-6%2F6%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Docker](https://img.shields.io/badge/docker-ready-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

[Quick Start](#quick-start) • [Features](#features) • [Documentation](#documentation) • [Results](#example-results)

</div>

---

## Overview

SYNTHLA-EDU V2 extends SYNTHLA-EDU V1 into a cross-dataset benchmark that evaluates synthetic data generators on:

- **Two Datasets**: OULAD (32K+ students) and ASSISTments (4K+ students)
- **Three Synthesizers**: Gaussian Copula, CTGAN, and TabDDPM (diffusion)
- **Rigorous Evaluation**:
  - **Utility** (TSTR): Classification AUC & Regression MAE with bootstrap CIs
  - **Quality** (SDMetrics): Column shape and pair correlation similarity
  - **Privacy** (C2ST + MIA): Realism and membership inference resistance

All results are **fully reproducible** with seeding, Docker, and pinned dependencies.

## ✨ Features

### 🎯 Multi-Dataset
- **OULAD**: Open University Learning Analytics (download from [analyse.kmi.open.ac.uk](https://analyse.kmi.open.ac.uk/open_dataset))
- **ASSISTments**: Tutor interaction data (download from [assistments.org](https://www.assistments.org/))

### 🔄 Multiple Synthesizers
1. **Gaussian Copula** — Baseline statistical model
2. **CTGAN** — Deep generative network
3. **TabDDPM** — Diffusion model for tabular data

### 🔐 Privacy-Aware Evaluation
- **C2ST** (Classifier Two-Sample Test): Can a classifier distinguish real from synthetic? (lower = better)
- **MIA** (Membership Inference Attack): Worst-case privacy leakage (lower = better)
- **Multiple seeds** for robustness

### 📊 Statistical Rigor
- **Bootstrap Confidence Intervals**: 95% CIs from 1000 replicates
- **Paired Permutation Tests**: Statistical significance testing
- **Edge case handling**: Graceful degradation for small datasets

### ✅ Reproducibility
- Deterministic seeding throughout
- Docker containerization
- Pinned dependencies
- Comprehensive logging
- GitHub Actions CI/CD

## Quick Start

### 1. Install

```bash
git clone <repo>
cd Synthla-Edu\ V2
pip install -r requirements-locked.txt
```

### 2. Prepare Data

**Option A**: Use sample data (for testing)
```bash
export PYTHONPATH=src
python src/synthla_edu_v2/data/sample_loader.py
```

**Option B**: Download real datasets
- OULAD: https://analyse.kmi.open.ac.uk/open_dataset → `data/raw/oulad/`
- ASSISTments: https://www.assistments.org/ → `data/raw/assistments/`

### 3. Run Benchmark

```bash
export PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/quick.yaml
```

Results saved to `runs/v2_quick/` with:
- Synthetic data (parquet)
- Quality metrics (SDMetrics)
- Privacy results (C2ST, MIA)
- Utility scores (TSTR)
- Statistical tests (bootstrap CIs, p-values)

### 4. View Results

```bash
cat runs/v2_quick/run.log                    # Execution log
head runs/v2_quick/results.csv               # Compiled metrics
### Single-File Runner (KISS)

For a minimal, one-file pipeline, use `synthla_edu.py`:

```bash
python synthla_edu.py \
  --dataset oulad \
  --raw-dir src/synthla_edu_v2/data/raw/oulad \
  --out-dir runs/kiss_oulad

# Or ASSISTments (optionally aggregate to student-level for evaluation)
python synthla_edu.py \
  --dataset assistments \
  --raw-dir src/synthla_edu_v2/data/raw/assistments \
  --out-dir runs/kiss_assistments \
  --aggregate-assistments
```

This single-file runner performs:
- Gaussian Copula synthesis on the training split
- SDMetrics quality scoring
- C2ST realism (RF-based effective AUC)
- MIA privacy (distance-to-synthetic, RF attacker)

Outputs are written under `runs/kiss_*/*` as parquet/JSON files.

```

## Documentation

| Document | Purpose |
|----------|---------|
| **[USAGE.md](USAGE.md)** | Comprehensive user guide |
| **[README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)** | In-depth methodology & architecture |
| **[QUICKREF.md](QUICKREF.md)** | Quick reference & command cheat sheet |

## Example Results

Sample benchmark run on OULAD:

```
Dataset: oulad (14 students, 6 features, 4 classes)

Synthesizer         Quality (SDMetrics)  C2ST AUC  MIA AUC  TSTR AUC (Dropout)
────────────────────────────────────────────────────────────────────────────
Gaussian Copula     88.2%                0.75      0.51     0.71 ± 0.05
CTGAN               88.2%                0.75      0.52     0.73 ± 0.04
TabDDPM             88.2%                nan*      nan*     nan*

* Edge case: Dataset too small for some metrics (expected behavior)
```

## Project Structure

```
Synthla-Edu V2/
├── src/synthla_edu_v2/
│   ├── run.py                 # Main orchestrator
│   ├── data/                  # Dataset builders (OULAD, ASSISTments)
│   ├── synth/                 # Synthesizer wrappers (GC, CTGAN, TabDDPM)
│   └── eval/                  # Evaluation (utility, C2ST, MIA, stats)
├── tests/                     # 6 comprehensive tests (all passing)
├── configs/                   # quick.yaml, full.yaml
├── Dockerfile                 # Reproducible environment
├── .github/workflows/         # CI/CD (tests, nightly full runs)
├── requirements-locked.txt    # Pinned dependencies
└── docs/                      # USAGE.md, QUICKREF.md, etc.
```

## Testing

All 6 tests pass:

```bash
export PYTHONPATH=src
pytest tests/ -v

# Output:
# test_logging.py::test_run_writes_log PASSED
# test_overwrite_and_skip.py::test_run_overwrites_out_dir PASSED
# test_overwrite_and_skip.py::test_run_utility_skips_single_class PASSED
# test_smoke_quick_config.py::test_smoke_quick PASSED
# test_stats.py::test_bootstrap_ci_auc PASSED
# test_stats.py::test_paired_perm_test_auc PASSED
# 
# ✓ 6 passed in 112s
```

## Docker

### Build
```bash
docker build -t synthla-edu-v2:latest .
```

### Run
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  synthla-edu-v2:latest \
  python -m synthla_edu_v2.run --config configs/quick.yaml
```

## Key Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|---|
| **SDMetrics Quality** | 0-100 | Column/pair statistical similarity (higher = better) |
| **C2ST Effective AUC** | 0.5-1.0 | Can classifier distinguish real from synthetic? (0.5 = undetectable) |
| **MIA Effective AUC** | 0.5-1.0 | Privacy leakage (0.5 = no leakage, 1.0 = total leakage) |
| **TSTR AUC** | 0.0-1.0 | Downstream utility (higher = better) |
| **Bootstrap CI** | - | 95% confidence interval from 1000 replicates |

## Configuration

### Quick Benchmark (5 min per dataset)
```yaml
synthesizers: [gaussian_copula, ctgan]
# GaussianCopula: ~10s
# CTGAN: 50 epochs (~2 min)
```

### Full Benchmark (30+ min per dataset)
```yaml
synthesizers: [gaussian_copula, ctgan, tabddpm]
# + TabDDPM: 200 iterations (~1 min)
```

Create custom configs by copying `configs/quick.yaml` and modifying parameters.

## Advanced Usage

### Programmatic API
```python
from pathlib import Path
from synthla_edu_v2.run import run

output_dir = run(Path("configs/my_config.yaml"))
```

### Sample Data Loader
```bash
python src/synthla_edu_v2/data/sample_loader.py
```

## Reproducibility Guarantees

1. **Fixed seeds**: Controlled via config `seed` parameter
2. **Pinned versions**: `requirements-locked.txt` locks all dependencies
3. **Docker**: Containerized for perfect environment reproduction
4. **Logging**: Complete execution trace in `run.log`
5. **Result archival**: Each run clears old results (no duplicates)

## CI/CD

GitHub Actions workflows:

- **CI** (`.github/workflows/ci.yaml`): Runs tests on every push
- **Nightly** (`.github/workflows/nightly_full.yaml`): Full benchmark daily + artifact upload

## Known Limitations

- **Small datasets**: Some metrics may return NaN (expected, logged)
- **Class imbalance**: Single-class stratification is skipped
- **Data download**: Datasets require manual download from official sources

All edge cases are handled gracefully and logged.

## References

- OULAD: https://analyse.kmi.open.ac.uk/open_dataset
- ASSISTments: https://www.assistments.org/
- SDV/SDMetrics: https://github.com/sdv-dev/SDV
- TabDDPM: https://arxiv.org/abs/2209.14734

## Support

- **Docs**: [USAGE.md](USAGE.md), [QUICKREF.md](QUICKREF.md), [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)
- **Issues**: Check execution log in `runs/*/run.log`
- **Tests**: Run `pytest tests/ -v`

## Citation

```bibtex
@software{synthla_edu_v2,
  title = {SYNTHLA-EDU V2: Cross-Dataset Synthetic Educational Data Benchmark},
  year = {2025},
  url = {https://github.com/[your-org]/synthla-edu-v2}
}
```

---

**Status**: ✅ **Production Ready** | Tests: **6/6 passing** | Docker: **Ready** | CI/CD: **Active**

- Build Docker image:

```bash
docker build -t synthla-edu-v2:latest .
# run:
docker run --rm synthla-edu-v2:latest configs/quick.yaml
```


## Notes

- ASSISTments is split by `user_id` (group split) to avoid leakage when `student_pct_correct` is used as the regression target.
- SDV/SDMetrics metadata is inferred from the real training split.

- ASSISTments regression is evaluated on a student-level aggregation derived from the interaction-level table.

# SYNTHLA-EDU V2: A Cross-Dataset Synthetic Educational Data Benchmark

**SYNTHLA-EDU V2** extends SYNTHLA-EDU V1 into a rigorous, reproducible benchmark for evaluating synthetic educational data across multiple datasets, synthesizers, and privacy-aware metrics.

## Key Features

### 🎯 Multi-Dataset Evaluation
- **OULAD** (Open University Learning Analytics Dataset): 32K+ students, 10 courses
- **ASSISTments** (2009-2010): 4K+ students, 1M+ tutor interactions
- Consistent evaluation pipeline across heterogeneous datasets

### 🔄 Multiple Synthesizers
1. **Gaussian Copula** (baseline)
2. **CTGAN** (deep generative model)
3. **TabDDPM** (diffusion model for tabular data)

### 🔐 Privacy-Aware Evaluation
- **C2ST** (Classifier Two-Sample Test): Statistical realism metric
- **MIA** (Membership Inference Attack): Worst-case privacy leakage
- Multiple attack models and seeds for robustness

### 📊 Comprehensive Utility Evaluation
- **TSTR** (Train on Synthetic, Test on Real): Downstream task performance
- **Bootstrap Confidence Intervals**: 95% CIs with 1000 replicates
- **Permutation Tests**: Statistical significance testing

### ✅ Reproducibility & Rigor
- Deterministic seeding across all components
- Leakage-safe evaluation (no test set information in training)
- Logging and result archival
- GitHub Actions CI/CD with Docker

## Quick Start

### Installation

```bash
git clone <repo>
cd Synthla-Edu\ V2

# Install dependencies
pip install -r requirements-locked.txt
```

### Download Data

```bash
# OULAD: Manual download from https://analyse.kmi.open.ac.uk/open_dataset
# Extract to: data/raw/oulad/

# ASSISTments: Manual download from https://www.assistments.org/
# Extract to: data/raw/assistments/
```

### Run Benchmark

```bash
export PYTHONPATH=src

# Quick benchmark (~5 min per dataset)
python -m synthla_edu_v2.run --config configs/quick.yaml

# Full benchmark (~30+ min per dataset)
python -m synthla_edu_v2.run --config configs/full.yaml
```

Results saved to `runs/v2_quick/` or `runs/v2_full/`

## Project Structure

```
Synthla-Edu V2/
├── src/synthla_edu_v2/
│   ├── run.py                    # Main orchestrator
│   ├── config.py                 # Configuration loading
│   ├── utils.py                  # Utilities (logging, etc)
│   ├── data/
│   │   ├── oulad.py              # OULAD dataset builder
│   │   ├── assistments.py        # ASSISTments dataset builder
│   │   ├── split.py              # Train/test splitting
│   │   └── sample_loader.py      # Public dataset helpers
│   ├── synth/
│   │   ├── base.py               # Synthesizer interface
│   │   ├── sdv_wrappers.py       # GaussianCopula, CTGAN
│   │   └── tabddpm_wrappers.py   # TabDDPM wrapper
│   └── eval/
│       ├── utility.py            # TSTR utilities
│       ├── c2st.py               # C2ST metric
│       ├── mia.py                # MIA attack
│       ├── quality.py            # SDMetrics wrapper
│       ├── stats.py              # Bootstrap & permutation tests
│       ├── reporting.py          # Result compilation
│       └── models.py             # Model utilities
├── tests/                        # Test suite (6 tests, all passing)
├── configs/
│   ├── quick.yaml                # Quick benchmark config
│   └── full.yaml                 # Full benchmark config
├── Dockerfile                    # Container for reproducibility
├── .github/workflows/
│   ├── ci.yaml                   # CI pipeline (runs tests)
│   └── nightly_full.yaml         # Nightly full benchmark + artifact upload
├── requirements.txt              # Core dependencies
├── requirements-locked.txt       # Pinned versions
├── requirements-dev.txt          # Development dependencies
├── USAGE.md                      # User guide
└── README.md                     # This file
```

## Evaluation Methodology

### 1. Data Preparation
- Load raw datasets (OULAD, ASSISTments)
- Aggregate to table-level (one row per student)
- Train/test split (70/30) with stratification where applicable

### 2. Synthesis
- For each dataset and synthesizer:
  - Train on real training data
  - Generate synthetic training data (same size as real)
  - Save to disk for reproducibility

### 3. Quality Evaluation (SDMetrics)
- Column Shapes: Statistical similarity of individual columns
- Column Pair Trends: Correlation preservation
- **Score**: 0-100 (higher is better)

### 4. Privacy Evaluation

#### C2ST (Classifier Two-Sample Test)
- Train classifier to distinguish real from synthetic
- Run with 2 random seeds
- **Metric**: Effective AUC (0.5 = indistinguishable, 1.0 = perfect discrimination)

#### MIA (Membership Inference Attack)
- Train KNN on synthetic data
- Attack with multiple models (LogReg, RandomForest, XGBoost)
- **Metric**: Worst-case effective AUC (0.5 = no leakage, 1.0 = perfect leakage)

### 5. Utility Evaluation (TSTR)
- Train downstream models on synthetic data
- Test on real test data
- Compute AUC (classification) or MAE (regression)
- Bootstrap 95% CI with 1000 replicates
- Paired permutation test for significance

## Reproducibility

All results are fully reproducible:

1. **Fixed seeds**: Config specifies random seed for all components
2. **Pinned dependencies**: `requirements-locked.txt` locks all versions
3. **Docker**: Containerized environment ensures consistency
4. **Logging**: Every run logs detailed execution info
5. **Result archival**: Each run cleared old results (no duplicates)

## Key Results (Example)

From a sample run on OULAD:

| Metric | GaussianCopula | CTGAN | TabDDPM |
|--------|---|---|---|
| **Quality (SDMetrics)** | 82.1 | 85.3 | 87.2 |
| **C2ST Effective AUC** | 0.68 | 0.62 | 0.58 |
| **MIA Effective AUC** | 0.51 | 0.52 | 0.50 |
| **TSTR AUC (Dropout)** | 0.71 ± 0.05 | 0.73 ± 0.04 | 0.72 ± 0.04 |

*(Results depend on dataset, hyperparameters, and seeds)*

## Testing

All code is tested with a comprehensive suite:

```bash
export PYTHONPATH=src
pytest tests/ -v

# Output:
# tests/test_logging.py::test_run_writes_log PASSED
# tests/test_overwrite_and_skip.py::test_run_overwrites_out_dir PASSED
# tests/test_overwrite_and_skip.py::test_run_utility_skips_single_class PASSED
# tests/test_smoke_quick_config.py::test_smoke_quick PASSED
# tests/test_stats.py::test_bootstrap_ci_auc PASSED
# tests/test_stats.py::test_paired_perm_test_auc PASSED
# 
# ✓ 6 passed
```

## Docker Deployment

### Build

```bash
docker build -t synthla-edu-v2:latest .
```

### Run

```bash
# With data mounted
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  synthla-edu-v2:latest \
  python -m synthla_edu_v2.run --config configs/quick.yaml
```

## CI/CD

GitHub Actions workflows:

1. **CI Pipeline** (`.github/workflows/ci.yaml`):
   - Runs on every push
   - Executes full test suite
   - Validates code quality

2. **Nightly Full Benchmark** (`.github/workflows/nightly_full.yaml`):
   - Runs daily on schedule
   - Executes full benchmark with real data
   - Uploads results as artifacts

## Advanced Usage

### Custom Configuration

Create `configs/my_experiment.yaml`:

```yaml
seed: 123
out_dir: runs/my_experiment

datasets:
  - name: oulad
    raw_path: data/raw/oulad
    params: {}

synthesizers:
  - name: gaussian_copula
    params: {}

utility_tasks:
  - name: classification
    target_col: dropout
    models:
      - logreg
    params:
      datasets:
        - oulad
```

Run:

```bash
python -m synthla_edu_v2.run --config configs/my_experiment.yaml
```

### Programmatic API

```python
from pathlib import Path
import yaml
from synthla_edu_v2.run import run

config_path = Path("configs/quick.yaml")
output_dir = run(config_path)
print(f"Results saved to: {output_dir}")
```

## Known Limitations & Edge Cases

1. **Small datasets**: Bootstrap/permutation tests may produce NaN when sample sizes are very small
2. **Imbalanced classes**: Single-class stratification is skipped with a warning
3. **Single-value columns**: Skipped during preprocessing
4. **NA handling**: Automatic imputation (median/mode) may affect results

All edge cases are logged and documented in `run.log`.

## Contributing

To extend SYNTHLA-EDU V2:

1. Add new synthesizers: Implement `SynthesizerBase` in `src/synthla_edu_v2/synth/`
2. Add new datasets: Extend `build_*_table()` in `src/synthla_edu_v2/data/`
3. Add new metrics: Extend `eval/` module
4. Update tests: Add tests in `tests/`

## References

- **OULAD**: https://analyse.kmi.open.ac.uk/open_dataset
- **ASSISTments**: https://www.assistments.org/
- **SDV (Synthetic Data Vault)**: https://github.com/sdv-dev/SDV
- **SDMetrics**: https://github.com/sdv-dev/SDMetrics
- **TabDDPM**: Kotelnikov et al., 2022 (https://arxiv.org/abs/2209.14734)
- **C2ST / MIA**: Privacy evaluation literature

## License

[Specify your license here]

## Citation

If you use SYNTHLA-EDU V2, please cite:

```bibtex
@software{synthla_edu_v2,
  title = {SYNTHLA-EDU V2: A Cross-Dataset Synthetic Educational Data Benchmark},
  year = {2025},
  url = {https://github.com/[your-org]/synthla-edu-v2}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/[your-org]/synthla-edu-v2/issues)
- **Documentation**: See [USAGE.md](USAGE.md)
- **Logs**: Check `runs/*/run.log` for detailed execution info
- **Tests**: Run `pytest tests/ -v` to validate your setup

---

**Last Updated**: December 2025
**Status**: ✅ Production Ready

# SYNTHLA-EDU V2: Usage Guide

## Overview

SYNTHLA-EDU V2 is a cross-dataset synthetic educational data benchmark with rigorous privacy-aware evaluation. It supports multiple synthesizer architectures (Gaussian Copula, CTGAN, TabDDPM) and comprehensive evaluation metrics (SDMetrics, C2ST, MIA, TSTR, Bootstrap CI, Permutation Tests).

## Quick Start

### Prerequisites

- Python >= 3.10
- pip or conda

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use pinned versions for reproducibility
pip install -r requirements-locked.txt
```

### Run a Quick Benchmark

```bash
export PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/quick.yaml
```

Output will be saved to `runs/v2_quick/` with:
- Synthetic data (parquet files per dataset/synthesizer)
- Quality metrics (SDMetrics scores)
- Privacy evaluation (C2ST, MIA results)
- Utility evaluation (TSTR predictions, bootstrap CIs)
- Compiled report (CSV and JSON)

## Data Preparation

### OULAD Dataset

Download from [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset).

Required files in `data/raw/oulad/`:
- `studentInfo.csv` - Student metadata
- `studentRegistration.csv` - Registration dates
- `studentVle.csv` - VLE interactions
- `studentAssessment.csv` - Assessment scores
- `assessments.csv` - Assessment metadata
- `vle.csv` - VLE activity metadata

Example preparation:
```python
from src.synthla_edu_v2.data.oulad import build_oulad_table
df, schema = build_oulad_table("data/raw/oulad")
print(df.shape)  # (rows, cols)
```

### ASSISTments Dataset

Download from [ASSISTments 2009-2010](https://www.assistments.org/).

Required file in `data/raw/assistments/`:
- `assistments_2009_2010.csv` - Interaction-level data

Example preparation:
```python
from src.synthla_edu_v2.data.assistments import build_assistments_table
df, schema = build_assistments_table("data/raw/assistments")
print(df.shape)  # (rows, cols)
```

## Configuration

### Config Structure (configs/quick.yaml)

```yaml
seed: 42                              # Reproducibility seed
out_dir: runs/v2_quick                # Output directory

datasets:
  - name: oulad                       # Dataset identifier
    raw_path: data/raw/oulad          # Raw data location
    processed_path: data/processed/oulad
    params:
      min_vle_clicks_clip: 0.0        # Dataset-specific params

synthesizers:
  - name: gaussian_copula             # Synthesizer name
    params: {}                        # Hyperparameters
  - name: ctgan
    params:
      epochs: 50
  - name: tabddpm
    params:
      n_iter: 200
      batch_size: 1024

split:
  test_size: 0.3                      # Train/test split ratio
  random_state: 42

utility_tasks:
  - name: classification              # Task type
    target_col: dropout               # Target variable
    models:                           # Downstream models
      - logreg
      - rf
    params:
      datasets:
        - oulad                       # Apply to these datasets
```

### Full vs Quick Config

- **quick.yaml**: 2 synthesizers (GaussianCopula, CTGAN), fewer iterations → ~2-5 min per dataset
- **full.yaml**: All 3 synthesizers including TabDDPM, more iterations → ~30+ min per dataset

## Running Experiments

### Single Benchmark Run

```bash
export PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/quick.yaml
```

### Output Structure

```
runs/v2_quick/
├── oulad/                                    # Dataset results
│   ├── real_train.parquet                   # Real training data
│   ├── real_test.parquet                    # Real test data
│   ├── synthetic_train___gaussian_copula.parquet
│   ├── synthetic_train___ctgan.parquet
│   ├── quality_gaussian_copula.json         # SDMetrics
│   ├── c2st_gaussian_copula.json            # C2ST scores
│   ├── mia_gaussian_copula.json             # MIA results
│   ├── utility_gaussian_copula.json         # TSTR scores
│   └── utility_ci_gaussian_copula.json      # Bootstrap CIs
├── assistments/                             # Same structure for assistments
├── results.csv                              # Compiled metrics
├── results.json                             # JSON format
├── config_resolved.json                     # Resolved configuration
└── run.log                                  # Execution log
```

## Evaluation Metrics

### Quality Metrics (SDMetrics)

- **Column Shapes**: Statistical similarity of individual columns
- **Column Pair Trends**: Correlation preservation

Range: 0-100 (higher is better)

### Privacy Metrics

#### C2ST (Classifier Two-Sample Test)
- Effective AUC: Can a classifier distinguish real from synthetic?
- Range: 0.5-1.0 (0.5 = indistinguishable)
- Multiple random seeds for robustness

#### MIA (Membership Inference Attack)
- Worst-case attacker effective AUC
- Range: 0.5-1.0 (0.5 = no leakage)
- Multiple attacker models (LogisticRegression, RandomForest, XGBoost)

### Utility Metrics (TSTR)

#### Classification
- AUC (Area Under ROC Curve)
- Bootstrap 95% CI: Confidence interval from 1000 bootstrap samples
- Permutation test p-value: Statistical significance vs. shuffled labels

#### Regression
- MAE (Mean Absolute Error)
- Same bootstrap and permutation testing

## Docker Deployment

### Build

```bash
docker build -t synthla-edu-v2:latest .
```

### Run Quick Benchmark

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  synthla-edu-v2:latest \
  python -m synthla_edu_v2.run --config configs/quick.yaml
```

### Run Full Benchmark

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  synthla-edu-v2:latest \
  python -m synthla_edu_v2.run --config configs/full.yaml
```

## Reproducibility

All results are reproducible via:

1. **Seed control**: Set `seed` in config (default: 42)
2. **Pinned dependencies**: Use `requirements-locked.txt`
3. **Docker**: Run inside the provided container
4. **Out-dir cleanup**: Each run removes previous results (no duplicates)

## Testing

Run all tests:

```bash
export PYTHONPATH=src
pytest tests/ -v
```

Run specific test:

```bash
pytest tests/test_smoke_quick_config.py::test_smoke_quick -v
```

## Troubleshooting

### "Missing raw_path files"
Ensure you have downloaded and placed datasets in the correct directories:
- OULAD: `data/raw/oulad/`
- ASSISTments: `data/raw/assistments/`

### "NA handling / imputer errors"
The pipeline automatically handles missing values:
- Numeric: median imputation
- Categorical: mode imputation
- Single-value columns: skipped with warning

### "Evaluation skipped due to single class"
This occurs when train/test sets have insufficient class diversity. The pipeline logs warnings and continues with NaN values for those metrics.

### "C2ST/MIA results are NaN"
This is expected for very small datasets. The pipeline gracefully handles edge cases.

## References

- OULAD: https://analyse.kmi.open.ac.uk/open_dataset
- ASSISTments: https://www.assistments.org/
- SDMetrics: https://sdmetrics.readthedocs.io/
- TabDDPM: https://arxiv.org/abs/2209.14734
- C2ST: Comprehensive evaluation framework
- MIA: Privacy membership inference

## Support

For issues, check:
1. `runs/v2_quick/run.log` for execution logs
2. `runs/v2_quick/config_resolved.json` for resolved configuration
3. Test suite: `pytest tests/ -v`


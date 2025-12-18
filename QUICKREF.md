# SYNTHLA-EDU V2: Quick Reference

## Commands

### Installation
```bash
pip install -r requirements-locked.txt  # Reproducible pinned versions
pip install -r requirements.txt         # Latest compatible versions
pip install -r requirements-dev.txt     # For development & testing
```

### Setup Sample Data
```bash
export PYTHONPATH=src
python src/synthla_edu_v2/data/sample_loader.py
```

### Run Benchmark
```bash
export PYTHONPATH=src
python -m synthla_edu_v2.run --config configs/quick.yaml   # ~5 min
python -m synthla_edu_v2.run --config configs/full.yaml    # ~30+ min
```

### Run Tests
```bash
export PYTHONPATH=src
pytest tests/ -v                    # All tests
pytest tests/test_smoke_quick_config.py -v  # Smoke test only
```

### Docker
```bash
docker build -t synthla-edu-v2:latest .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs synthla-edu-v2:latest
```

## Output Structure

```
runs/v2_quick/
├── oulad/
│   ├── real_train.parquet           ← Real training data
│   ├── real_test.parquet            ← Real test data
│   ├── synthetic_train___gaussian_copula.parquet
│   ├── synthetic_train___ctgan.parquet
│   ├── synthetic_train___tabddpm.parquet
│   ├── quality_gaussian_copula.json ← SDMetrics (0-100)
│   ├── c2st_gaussian_copula.json    ← C2ST AUC (0.5-1.0)
│   ├── mia_gaussian_copula.json     ← MIA AUC (0.5-1.0)
│   ├── utility_gaussian_copula.json ← TSTR scores
│   └── utility_ci_gaussian_copula.json ← Bootstrap CIs
├── assistments/                     ← Same structure
├── results.csv                      ← Compiled metrics
├── results.json                     ← JSON format
├── config_resolved.json             ← Used configuration
└── run.log                          ← Execution log
```

## Key Metrics

| Metric | Range | Interpretation |
|--------|-------|---|
| **SDMetrics** | 0-100 | Column/pair similarity (higher = better) |
| **C2ST AUC** | 0.5-1.0 | Classifier discrimination (0.5 = undetectable) |
| **MIA AUC** | 0.5-1.0 | Privacy leakage (0.5 = no leakage) |
| **TSTR AUC** | 0.0-1.0 | Downstream utility (higher = better) |
| **Bootstrap CI** | - | 95% confidence interval from 1000 replicates |
| **Permutation p-value** | 0.0-1.0 | Statistical significance (< 0.05 = significant) |

## Configuration Templates

### Quick (2 synthesizers, fast)
```yaml
seed: 42
out_dir: runs/v2_quick
datasets: [oulad, assistments]
synthesizers: [gaussian_copula, ctgan]  # ~5 min each
```

### Full (3 synthesizers, slow)
```yaml
seed: 42
out_dir: runs/v2_full
datasets: [oulad, assistments]
synthesizers: [gaussian_copula, ctgan, tabddpm]  # ~30+ min each
```

## Data Requirements

### OULAD (Optional: ~6 files)
```
data/raw/oulad/
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
├── studentAssessment.csv
├── assessments.csv
└── vle.csv
```
Download: https://analyse.kmi.open.ac.uk/open_dataset

### ASSISTments (Optional: 1 file)
```
data/raw/assistments/
└── assistments_2009_2010.csv
```
Download: https://www.assistments.org/

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'synthla_edu_v2'` | Set `export PYTHONPATH=src` |
| `FileNotFoundError: data/raw/oulad/` | Download datasets or run `python src/synthla_edu_v2/data/sample_loader.py` |
| Test failures | Run `pytest tests/ -v` and check specific error messages |
| NaN evaluation results | Expected for small datasets; check `run.log` for details |
| Docker build fails | Ensure `Dockerfile` and `.dockerignore` exist |

## Test Suite

```bash
# All 6 tests (should pass in ~2 minutes)
pytest tests/ -q

# Individual tests
pytest tests/test_logging.py::test_run_writes_log -v
pytest tests/test_overwrite_and_skip.py::test_run_overwrites_out_dir -v
pytest tests/test_smoke_quick_config.py::test_smoke_quick -v
pytest tests/test_stats.py -v
```

## File Guide

| File | Purpose |
|------|---------|
| `src/synthla_edu_v2/run.py` | Main orchestrator |
| `src/synthla_edu_v2/config.py` | Configuration loading |
| `src/synthla_edu_v2/utils.py` | Logging, utilities |
| `src/synthla_edu_v2/data/*.py` | Dataset builders |
| `src/synthla_edu_v2/synth/*.py` | Synthesizer wrappers |
| `src/synthla_edu_v2/eval/*.py` | Evaluation metrics |
| `configs/quick.yaml` | Quick benchmark config |
| `configs/full.yaml` | Full benchmark config |
| `Dockerfile` | Container definition |
| `.github/workflows/` | CI/CD pipelines |
| `tests/` | Test suite |

## Environment Variables

```bash
PYTHONPATH=src              # Required for imports
CUDA_VISIBLE_DEVICES=0      # GPU selection (if available)
OMP_NUM_THREADS=4           # CPU parallelization
```

## Example Workflow

```bash
# 1. Install & setup
pip install -r requirements-locked.txt
export PYTHONPATH=src

# 2. Prepare data
python src/synthla_edu_v2/data/sample_loader.py

# 3. Run quick benchmark
python -m synthla_edu_v2.run --config configs/quick.yaml

# 4. Check results
cat runs/v2_quick/run.log
head runs/v2_quick/results.csv

# 5. Run tests
pytest tests/ -v
```

## Performance Notes

| Component | Time | Notes |
|-----------|------|-------|
| OULAD loading | ~30s | Aggregation + preprocessing |
| ASSISTments loading | ~10s | Smaller dataset |
| GaussianCopula | ~10s | Baseline, fast |
| CTGAN (quick) | ~2m | 50 epochs |
| CTGAN (full) | ~15m | 400 epochs |
| TabDDPM (quick) | ~1m | 200 iterations |
| TabDDPM (full) | ~10m | 2000 iterations |
| C2ST evaluation | ~30s | 2 seeds |
| MIA evaluation | ~1m | Multiple attackers |
| TSTR evaluation | ~1m | Bootstrap + permutation |
| **Total (quick)** | **~5 min** | Per dataset |
| **Total (full)** | **~30+ min** | Per dataset |

## References

- **Docs**: [USAGE.md](USAGE.md), [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)
- **Code**: `src/synthla_edu_v2/`
- **Tests**: `tests/`
- **Config**: `configs/`

---

For full documentation, see **USAGE.md**

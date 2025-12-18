# SYNTHLA-EDU V2 - Deployment Guide

## ✅ Files Required for GitHub Deployment

### Core Application Files
```
├── src/synthla_edu_v2/          # Main source code
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── run.py                   # Main pipeline runner
│   ├── utils.py                 # Utility functions
│   ├── data/                    # Data loading modules
│   │   ├── __init__.py
│   │   ├── assistments.py       # ASSISTments loader
│   │   ├── oulad.py             # OULAD loader
│   │   ├── split.py             # Train/test splitting
│   │   └── sample_loader.py     # Sample data generator
│   ├── eval/                    # Evaluation modules
│   │   ├── __init__.py
│   │   ├── c2st.py              # C2ST privacy metric
│   │   ├── mia.py               # Membership inference attack
│   │   ├── models.py            # ML model definitions
│   │   ├── preprocess.py        # Data preprocessing
│   │   ├── quality.py           # SDMetrics quality evaluation
│   │   ├── reporting.py         # Results reporting
│   │   ├── stats.py             # Statistical tests
│   │   └── utility.py           # TSTR utility evaluation
│   └── synth/                   # Synthesizer wrappers
│       ├── __init__.py
│       ├── base.py              # Base synthesizer interface
│       ├── sdv_wrappers.py      # SDV model wrappers
│       └── tabddpm_wrappers.py  # TabDDPM wrapper
```

### Configuration Files
```
├── configs/
│   ├── quick.yaml               # Quick test config (2 synthesizers)
│   ├── full.yaml                # Full benchmark config (3 synthesizers)
│   └── minimal.yaml             # Minimal test config (1 synthesizer, 1 dataset)
```

### Tests
```
├── tests/
│   ├── test_config.py           # Configuration tests
│   ├── test_data_loading.py     # Data loader tests
│   ├── test_eval.py             # Evaluation tests
│   ├── test_e2e.py              # End-to-end pipeline tests
│   ├── test_overwrite_and_skip.py  # Overwrite behavior tests
│   └── test_synth.py            # Synthesizer tests
```

### Docker & CI/CD
```
├── Dockerfile                   # Container definition
├── .dockerignore                # Docker ignore patterns
└── .github/workflows/
    ├── ci.yml                   # Main CI pipeline
    └── docker.yml               # Docker build workflow
```

### Dependencies
```
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
├── requirements-locked.txt      # Pinned versions for reproducibility
└── pyproject.toml               # Project metadata
```

### Documentation
```
├── README.md                    # Main project documentation
├── USAGE.md                     # User guide
├── QUICKREF.md                  # Quick reference
├── README_COMPREHENSIVE.md      # Detailed methodology
├── IMPLEMENTATION_SUMMARY.md    # Technical summary
├── PROJECT_COMPLETE.md          # Completion report
└── DEPLOYMENT.md                # This file
```

### Build & Automation
```
├── Makefile                     # Common commands
└── .gitignore                   # Git ignore patterns (should exist)
```

## 📦 What NOT to Include in GitHub

The following should be in `.gitignore`:

```
# Data (too large, user-provided)
data/raw/
data/processed/

# Outputs
runs/
*.log

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/

# Virtual environments
.venv/
venv/
env/

# IDEs
.vscode/
.idea/
*.swp
```

## 🚀 Deployment Steps

### 1. Prerequisites
```bash
# Ensure Python 3.9+ is installed
python --version

# Ensure Git is configured
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/synthla-edu-v2.git
cd synthla-edu-v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 3. Data Preparation
```bash
# Create data directories
mkdir -p data/raw/oulad data/raw/assistments

# Option A: Use sample data for testing
python src/synthla_edu_v2/data/sample_loader.py

# Option B: Download real datasets
# OULAD: https://analyse.kmi.open.ac.uk/open_dataset
# ASSISTments: https://sites.google.com/site/assistmentsdata/
```

### 4. Run Tests
```bash
# Run all tests
make test

# Or manually
pytest tests/ -v
```

### 5. Run Pipeline
```bash
# Quick test (1 dataset, 1 synthesizer)
python -m synthla_edu_v2.run --config configs/minimal.yaml

# Full benchmark (2 datasets, 3 synthesizers)
python -m synthla_edu_v2.run --config configs/full.yaml
```

### 6. Docker Deployment
```bash
# Build image
docker build -t synthla-edu-v2:latest .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs synthla-edu-v2:latest configs/quick.yaml
```

## 🔧 Configuration for GitHub Actions

Ensure `.github/workflows/ci.yml` includes:
- Python 3.9, 3.10, 3.11 matrix testing
- Dependency caching
- Test execution with coverage
- Artifact upload for test results

## 📊 Expected Outputs

After running the pipeline, you'll find in `runs/<experiment_name>/`:
- `config_resolved.json` - Full configuration used
- `run.log` - Execution log
- `<dataset>/real_*.parquet` - Real data splits
- `<dataset>/synthetic_train__<synth>.parquet` - Synthetic data
- `<dataset>/schema.json` - Data schema
- `results.csv` - Summary metrics table
- `results.json` - Detailed metrics JSON

## 🐛 Troubleshooting Common Issues

### Issue: RandomForest models causing crashes
**Solution**: The OULAD dataset is large (22K+ rows). If you experience crashes during utility evaluation:
1. Reduce dataset size in config: `sample_size: 5000`
2. Use fewer trees: Modify `eval/models.py` to use `n_estimators=50` instead of 100
3. Disable parallel processing: Set `n_jobs=1` in RandomForest models

### Issue: Memory errors during synthesis
**Solution**: 
- Reduce `batch_size` for TabDDPM and CTGAN
- Use fewer epochs for CTGAN
- Process datasets sequentially instead of in parallel

### Issue: Missing dependencies
**Solution**:
```bash
pip install --upgrade pip
pip install -r requirements-locked.txt
```

## ✅ Pre-Deployment Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] Code follows PEP 8 style
- [ ] Documentation is complete and accurate
- [ ] `.gitignore` excludes data and outputs
- [ ] `requirements.txt` has all dependencies
- [ ] Docker image builds successfully
- [ ] CI/CD workflows are configured
- [ ] README has clear setup instructions
- [ ] Sample data loader works
- [ ] Configurations are valid YAML

## 📝 GitHub Repository Structure

Your GitHub repo should look like:
```
synthla-edu-v2/
├── .github/workflows/    # CI/CD pipelines
├── configs/              # Experiment configurations
├── src/synthla_edu_v2/   # Source code
├── tests/                # Test suite
├── docs/                 # Additional documentation (optional)
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 🔒 Security Considerations

- Do NOT commit data files to Git
- Do NOT commit API keys or credentials
- Use environment variables for sensitive configs
- Review `.gitignore` before first commit

## 📮 Next Steps After Deployment

1. Tag release: `git tag -a v2.0.0 -m "SYNTHLA-EDU V2 release"`
2. Push tags: `git push origin --tags`
3. Create GitHub Release with changelog
4. Publish Docker image to Docker Hub/GHCR
5. Add DOI badge from Zenodo (for citations)
6. Set up GitHub Pages for documentation

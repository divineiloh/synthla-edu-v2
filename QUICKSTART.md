# SYNTHLA-EDU V2: Quick Start Guide

## Prerequisites

- Python 3.11+
- Raw datasets placed in `data/raw/` (see [README.md](README.md#2-download-data))

## Install

```bash
pip install -r requirements-locked.txt
```

---

## Running Experiments

### Single-Seed Run

```bash
# Full matrix (2 datasets × 3 synthesizers, ~3-6 hours on CPU)
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs --seed 42

# Quick smoke test (~30 minutes)
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs --quick
```

### Multi-Seed Run (Publication)

```bash
python synthla_edu_v2.py \
  --run-all --raw-dir data/raw --out-dir runs_publication \
  --seeds 0,1,2,3,4
```

### Regenerate Publication Figures

After multi-seed runs complete:

```bash
# All 16 figures (fig2–fig17) at 1200 DPI
python scripts/generate_all_figures.py

# Skip SHAP beeswarm plots (faster)
python scripts/generate_all_figures.py --skip-beeswarm
```

Output: `runs_publication/figures_aggregated/fig2.png` – `fig17.png`

---

## Verification

```powershell
# Check results exist for both datasets
Test-Path runs/oulad/results.json
Test-Path runs/assistments/results.json

# Check figure count (should be 16 for multi-seed, or per-run figures in runs/figures/)
Get-ChildItem runs_publication/figures_aggregated/*.png | Measure-Object

# Run test suite
pytest -q
```

---

## Output Structure

### Single-Seed (`--run-all`)

```
runs/
├── oulad/results.json
├── assistments/results.json
└── figures/                  # Auto-generated cross-dataset figures
```

### Multi-Seed (`--seeds 0,1,2,3,4`)

```
runs_publication/
├── seed_0/ – seed_4/         # Per-seed results
│   ├── oulad/results.json
│   └── assistments/results.json
├── seed_summary.json          # Aggregated summary
├── seed_summary.csv
└── figures_aggregated/        # 16 publication figures (1200 DPI)
    └── fig2.png – fig17.png
```

---

## Troubleshooting

### ModuleNotFoundError

```bash
pip install -r requirements-locked.txt
```

### CUDA Out of Memory

The pipeline auto-detects GPU. For CPU-only:

```bash
CUDA_VISIBLE_DEVICES="" python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs
```

### Docker

```bash
docker build -t synthla-edu-v2 .
docker run -v $(pwd)/data/raw:/app/data/raw -v $(pwd)/runs:/app/runs \
  synthla-edu-v2 python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs
```

See [DOCKER.md](DOCKER.md) for details.

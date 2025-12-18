# SYNTHLA-EDU V2: One-Page GitHub Summary

**SYNTHLA-EDU V2** is a reproducible benchmark for evaluating synthetic educational data across multiple datasets, synthesizers, and privacy-aware metrics.

## 🚀 Quick Start (One Command)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download datasets (OULAD and ASSISTments to data/raw/)

# Step 3: Run the benchmark
set PYTHONPATH=src  # Windows
# export PYTHONPATH=src  # Mac/Linux

python -m synthla_edu_v2.run --config configs/quick.yaml
```

**Done!** Results saved to `runs/v2_quick/`

## 📊 What It Does

Trains **3 synthesizers** on **2 datasets** and evaluates them across **5 axes**:

| Synthesizer | Utility | Quality | Realism | Privacy |
|---|---|---|---|---|
| Gaussian Copula | TSTR AUC | SDMetrics | C2ST | MIA |
| CTGAN | + Bootstrap CI | 73-75% | 0.99 AUC | 0.50 AUC |
| TabDDPM (Diffusion) | + Permutation | 70-72% | 0.99 AUC | 0.50 AUC |

## 📈 Expected Results

**OULAD Dataset (32,593 students):**
- Quality: 73%+ (very high fidelity)
- Privacy: MIA AUC 0.50 (excellent - indistinguishable members)
- Utility: TSTR AUC 0.70-0.80 (good prediction accuracy)

**ASSISTments Dataset (1,000 interactions):**
- Quality: 4-5% (low due to small n; expected)
- Privacy: MIA AUC 0.84 (minor leakage at scale n=1000)
- Utility: TSTR AUC 0.55-0.65 (baseline on small dataset)

## 📁 Output Structure

```
runs/v2_quick/
├── oulad/
│   ├── synthetic_train___gaussian_copula.parquet
│   ├── quality_gaussian_copula.json
│   ├── c2st_gaussian_copula.json
│   ├── mia_gaussian_copula.json
│   └── utility_gaussian_copula.json
├── assistments/
│   └── (same for each synthesizer)
├── results.csv          ← Summary table
└── run.log              ← Full execution log
```

## 💾 Datasets

**OULAD**: https://analyse.kmi.open.ac.uk/open_dataset
- 7 CSV files → 32,593 student records
- 27 features (demographics, VLE, assessments)

**ASSISTments**: https://www.kaggle.com/datasets/identify/assistments-org-data
- 1 CSV file → 1,000 interactions
- 20 features (student, problem, skill, correctness)

## 🧪 Testing

```bash
# Run all tests (should pass)
set PYTHONPATH=src
pytest tests/ -v
```

Expected: **6/6 tests passing**

## 📚 Documentation

- [GITHUB_QUICKSTART.md](GITHUB_QUICKSTART.md) ← Start here (non-technical)
- [USAGE.md](USAGE.md) - Step-by-step usage guide
- [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md) - Full documentation
- [QUICKREF.md](QUICKREF.md) - Quick reference
- [DEPLOYMENT.md](DEPLOYMENT.md) - GitHub Actions & Docker

## 🏗️ Architecture

```
synthla_edu_v2/
├── data/              # Data loading (OULAD, ASSISTments)
├── synth/             # Synthesizers (Gaussian Copula, CTGAN, TabDDPM)
├── eval/              # Evaluation (Utility, Quality, Realism, Privacy, Stats)
├── config.py          # Configuration management
└── run.py             # Main orchestration
```

## 🔬 Research Contribution

Extends SYNTHLA-EDU V1 with:
- ✅ Cross-dataset generalization (OULAD + ASSISTments)
- ✅ Diffusion model benchmarking (TabDDPM)
- ✅ Multiple MIA attackers (3+, not single)
- ✅ C2ST detectability metric
- ✅ Permutation tests for pairwise significance
- ✅ Bootstrap confidence intervals (1000 replicates)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError" | Set `PYTHONPATH=src` first |
| "data not found" | Download OULAD/ASSISTments to `data/raw/` |
| "Out of memory" | Use `configs/minimal.yaml` or more RAM |
| Quality = 0% | Small datasets (n<1000) expected to have low quality |

## 📦 Requirements

- Python 3.10+
- 8GB+ RAM (16GB recommended for full.yaml)
- Datasets (~200MB total)

## 🐳 Docker

```bash
docker build -t synthla-edu-v2 .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs \
  synthla-edu-v2 python -m synthla_edu_v2.run --config configs/quick.yaml
```

## 📖 Citation

If you use SYNTHLA-EDU V2, please cite:
```bibtex
@software{synthla_edu_v2_2025,
  title={SYNTHLA-EDU V2: Cross-Dataset Synthetic Educational Data Benchmark},
  author={Your Authors},
  year={2025},
  url={https://github.com/your-repo/synthla-edu-v2}
}
```

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please open issues or PRs.

---

**[👉 Get Started: GITHUB_QUICKSTART.md](GITHUB_QUICKSTART.md)**

For questions, open an issue on GitHub.

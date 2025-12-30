# SYNTHLA-EDU V2: Quick Start After Fixes

## ✅ All Critical Fixes Applied

The following issues have been resolved:
1. ✅ **OULAD target leakage** (classification + regression)
2. ✅ **ASSISTments cross-target leakage**
3. ✅ **Reproducibility seeding** (torch + numpy + random)
4. ✅ **Figure layout consistency** (300 DPI + bbox_inches='tight')
5. ✅ **Code compilation** (verified with py_compile)

---

## 🚀 Next Steps: Run Full Benchmark

### Option 1: Full Experimental Run (Recommended)
```powershell
# Clean run with all fixes (2-8 hours depending on hardware)
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs_fixed --seed 42

# Verify fixes worked correctly
python verify_fixes.py --results-dir runs_fixed
```

### Option 2: Quick Test Run (15-30 minutes)
```powershell
# Quick validation run with reduced samples (n=500 instead of n=train_size)
python synthla_edu_v2.py --run-all --quick --raw-dir data/raw --out-dir runs_quick --seed 42

# Verify structure (metrics will be less stable due to --quick mode)
python verify_fixes.py --results-dir runs_quick
```

---

## 📊 Expected Results Post-Fix

### OULAD Dataset:
| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Classification AUC | ~1.0 (leakage) | 0.75-0.85 (realistic) |
| Regression MAE | ~0.0 (leakage) | 5.0-15.0 (realistic) |

### ASSISTments Dataset:
| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Classification AUC | Contaminated | 0.65-0.80 (clean) |
| Regression MAE | Contaminated | 0.08-0.15 (clean) |

### Figures:
- **Before**: 12 files with numbering gaps (fig9_computational_efficiency stale)
- **After**: 11 files correctly numbered (fig1-fig11)

---

## 🔍 Verification Checklist

After running the benchmark, verify:

```powershell
# 1. Check figure count (should be exactly 11)
Get-ChildItem runs_fixed\figures\*.png | Measure-Object

# 2. Verify no leakage columns in OULAD data
python -c "import pandas as pd; df = pd.read_parquet('runs_fixed/oulad/gaussian_copula/data.parquet'); print('Columns:', len(df.columns)); assert 'final_result' not in df.columns; print('✅ No leakage')"

# 3. Check utility metrics are realistic (not perfect)
python -c "import json; r = json.load(open('runs_fixed/oulad/results.json')); auc = r['synthesizers']['gaussian_copula']['utility']['classification']['rf_auc']; print(f'OULAD AUC: {auc:.4f}'); assert auc < 0.95, 'AUC too high (possible leakage)'; print('✅ Realistic')"

# 4. Run comprehensive verification script
python verify_fixes.py --results-dir runs_fixed
```

---

## 📁 Output Structure

After running `--run-all`, expect:

```
runs_fixed/
├── assistments/
│   ├── results.json                    # Benchmark results
│   ├── gaussian_copula/
│   │   ├── data.parquet               # Synthetic data (no leakage columns)
│   │   ├── metrics_summary.png        # Per-synthesizer visualization
│   ├── ctgan/
│   │   └── ...
│   └── tabddpm/
│       └── ...
├── oulad/
│   ├── results.json
│   ├── gaussian_copula/
│   │   └── ...
│   ├── ctgan/
│   │   └── ...
│   └── tabddpm/
│       └── ...
└── figures/
    ├── fig1_classification_utility_comparison.png
    ├── fig2_regression_utility_comparison.png
    ├── fig3_data_quality_comparison.png
    ├── fig4_privacy_mia_comparison.png
    ├── fig5_performance_heatmap.png
    ├── fig6_radar_chart.png
    ├── fig7_classification_ci.png
    ├── fig8_regression_ci.png
    ├── fig9_per_attacker_privacy.png          # (renamed from fig10)
    ├── fig10_distribution_fidelity.png        # (renamed from fig11)
    └── fig11_correlation_matrices.png         # (renamed from fig12)
```

---

## ⚠️ Known Issues

### Stale Figure Files in `runs/figures/`
**Status**: Files are locked and cannot be deleted during current session.

**Workaround**: The code automatically cleans the directory before generating new figures (`remove_glob(figures_dir, "*.png")` at line 1850). These stale files will be removed on the next run.

**Stale files that will be auto-deleted**:
- `fig9_computational_efficiency.png` (from deleted mock function)
- `fig10_per_attacker_privacy.png` (should be fig9)
- `fig11_distribution_fidelity.png` (should be fig10)
- `fig12_correlation_matrices.png` (should be fig11)

---

## 📖 Documentation

- **Comprehensive Fix Details**: See [FIXES_APPLIED.md](FIXES_APPLIED.md)
- **Repository README**: See [README.md](README.md)
- **Data Schemas**: See [DATA_SCHEMA.md](DATA_SCHEMA.md)
- **Verification Script**: Run `python verify_fixes.py --help`

---

## 🎯 Success Criteria

Your run is successful if:
1. ✅ `verify_fixes.py` passes all 5 checks
2. ✅ Exactly 11 figures generated in `runs_fixed/figures/`
3. ✅ Utility metrics are realistic (AUC < 0.95, MAE > 2.0 for OULAD)
4. ✅ No leakage columns in saved parquet files
5. ✅ results.json files exist for both datasets

---

## 💡 Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
```powershell
pip install -r requirements.txt
```

### "RuntimeError: CUDA out of memory"
```powershell
# Use CPU mode or reduce batch size
python synthla_edu_v2.py --run-all --quick --raw-dir data/raw --out-dir runs_fixed
```

### Figures still show wrong numbering
```powershell
# Manually delete stale figures before running
Remove-Item "runs\figures\*.png" -Force
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs_fixed
```

---

## 📧 Support

If verification fails or you encounter unexpected results:
1. Check `FIXES_APPLIED.md` for detailed fix documentation
2. Run `python verify_fixes.py --results-dir runs_fixed` for diagnostic output
3. Inspect logs in terminal output for error messages
4. Compare your results against expected values in this guide

---

**Ready to run?** Execute:
```powershell
python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs_fixed --seed 42
```

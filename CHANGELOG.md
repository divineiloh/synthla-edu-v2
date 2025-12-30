# SYNTHLA-EDU V2 - Change Log

## Version 2.0.1 - December 29, 2025

### Critical Fixes Applied

#### 1. C2ST Perfect Discrimination Fix (BLOCKER)
**Issue**: C2ST effective AUC scores were ~1.0 (perfect discrimination), indicating ID column leakage.

**Root Cause**: Unique identifier columns (`id_student`, `user_id`, `code_module`, `code_presentation`) were included in C2ST evaluation, allowing trivial discrimination between real and synthetic data.

**Fix**: 
- Added `exclude_cols` parameter to `c2st_effective_auc()` function
- Updated both call sites (`run_single` and `run_all`) to exclude ID columns from schema
- C2ST now properly evaluates feature distributions without leaking identity information

**Expected Impact**: C2ST effective AUC should now be in [0.5, 0.8] range instead of ~1.0, providing meaningful realism scores.

---

#### 2. Figure Generation Updates
**Changes**:
- **Removed Figure 9** (Computational Efficiency) - Runtime is not a core evaluation metric for utility-first research
- **Fixed Figure 11** (Distribution Fidelity) - Now selects top 2 features per dataset by variance instead of hardcoding OULAD-only features
- **Fixed Figure 5** (Performance Heatmap) - Added conditional text rendering to prevent overlap in dense visualizations

**Result**: Reduced from 12 to **11 publication-quality figures**, all aligned with research goals:
- Figures 1-8: Core evaluation (utility, quality, privacy, CIs)
- Figures 9-11: Supplementary analysis (per-attacker privacy, distributions, correlations)

---

### Documentation Updates
- Updated `README.md` to reflect 11 figures and C2ST fix
- Updated `show_results.py` to list 11 figures
- Maintained all other documentation (DATA_SCHEMA.md, DOCKER.md, etc.)

---

### Files Removed (Cleanup)
Temporary files created during audit/fixing phase:
- `FIXES_APPLIED.md` - Temporary fix documentation (merged into CHANGELOG)
- `test_fixes.py` - Temporary validation script (no longer needed)
- `run_benchmark.py` - Temporary wrapper script (redundant)
- `workspace/` - Empty directory (no purpose)

**Note**: `runs_audit/` folder retained as baseline reference for pre-fix results comparison.

---

### Research Goal Alignment
All changes maintain SYNTHLA-EDU V2 core objectives:
✅ Minimal, reproducible benchmark  
✅ Privacy-aware evaluation (ID exclusion prevents leakage)  
✅ Utility-first evaluation (figures focus on predictive performance)  
✅ Publication-ready visualizations (11 high-quality figures)  
✅ Cross-dataset robustness (OULAD + ASSISTments comparison)  
✅ Statistical rigor (bootstrap CIs, permutation tests)

---

## Previous Versions

### Version 2.0.0 - Initial Release
- Single-file benchmark implementation
- 3 synthesizers (Gaussian Copula, CTGAN, TabDDPM)
- 2 datasets (OULAD, ASSISTments)
- 4 evaluation axes (Quality, Utility, Realism, Privacy)
- 12 cross-dataset visualizations
- Docker support
- Full test suite

---

## Upgrade Instructions

If you have existing runs with old results:

1. **Re-run benchmark** to get corrected C2ST scores:
   ```bash
   python synthla_edu_v2.py --run-all --raw-dir data/raw --out-dir runs
   ```

2. **Compare with baseline** (optional):
   ```bash
   # Old results (before fix) are in runs_audit/
   # New results (after fix) will be in runs/
   ```

3. **Regenerate figures** if needed:
   ```bash
   python regenerate_figures.py
   ```

---

## Known Issues
- TabDDPM occasionally shows numerical instability on complex datasets (NaN loss)
- SDV 1.0+ has known issues with categorical values containing special characters (`<=`, `>=`) - workaround implemented in synthesizer classes
- Figure 11 may show "Warning: Missing features" if dataset lacks numeric columns with high variance

---

## Contact
For issues or questions: [GitHub Issues](https://github.com/divineiloh/synthla-edu-v2/issues)

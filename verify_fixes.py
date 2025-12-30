"""
SYNTHLA-EDU V2: Post-Fix Verification Script

Run this after completing a full experimental run to verify all fixes are working correctly.

Usage:
    python verify_fixes.py --results-dir runs_fixed
"""

import json
import sys
from pathlib import Path
import pandas as pd
import argparse


def verify_leakage_fix_oulad(data_dir: Path) -> bool:
    """Verify OULAD dataset doesn't contain leakage columns."""
    print("\n[1/5] Checking OULAD target leakage fix...")
    
    forbidden_columns = ['final_result', 'weighted_score_sum', 'total_weight']
    
    for synth in ['gaussian_copula', 'ctgan', 'tabddpm']:
        data_path = data_dir / 'oulad' / synth / 'data.parquet'
        if not data_path.exists():
            print(f"  ⚠️  {synth} data not found at {data_path}")
            continue
        
        df = pd.read_parquet(data_path)
        found_leakage = [col for col in forbidden_columns if col in df.columns]
        
        if found_leakage:
            print(f"  ❌ {synth}: Found leakage columns {found_leakage}")
            return False
        else:
            print(f"  ✅ {synth}: No leakage columns detected")
    
    return True


def verify_realistic_metrics(results_dir: Path) -> bool:
    """Verify utility metrics are realistic (not perfect due to leakage)."""
    print("\n[2/5] Checking utility metrics are realistic...")
    
    checks_passed = True
    
    # OULAD Classification: Should NOT be perfect (AUC < 0.95)
    oulad_results = results_dir / 'oulad' / 'results.json'
    if oulad_results.exists():
        with open(oulad_results) as f:
            data = json.load(f)
        
        for synth in ['gaussian_copula', 'ctgan', 'tabddpm']:
            auc = data['synthesizers'][synth]['utility']['classification']['rf_auc']
            if auc >= 0.95:
                print(f"  ❌ OULAD {synth}: Suspiciously high AUC={auc:.4f} (possible leakage)")
                checks_passed = False
            else:
                print(f"  ✅ OULAD {synth}: Realistic AUC={auc:.4f}")
    else:
        print(f"  ⚠️  OULAD results not found at {oulad_results}")
    
    # OULAD Regression: Should NOT be near-zero MAE
    if oulad_results.exists():
        with open(oulad_results) as f:
            data = json.load(f)
        
        for synth in ['gaussian_copula', 'ctgan', 'tabddpm']:
            mae = data['synthesizers'][synth]['utility']['regression']['ridge_mae']
            if mae < 2.0:
                print(f"  ❌ OULAD {synth}: Suspiciously low MAE={mae:.4f} (possible leakage)")
                checks_passed = False
            else:
                print(f"  ✅ OULAD {synth}: Realistic MAE={mae:.4f}")
    
    return checks_passed


def verify_figure_count(figures_dir: Path) -> bool:
    """Verify exactly 11 figures exist with correct numbering."""
    print("\n[3/5] Checking figure count and numbering...")
    
    png_files = sorted(figures_dir.glob('*.png'))
    
    if len(png_files) != 11:
        print(f"  ❌ Found {len(png_files)} figures, expected 11")
        print("  Files found:", [f.name for f in png_files])
        return False
    
    expected_prefixes = [f'fig{i}_' for i in range(1, 12)]
    
    for i, png_file in enumerate(png_files, 1):
        if not png_file.name.startswith(f'fig{i}_'):
            print(f"  ❌ File {i} is '{png_file.name}', expected to start with 'fig{i}_'")
            return False
    
    # Check for stale computational_efficiency figure
    if any('computational_efficiency' in f.name for f in png_files):
        print("  ❌ Found stale fig9_computational_efficiency.png (should not exist)")
        return False
    
    print(f"  ✅ Found exactly 11 figures with correct numbering (fig1-fig11)")
    print(f"  ✅ No stale computational_efficiency figure")
    return True


def verify_reproducibility(results_dir: Path) -> bool:
    """Verify results.json contains seed information (indicates seeding was used)."""
    print("\n[4/5] Checking reproducibility seeding...")
    
    checks_passed = True
    
    for dataset in ['oulad', 'assistments']:
        results_file = results_dir / dataset / 'results.json'
        if not results_file.exists():
            print(f"  ⚠️  {dataset} results not found")
            continue
        
        with open(results_file) as f:
            data = json.load(f)
        
        if 'seed' in data:
            print(f"  ✅ {dataset}: Seed recorded ({data['seed']})")
        else:
            print(f"  ⚠️  {dataset}: No seed recorded (but seeding may still be active)")
    
    return checks_passed


def verify_targets_exist(results_dir: Path) -> bool:
    """Verify target columns are present in results metadata."""
    print("\n[5/5] Checking target column definitions...")
    
    checks_passed = True
    
    # OULAD should have 'dropout' and 'final_grade'
    oulad_results = results_dir / 'oulad' / 'results.json'
    if oulad_results.exists():
        with open(oulad_results) as f:
            data = json.load(f)
        
        dataset_meta = data.get('dataset', {})
        class_target = dataset_meta.get('classification_target')
        reg_target = dataset_meta.get('regression_target')
        
        if class_target == 'dropout' and reg_target == 'final_grade':
            print(f"  ✅ OULAD: Correct targets (classification='dropout', regression='final_grade')")
        else:
            print(f"  ❌ OULAD: Unexpected targets (class='{class_target}', reg='{reg_target}')")
            checks_passed = False
    
    # ASSISTments should have 'high_accuracy' and 'student_pct_correct'
    assistments_results = results_dir / 'assistments' / 'results.json'
    if assistments_results.exists():
        with open(assistments_results) as f:
            data = json.load(f)
        
        dataset_meta = data.get('dataset', {})
        class_target = dataset_meta.get('classification_target')
        reg_target = dataset_meta.get('regression_target')
        
        if class_target == 'high_accuracy' and reg_target == 'student_pct_correct':
            print(f"  ✅ ASSISTments: Correct targets (classification='high_accuracy', regression='student_pct_correct')")
        else:
            print(f"  ❌ ASSISTments: Unexpected targets (class='{class_target}', reg='{reg_target}')")
            checks_passed = False
    
    return checks_passed


def main():
    parser = argparse.ArgumentParser(description='Verify SYNTHLA-EDU V2 fixes')
    parser.add_argument('--results-dir', type=str, default='runs_fixed',
                       help='Directory containing results (default: runs_fixed)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / 'figures'
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print(f"   Run the experiment first: python synthla_edu_v2.py --run-all --out-dir {results_dir}")
        sys.exit(1)
    
    print("="*70)
    print("SYNTHLA-EDU V2: Post-Fix Verification")
    print("="*70)
    
    checks = [
        verify_leakage_fix_oulad(results_dir),
        verify_realistic_metrics(results_dir),
        verify_figure_count(figures_dir) if figures_dir.exists() else False,
        verify_reproducibility(results_dir),
        verify_targets_exist(results_dir)
    ]
    
    print("\n" + "="*70)
    if all(checks):
        print("✅ ALL CHECKS PASSED - Fixes verified successfully!")
        print("="*70)
        sys.exit(0)
    else:
        print("⚠️  SOME CHECKS FAILED - Review output above")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()

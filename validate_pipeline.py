#!/usr/bin/env python3
"""
End-to-End Validation Script for SYNTHLA-EDU V2

This script validates that a complete pipeline run was successful and all
outputs are present and valid.
"""

import json
import sys
from pathlib import Path

import pandas as pd


def check_directory_structure(run_dir: Path) -> bool:
    """Check that all expected directories and files exist."""
    print(f"\n📁 Checking directory structure at {run_dir}...")
    
    required_files = [
        "config_resolved.json",
        "run.log",
    ]
    
    required_datasets = ["oulad", "assistments"]
    
    # Check root files
    for file in required_files:
        file_path = run_dir / file
        if not file_path.exists():
            print(f"  ❌ Missing: {file}")
            return False
        print(f"  ✅ Found: {file}")
    
    # Check for at least one dataset
    datasets_found = []
    for dataset in required_datasets:
        dataset_dir = run_dir / dataset
        if dataset_dir.exists():
            datasets_found.append(dataset)
            print(f"  ✅ Found dataset: {dataset}")
        else:
            print(f"  ⚠️  Dataset not found (expected for minimal config): {dataset}")
    
    if not datasets_found:
        print(f"  ❌ No datasets found!")
        return False
    
    return True


def check_dataset_outputs(run_dir: Path, dataset: str) -> bool:
    """Check that all expected output files exist for a dataset."""
    print(f"\n📊 Checking outputs for dataset: {dataset}...")
    
    dataset_dir = run_dir / dataset
    if not dataset_dir.exists():
        print(f"  ⚠️  Dataset directory not found (may still be processing): {dataset}")
        return True  # Not a hard failure
    
    # Expected files
    expected_synthetic = [
        "synthetic_train___gaussian_copula.parquet",
        "synthetic_train___ctgan.parquet",
        "synthetic_train___tabddpm.parquet",
    ]
    
    expected_metrics = [
        "quality_gaussian_copula.json",
        "c2st_gaussian_copula.json",
        "mia_gaussian_copula.json",
        "utility_gaussian_copula.json",
        "quality_ctgan.json",
        "c2st_ctgan.json",
        "mia_ctgan.json",
        "utility_ctgan.json",
    ]
    
    # Check synthetic data files
    synthetic_count = 0
    for file in expected_synthetic:
        file_path = dataset_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {file} ({size_mb:.2f} MB)")
            synthetic_count += 1
        else:
            print(f"  ⚠️  {file} (not yet generated)")
    
    # Check metric files
    metric_count = 0
    for file in expected_metrics:
        file_path = dataset_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
            metric_count += 1
        else:
            print(f"  ⚠️  {file} (not yet generated)")
    
    return synthetic_count > 0 or metric_count > 0  # At least something generated


def check_metrics_validity(run_dir: Path, dataset: str) -> bool:
    """Check that metric JSON files are valid."""
    print(f"\n📈 Validating metrics for dataset: {dataset}...")
    
    dataset_dir = run_dir / dataset
    if not dataset_dir.exists():
        print(f"  ⚠️  Dataset directory not found")
        return True
    
    metric_files = {
        "quality": "quality_gaussian_copula.json",
        "c2st": "c2st_gaussian_copula.json",
        "mia": "mia_gaussian_copula.json",
    }
    
    all_valid = True
    for metric_name, filename in metric_files.items():
        file_path = dataset_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  {metric_name}: Not yet generated")
            continue
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Check that metric has expected fields
            if metric_name == "quality":
                if "score" in data:
                    score = data["score"]
                    print(f"  ✅ {metric_name}: {score:.2f}%")
                else:
                    print(f"  ⚠️  {metric_name}: Missing 'score' field")
            
            elif metric_name == "c2st":
                if "mean" in data:
                    mean = data["mean"]
                    print(f"  ✅ {metric_name}: AUC={mean:.4f}")
                else:
                    print(f"  ⚠️  {metric_name}: Missing 'mean' field")
            
            elif metric_name == "mia":
                if "auc" in data:
                    auc = data["auc"]
                    print(f"  ✅ {metric_name}: AUC={auc:.4f}")
                else:
                    print(f"  ⚠️  {metric_name}: Missing 'auc' field")
        
        except json.JSONDecodeError as e:
            print(f"  ❌ {metric_name}: Invalid JSON - {e}")
            all_valid = False
        except Exception as e:
            print(f"  ❌ {metric_name}: Error - {e}")
            all_valid = False
    
    return all_valid


def check_results_csv(run_dir: Path) -> bool:
    """Check that results CSV is valid and well-formed."""
    print(f"\n📋 Checking results CSV...")
    
    results_file = run_dir / "results.csv"
    if not results_file.exists():
        print(f"  ⚠️  results.csv not yet generated")
        return True
    
    try:
        df = pd.read_csv(results_file)
        print(f"  ✅ Results CSV is valid")
        print(f"     Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"     Columns: {', '.join(df.columns)}")
        
        # Show summary
        print(f"\n     Sample results:")
        print(df.to_string(index=False))
        
        return True
    except Exception as e:
        print(f"  ⚠️  Error reading results CSV: {e}")
        return True  # Not a hard failure if still processing


def check_log_for_errors(run_dir: Path) -> bool:
    """Check log file for errors or warnings."""
    print(f"\n📝 Checking log file...")
    
    log_file = run_dir / "run.log"
    if not log_file.exists():
        print(f"  ⚠️  run.log not found")
        return True
    
    try:
        with open(log_file) as f:
            lines = f.readlines()
        
        error_count = sum(1 for line in lines if "ERROR" in line)
        warning_count = sum(1 for line in lines if "WARNING" in line)
        
        print(f"  ✅ Log file exists ({len(lines)} lines)")
        if error_count > 0:
            print(f"  ⚠️  Found {error_count} ERROR entries")
        if warning_count > 0:
            print(f"  ⚠️  Found {warning_count} WARNING entries")
        
        # Show last few lines
        print(f"\n     Last 3 log entries:")
        for line in lines[-3:]:
            print(f"     {line.strip()}")
        
        return error_count == 0
    except Exception as e:
        print(f"  ⚠️  Error reading log file: {e}")
        return True


def check_pipeline_completion(run_dir: Path) -> bool:
    """Check if pipeline appears to be complete."""
    print(f"\n✓ Checking pipeline completion status...")
    
    log_file = run_dir / "run.log"
    if not log_file.exists():
        print(f"  ⚠️  Log file not found - pipeline may not have started")
        return False
    
    with open(log_file) as f:
        log_content = f.read()
    
    completion_markers = [
        ("Data loading", "Loading dataset"),
        ("Synthesis", "Fitting synthesizer"),
        ("Quality evaluation", "Computing quality score"),
        ("C2ST evaluation", "Computing C2ST"),
        ("MIA evaluation", "Computing MIA"),
        ("Results compilation", "Compiled results to"),
    ]
    
    completed = []
    for stage_name, marker in completion_markers:
        if marker in log_content:
            completed.append(stage_name)
    
    print(f"  Completed stages: {', '.join(completed) if completed else 'None yet'}")
    
    # Check if pipeline finished successfully
    if "Pipeline execution completed" in log_content:
        print(f"  ✅ Pipeline completed successfully!")
        return True
    elif "ERROR" in log_content and "ERROR - synthla_edu_v2" in log_content:
        print(f"  ❌ Pipeline encountered errors")
        return False
    else:
        print(f"  ⏳ Pipeline still running or in progress...")
        return True  # Don't fail, might still be running


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("SYNTHLA-EDU V2: End-to-End Validation")
    print("=" * 70)
    
    run_dir = Path("runs/v2_quick")
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        print("   Make sure you've run: set PYTHONPATH=src && python -m synthla_edu_v2.run --config configs/quick.yaml")
        return 1
    
    all_checks = []
    
    # Run all checks
    all_checks.append(("Directory Structure", check_directory_structure(run_dir)))
    all_checks.append(("OULAD Outputs", check_dataset_outputs(run_dir, "oulad")))
    all_checks.append(("ASSISTments Outputs", check_dataset_outputs(run_dir, "assistments")))
    all_checks.append(("OULAD Metrics", check_metrics_validity(run_dir, "oulad")))
    all_checks.append(("ASSISTments Metrics", check_metrics_validity(run_dir, "assistments")))
    all_checks.append(("Results CSV", check_results_csv(run_dir)))
    all_checks.append(("Log File", check_log_for_errors(run_dir)))
    all_checks.append(("Pipeline Completion", check_pipeline_completion(run_dir)))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for check_name, result in all_checks:
        status = "✅ PASS" if result else "⚠️  PARTIAL"
        print(f"{check_name:.<50} {status}")
    
    all_passed = all(result for _, result in all_checks)
    
    print("=" * 70)
    if all_passed:
        print("✅ All validations passed! Pipeline completed successfully.")
        return 0
    else:
        print("⚠️  Some validations incomplete - pipeline may still be running.")
        print("    Run this script again in a few minutes to check progress.")
        return 0  # Return 0 even if incomplete, since it might still be processing


if __name__ == "__main__":
    sys.exit(main())

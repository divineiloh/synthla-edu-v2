#!/usr/bin/env python3
"""
Regenerate visualizations from existing experiment results without rerunning experiments.

This script loads the JSON results and parquet data files from a completed --run-all
execution and regenerates all 10 publication figures. Useful for:
- Adjusting figure aesthetics during paper revision
- Regenerating specific figures with different parameters
- Creating additional visualizations from the same experimental data

Usage:
    python regenerate_figures.py --results-dir runs

The script expects this directory structure:
    runs/
        oulad/
            results.json       # Metrics from all 3 synthesizers
            data.parquet       # Combined real + synthetic data
        assistments/
            results.json
            data.parquet
        figures/              # Output directory (will be created/cleaned)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import argparse

# Import the visualization function from the main module
from synthla_edu_v2 import create_cross_dataset_visualizations, ensure_dir, remove_glob


def load_results_from_disk(results_dir: Path) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, pd.DataFrame],
    Dict[str, Dict[str, pd.DataFrame]]
]:
    """
    Load previously saved experiment results from disk.
    
    Args:
        results_dir: Base directory containing dataset subdirectories
        
    Returns:
        Tuple of (all_dataset_results, all_train_data, all_synthetic_data)
    """
    
    datasets = ["oulad", "assistments"]
    synthesizers = ["gaussian_copula", "ctgan", "tabddpm"]
    
    all_dataset_results = {}
    all_train_data = {}
    all_synthetic_data = {}
    
    for dataset in datasets:
        dataset_dir = results_dir / dataset
        
        # Load results JSON
        results_file = dataset_dir / "results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Missing results file: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        all_dataset_results[dataset] = results
        
        # Load combined data parquet
        data_file = dataset_dir / "data.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"Missing data file: {data_file}")
        
        data_df = pd.read_parquet(data_file)
        
        # Extract real training data (split=="real_train", synthesizer=="real")
        train_df = data_df[(data_df["split"] == "real_train") & (data_df["synthesizer"] == "real")].copy()
        # Drop metadata columns
        train_df = train_df.drop(columns=["split", "synthesizer"])
        all_train_data[dataset] = train_df
        
        # Extract synthetic datasets for each synthesizer
        synthetic_datasets = {}
        for synth_name in synthesizers:
            # Get rows for this synthesizer with split=="synthetic_train"
            synth_df = data_df[(data_df["split"] == "synthetic_train") & (data_df["synthesizer"] == synth_name)].copy()
            if len(synth_df) > 0:
                synth_df = synth_df.drop(columns=["split", "synthesizer"])
                synthetic_datasets[synth_name] = synth_df
        
        if len(synthetic_datasets) == 0:
            raise ValueError(f"No synthetic data found for dataset: {dataset}")
        
        all_synthetic_data[dataset] = synthetic_datasets
        
        print(f"[{dataset.upper()}] Loaded:")
        print(f"  - Results: {len(results['synthesizers'])} synthesizers")
        print(f"  - Train data: {len(train_df):,} rows")
        print(f"  - Synthetic data: {', '.join(f'{k}={len(v):,}' for k, v in synthetic_datasets.items())}")
    
    return all_dataset_results, all_train_data, all_synthetic_data


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate visualizations from existing experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory containing oulad/ and assistments/ subdirectories (default: runs)"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        print("\nExpected directory structure:", file=sys.stderr)
        print("  runs/", file=sys.stderr)
        print("    oulad/", file=sys.stderr)
        print("      results.json", file=sys.stderr)
        print("      data.parquet", file=sys.stderr)
        print("    assistments/", file=sys.stderr)
        print("      results.json", file=sys.stderr)
        print("      data.parquet", file=sys.stderr)
        sys.exit(1)
    
    print("="*70)
    print("SYNTHLA-EDU V2: Visualization Regeneration")
    print("="*70)
    print(f"Results directory: {results_dir.absolute()}")
    print()
    
    # Load data from disk
    print("Loading experiment results from disk...")
    try:
        all_results, all_train_data, all_synthetic_data = load_results_from_disk(results_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\nAll data loaded successfully!")
    print()
    
    # Prepare figures directory
    figures_dir = ensure_dir(results_dir / "figures updated")
    print(f"Output directory: {figures_dir.absolute()}")
    print()
    
    # Clean previous figures
    print("Cleaning previous figures...")
    remove_glob(figures_dir, "fig*.png")
    print()
    
    # Generate visualizations
    print("Generating 11 cross-dataset visualizations...")
    print("-"*70)
    figure_paths = create_cross_dataset_visualizations(
        figures_dir,
        all_results,
        all_train_data,
        all_synthetic_data
    )
    print()
    
    # Summary
    print("="*70)
    print("REGENERATION COMPLETE")
    print("="*70)
    print(f"Generated {len(figure_paths)} figures:")
    for fig_path in figure_paths:
        size_kb = fig_path.stat().st_size / 1024
        print(f"  - {fig_path.name} ({size_kb:.0f} KB)")
    print()
    print("These figures are publication-ready and identical to those generated")
    print("by --run-all, but regenerated without rerunning the experiments.")
    print("="*70)


if __name__ == "__main__":
    main()

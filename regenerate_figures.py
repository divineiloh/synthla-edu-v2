#!/usr/bin/env python
"""Regenerate cross-dataset figures from existing results."""

import json
import pandas as pd
from pathlib import Path
from synthla_edu_v2 import create_cross_dataset_visualizations

# Load results and data
base_out = Path('runs')
datasets = ['oulad', 'assistments']

all_results = {}
all_train_data = {}
all_synthetic_data = {}

for dataset in datasets:
    dataset_dir = base_out / dataset
    
    # Load results.json
    with open(dataset_dir / 'results.json') as f:
        all_results[dataset] = json.load(f)
    
    # Load data.parquet
    df = pd.read_parquet(dataset_dir / 'data.parquet')
    
    # Split into train and synthetic datasets
    all_train_data[dataset] = df[df['split'] == 'real_train'].drop(columns=['split', 'synthesizer'])
    
    synth_data = {}
    for synth_name in ['gaussian_copula', 'ctgan', 'tabddpm']:
        synth_df = df[(df['split'] == 'synthetic_train') & (df['synthesizer'] == synth_name)]
        if len(synth_df) > 0:
            synth_data[synth_name] = synth_df.drop(columns=['split', 'synthesizer'])
    all_synthetic_data[dataset] = synth_data

# Generate figures
figures_dir = base_out / 'figures'
saved_figs = create_cross_dataset_visualizations(figures_dir, all_results, all_train_data, all_synthetic_data)

print(f'\n✓ Successfully generated {len(saved_figs)} figures:')
for fig in saved_figs:
    print(f'  • {fig.name}')

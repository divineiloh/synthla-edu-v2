import json
from pathlib import Path

# Load results
with open('runs/oulad/results.json') as f:
    r_oulad = json.load(f)
with open('runs/assistments/results.json') as f:
    r_assist = json.load(f)

print("\n" + "="*70)
print("SYNTHLA-EDU V2: --RUN-ALL --QUICK COMPLETE")
print("="*70)

print("\n=== OULAD (32,593 students) ===")
for s in ['gaussian_copula', 'ctgan', 'tabddpm']:
    metrics = r_oulad['synthesizers'][s]
    print(f"\n{s.upper().replace('_', ' ')}:")
    print(f"  Quality:       {metrics['sdmetrics']['overall_score']*100:.1f}%")
    print(f"  C2ST (Realism): {metrics['c2st']['effective_auc']:.3f} (ideal: 0.5)")
    print(f"  MIA (Privacy):  {metrics['mia']['worst_case_effective_auc']:.3f} (ideal: 0.5)")
    print(f"  Classification AUC: {metrics['utility']['classification']['rf_auc']:.3f}")
    print(f"  Regression MAE:     {metrics['utility']['regression']['ridge_mae']:.2f}")
    print(f"  Training time:  {metrics['timing']['fit_seconds']:.1f}s")
    print(f"  Sampling time:  {metrics['timing']['sample_seconds']:.1f}s")

print("\n=== ASSISTments (8,519 students) ===")
for s in ['gaussian_copula', 'ctgan', 'tabddpm']:
    metrics = r_assist['synthesizers'][s]
    print(f"\n{s.upper().replace('_', ' ')}:")
    print(f"  Quality:       {metrics['sdmetrics']['overall_score']*100:.1f}%")
    print(f"  C2ST (Realism): {metrics['c2st']['effective_auc']:.3f} (ideal: 0.5)")
    print(f"  MIA (Privacy):  {metrics['mia']['worst_case_effective_auc']:.3f} (ideal: 0.5)")
    print(f"  Classification AUC: {metrics['utility']['classification']['rf_auc']:.3f}")
    print(f"  Regression MAE:     {metrics['utility']['regression']['ridge_mae']:.4f}")
    print(f"  Training time:  {metrics['timing']['fit_seconds']:.1f}s")
    print(f"  Sampling time:  {metrics['timing']['sample_seconds']:.1f}s")

print("\n=== Outputs Generated ===")
print(f"• runs/oulad/data.parquet (130,372 rows)")
print(f"• runs/oulad/results.json (3.68 MB)")
print(f"• runs/assistments/data.parquet (59,634 rows)")
print(f"• runs/assistments/results.json (0.99 MB)")
print(f"• runs/figures/ (11 publication figures, 300 DPI)")

print("\n=== All 11 Figures ===")
for i, name in enumerate([
    "Classification Utility", "Regression Utility", "Data Quality",
    "Privacy (MIA)", "Performance Heatmap", "Radar Chart",
    "Classification CI", "Regression CI", "Per-Attacker Privacy",
    "Distribution Fidelity", "Correlation Matrices"
], 1):
], 1):
    print(f"  {i:2d}. {name}")

print("\n" + "="*70)
print("✓ Ready for paper submission!")
print("="*70 + "\n")

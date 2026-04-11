#!/usr/bin/env python
"""
generate_revised_heatmaps.py
============================
Regenerates fig16 (OULAD) and fig17 (ASSISTments) multi-objective
performance heatmaps with a corrected classification-AUC normalisation.

The fix (addressing reviewer comment R2-4):
    The original `generate_all_figures.py` normalises classification AUC
    as (TSTR / TRTR) × 100.  When TSTR is at or below chance (≈ 0.50),
    this formula produces scores of ≈ 60 % for CTGAN on ASSISTments
    (raw AUC = 0.498), making it appear visually competitive despite
    performing no better than random guessing.

    The revised formula floors at 0.50 (chance level) so the score
    reflects performance above chance:

        score = max(0, (TSTR − 0.50) / (TRTR − 0.50)) × 100

    Any TSTR ≤ 0.50  →  0 %.
    TSTR == TRTR     →  100 %.

    All other axes (Quality, Realism, Privacy, Regr. MAE) are unchanged.

Output:
    runs_publication/figures_aggregated/fig16_revised.png
    runs_publication/figures_aggregated/fig17_revised.png

Usage:
    python scripts/generate_revised_heatmaps.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "runs_publication"
OUT_DIR = BASE_DIR / "figures_aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["oulad", "assistments"]
SYNTHESIZERS = ["gaussian_copula", "ctgan", "tabddpm"]

# ── Matplotlib defaults (mirror generate_all_figures.py) ────────────
plt.rcParams.update({
    "figure.dpi": 1200,
    "savefig.dpi": 1200,
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


# ── Scoring helpers ──────────────────────────────────────────────────
def _invert_score(val: float) -> float:
    """Effective AUC (ideal 0.50) → 0–100 score.  Unchanged from original."""
    return max(0.0, (1.0 - (val - 0.5) / 0.5)) * 100


def _cls_auc_score(tstr: float, trtr: float) -> float:
    """
    Classification AUC → 0–100 score, floored at chance (0.50).

    Addresses R2-4: a synthesizer whose TSTR AUC is at or below 0.50
    (no better than random) should score 0 %, not ~60 %.

    Formula:
        effective_range = TRTR − 0.50          # useful range above chance
        score = max(0, (TSTR − 0.50) / effective_range) × 100
    """
    effective_range = trtr - 0.5
    if effective_range <= 0:
        return 0.0
    return max(0.0, (tstr - 0.5) / effective_range) * 100


def _mae_score(tstr_mae: float, trtr_mae: float) -> float:
    """Regression MAE → 0–100 score (lower MAE = better).  Unchanged."""
    if tstr_mae <= 0:
        return 100.0
    return min(100.0, (trtr_mae / tstr_mae) * 100)


# ── Data loading ─────────────────────────────────────────────────────
def load_all_results() -> dict:
    data: dict = {}
    for seed in SEEDS:
        data[seed] = {}
        for ds in DATASETS:
            path = BASE_DIR / f"seed_{seed}" / ds / "results.json"
            with open(path) as f:
                raw = json.load(f)
            # Drop large per_sample arrays to save memory
            for syn in SYNTHESIZERS:
                raw["synthesizers"][syn].get("utility", {}).pop("per_sample", None)
            data[seed][ds] = raw
    return data


# ── Heatmap ──────────────────────────────────────────────────────────
def plot_revised_heatmap(data: dict, dataset: str, fig_num: int):
    columns = ["Quality", "Realism", "Privacy", "Cls. AUC", "Regr. MAE"]
    matrix = np.zeros((3, 5))

    for i, syn in enumerate(SYNTHESIZERS):
        # Col 0 – SDMetrics quality score
        vals = [data[s][dataset]["synthesizers"][syn]["sdmetrics"]["overall_score"]
                for s in SEEDS]
        matrix[i, 0] = np.mean(vals) * 100

        # Col 1 – Realism (C2ST, inverted)
        vals = [data[s][dataset]["synthesizers"][syn]["c2st"]["effective_auc"]
                for s in SEEDS]
        matrix[i, 1] = _invert_score(np.mean(vals))

        # Col 2 – Privacy (MIA worst-case, inverted)
        vals = [data[s][dataset]["synthesizers"][syn]["mia"]["worst_case_effective_auc"]
                for s in SEEDS]
        matrix[i, 2] = _invert_score(np.mean(vals))

        # Col 3 – Classification AUC (TSTR vs TRTR, chance-floored)  ← REVISED
        trtr_cls = np.mean([data[s][dataset]["synthesizers"][syn]["utility"]["classification"]["trtr_rf_auc"]
                            for s in SEEDS])
        tstr_cls = np.mean([data[s][dataset]["synthesizers"][syn]["utility"]["classification"]["rf_auc"]
                            for s in SEEDS])
        matrix[i, 3] = _cls_auc_score(tstr_cls, trtr_cls)

        # Col 4 – Regression MAE (lower is better)
        trtr_mae = np.mean([data[s][dataset]["synthesizers"][syn]["utility"]["regression"]["trtr_rf_mae"]
                            for s in SEEDS])
        tstr_mae = np.mean([data[s][dataset]["synthesizers"][syn]["utility"]["regression"]["rf_mae"]
                            for s in SEEDS])
        matrix[i, 4] = _mae_score(tstr_mae, trtr_mae)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    for i in range(3):
        for j in range(5):
            v = matrix[i, j]
            tc = "white" if v < 40 else "black"
            ax.text(j, i, f"{v:.1f}%",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=tc)

    ax.set_xticks(range(5))
    ax.set_xticklabels(columns, fontsize=13)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Gaussian\nCopula", "CTGAN", "TabDDPM"], fontsize=14)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label("Score (%)", fontsize=14, fontweight="bold")

    fig.tight_layout()

    out = OUT_DIR / f"fig{fig_num}_revised.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")

    # Print matrix for reference
    ds_label = {"oulad": "OULAD", "assistments": "ASSISTments"}[dataset]
    print(f"\n  {ds_label} score matrix (revised):")
    synth_names = ["Gaussian Copula", "CTGAN", "TabDDPM"]
    header = f"  {'':20s}" + "".join(f"  {c:>10s}" for c in columns)
    print(header)
    for i, sn in enumerate(synth_names):
        row = f"  {sn:20s}" + "".join(f"  {matrix[i, j]:>9.1f}%" for j in range(5))
        print(row)


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading results …")
    data = load_all_results()
    print("OK\n")

    print("Generating fig16_revised.png (OULAD) …")
    plot_revised_heatmap(data, "oulad", 16)

    print("\nGenerating fig17_revised.png (ASSISTments) …")
    plot_revised_heatmap(data, "assistments", 17)

    print("\nDone.  Files written to:", OUT_DIR)

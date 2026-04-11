#!/usr/bin/env python
"""
generate_timing_table.py
========================
Extracts fit (training) and sample (generation) timing from each
seed's results.json and produces:

    runs_publication/figures_aggregated/timing_breakdown.csv
        Per-synthesizer, per-dataset mean ± SD for fit_seconds,
        sample_seconds, and total_seconds across the 5 seeds.

    runs_publication/figures_aggregated/fig_timing.png
        Grouped bar chart of fit vs. sample time, split by dataset.

Addresses reviewer comment R1-G:
    "Runtime measurements are reported only at a high level without
    detailed profiling of training versus sampling costs."

Usage:
    python scripts/generate_timing_table.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "runs_publication"
OUT_DIR = BASE_DIR / "figures_aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["oulad", "assistments"]
SYNTHESIZERS = ["gaussian_copula", "ctgan", "tabddpm"]

SYNTH_LABELS = {
    "gaussian_copula": "Gaussian Copula",
    "ctgan": "CTGAN",
    "tabddpm": "TabDDPM",
}
DATASET_LABELS = {"oulad": "OULAD", "assistments": "ASSISTments"}

# Color-blind friendly palette (same as generate_all_figures.py)
C_SYNTH = {
    "gaussian_copula": "#DE8F05",
    "ctgan": "#029E73",
    "tabddpm": "#CC78BC",
}

# ── Matplotlib defaults ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 1200,
    "savefig.dpi": 1200,
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


# ── Data extraction ──────────────────────────────────────────────────
def collect_timing():
    """Return a dict: timing[ds][syn] = {"fit": [...], "sample": [...], "total": [...]}"""
    timing: dict[str, dict[str, dict[str, list[float]]]] = {
        ds: {
            syn: {"fit": [], "sample": [], "total": []}
            for syn in SYNTHESIZERS
        }
        for ds in DATASETS
    }
    for seed in SEEDS:
        for ds in DATASETS:
            path = BASE_DIR / f"seed_{seed}" / ds / "results.json"
            with open(path) as f:
                raw = json.load(f)
            for syn in SYNTHESIZERS:
                t = raw["synthesizers"][syn]["timing"]
                timing[ds][syn]["fit"].append(t["fit_seconds"])
                timing[ds][syn]["sample"].append(t["sample_seconds"])
                timing[ds][syn]["total"].append(t["total_seconds"])
    return timing


# ── CSV table ────────────────────────────────────────────────────────
def build_csv(timing: dict) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for syn in SYNTHESIZERS:
            t = timing[ds][syn]
            rows.append({
                "Dataset": DATASET_LABELS[ds],
                "Synthesizer": SYNTH_LABELS[syn],
                "Fit Mean (s)": round(np.mean(t["fit"]), 2),
                "Fit SD (s)": round(np.std(t["fit"]), 2),
                "Sample Mean (s)": round(np.mean(t["sample"]), 2),
                "Sample SD (s)": round(np.std(t["sample"]), 2),
                "Total Mean (s)": round(np.mean(t["total"]), 2),
                "Total SD (s)": round(np.std(t["total"]), 2),
            })
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "timing_breakdown.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path.name}")
    return df


# ── Figure ───────────────────────────────────────────────────────────
def plot_timing(timing: dict):
    """
    2-panel grouped bar chart (one panel per dataset).
    Each group = one synthesizer; stacked-style bars: fit (solid) +
    sample (hatched) drawn side by side with error bars for total.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ds in zip(axes, DATASETS):
        n = len(SYNTHESIZERS)
        x = np.arange(n)
        bw = 0.32

        fit_means = [np.mean(timing[ds][syn]["fit"]) for syn in SYNTHESIZERS]
        fit_sds = [np.std(timing[ds][syn]["fit"]) for syn in SYNTHESIZERS]
        smp_means = [np.mean(timing[ds][syn]["sample"]) for syn in SYNTHESIZERS]
        smp_sds = [np.std(timing[ds][syn]["sample"]) for syn in SYNTHESIZERS]

        # Fit bars
        bars_fit = ax.bar(
            x - bw / 2, fit_means, bw, yerr=fit_sds,
            capsize=4, label="Training (fit)",
            color=[C_SYNTH[s] for s in SYNTHESIZERS],
            alpha=0.9, error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
        )
        # Sample bars (hatched)
        ax.bar(
            x + bw / 2, smp_means, bw, yerr=smp_sds,
            capsize=4, label="Sampling",
            color=[C_SYNTH[s] for s in SYNTHESIZERS],
            alpha=0.55, hatch="//",
            error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
        )

        ax.set_title(DATASET_LABELS[ds], fontweight="bold")
        ax.set_ylabel("Wall-clock time (seconds)")
        ax.set_xticks(x)
        ax.set_xticklabels([SYNTH_LABELS[s] for s in SYNTHESIZERS], rotation=15, ha="right")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate totals above each pair
        for xi, syn in enumerate(SYNTHESIZERS):
            total_m = np.mean(timing[ds][syn]["total"])
            total_sd = np.std(timing[ds][syn]["total"])
            ax.annotate(
                f"{total_m:.0f}±{total_sd:.0f}s",
                xy=(xi, max(fit_means[xi] + fit_sds[xi],
                            smp_means[xi] + smp_sds[xi])),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, color="#444444",
            )

    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888888", alpha=0.9, label="Training (fit)"),
        Patch(facecolor="#888888", alpha=0.55, hatch="//", label="Sampling"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False, fontsize=13,
    )

    fig.suptitle(
        "Training vs. Sampling Wall-Clock Time (CPU-only, mean ± SD, 5 seeds)",
        fontsize=14, y=1.07,
    )
    fig.tight_layout()

    out = OUT_DIR / "fig_timing.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Collecting timing data …")
    timing = collect_timing()

    print("Writing CSV table …")
    df = build_csv(timing)
    print(df.to_string(index=False))

    print("\nGenerating figure …")
    plot_timing(timing)

    print("\nDone.")

#!/usr/bin/env python
"""
generate_all_figures.py
=======================
Single entry point that generates ALL publication figures for the
Synthla-Edu V2 manuscript (fig2 – fig17).

Figure mapping (manuscript numbering):
    fig2  – OULAD classification utility (TSTR bar chart)
    fig3  – ASSISTments classification utility
    fig4  – OULAD regression utility (TSTR bar chart)
    fig5  – ASSISTments regression utility
    fig6  – SDMetrics quality scores (grouped bar)
    fig7  – MIA worst-case effective AUC (grouped bar)
    fig8  – OULAD MIA per-attacker (grouped bar)
    fig9  – ASSISTments MIA per-attacker
    fig10 – OULAD SHAP feature importance – Classification (top 7)
    fig11 – OULAD SHAP feature importance – Regression (top 7)
    fig12 – ASSISTments SHAP feature importance – Classification (all)
    fig13 – ASSISTments SHAP feature importance – Regression (all)
    fig14 – SHAP beeswarm: OULAD Classification – TRTR
    fig15 – SHAP beeswarm: OULAD Classification – TSTR (TabDDPM)
    fig16 – OULAD multi-objective performance heatmap
    fig17 – ASSISTments multi-objective performance heatmap

Output: runs_publication/figures_aggregated/

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --skip-beeswarm   # skip fig14/fig15
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
import matplotlib.text
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# PATHS & GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent / "runs_publication"
OUT_DIR = BASE_DIR / "figures_aggregated"
CACHE_DIR = BASE_DIR / "shap_cache"

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["oulad", "assistments"]
SYNTHESIZERS = ["gaussian_copula", "ctgan", "tabddpm"]

SYNTH_LABELS = {
    "gaussian_copula": "Gaussian Copula",
    "ctgan": "CTGAN",
    "tabddpm": "TabDDPM",
}
DATASET_LABELS = {"oulad": "OULAD", "assistments": "ASSISTments"}

ATTACKERS = ["logistic_regression", "random_forest", "xgboost"]
ATTACKER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

TOP_N_SHAP = 7  # OULAD SHAP bars (fig10, fig11)

# ─── Color-blind friendly palette ────────────────────────────────────
C_REAL = "#999999"
C_SYNTH = {
    "gaussian_copula": "#DE8F05",
    "ctgan": "#029E73",
    "tabddpm": "#CC78BC",
}
C_DATASET = {"oulad": "#0173B2", "assistments": "#E69F00"}
C_ATTACKER = {
    "logistic_regression": "#0173B2",
    "random_forest": "#E69F00",
    "xgboost": "#029E73",
}

# ─── Human-readable feature-name mappings ────────────────────────────
OULAD_NUM = {
    "x1": "Prev. Attempts",
    "x2": "Studied Credits",
    "x3": "Registration Date",
    "x4": "Unregistration Date",
    "x5": "Is Unregistered",
    "x6": "Total VLE Clicks",
    "x7": "VLE Records",
    "x8": "VLE Sites",
    "x9": "VLE Active Days",
    "x10": "Mean VLE Clicks",
    "x11": "Clicks/Active Day",
    "x12": "Assessments Submitted",
}
OULAD_CAT = {
    "x0": "Course Module",
    "x1": "Presentation",
    "x2": "Gender",
    "x3": "Region",
    "x4": "Education Level",
    "x5": "IMD Band",
    "x6": "Age Band",
    "x7": "Disability",
}
ASSIST_NUM = {
    "x1": "Unique Skills",
    "x2": "Hint Rate",
    "x3": "Avg. Attempts",
    "x4": "Avg. Response Time",
}

# ─── Beeswarm configuration ──────────────────────────────────────────
BEESWARM_MAX_DISPLAY = 10
BEESWARM_N_SHAP_SAMPLES = 50   # per seed
BEESWARM_N_ESTIMATORS = 30     # RF trees

# OULAD schema for beeswarm
ID_COLS = ["code_module", "code_presentation", "id_student"]
TARGET_COLS = ["dropout", "final_grade"]
CLASS_TARGET = "dropout"
CATEGORICAL_COLS = [
    "code_module", "code_presentation", "gender", "region",
    "highest_education", "imd_band", "age_band", "disability",
]


# ═══════════════════════════════════════════════════════════════════════
# MATPLOTLIB SETUP
# ═══════════════════════════════════════════════════════════════════════
def setup_rcparams():
    """Apply publication-quality matplotlib defaults."""
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


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING  (fig2–fig13, fig16–fig17)
# ═══════════════════════════════════════════════════════════════════════
def load_all_results():
    """Load results.json from all seeds and datasets."""
    data = {}
    for seed in SEEDS:
        data[seed] = {}
        for ds in DATASETS:
            path = BASE_DIR / f"seed_{seed}" / ds / "results.json"
            print(f"  Loading seed {seed} / {ds} … ", end="", flush=True)
            with open(path, "r") as f:
                raw = json.load(f)
            for synth in SYNTHESIZERS:
                if synth in raw.get("synthesizers", {}):
                    raw["synthesizers"][synth].get("utility", {}).pop(
                        "per_sample", None
                    )
            data[seed][ds] = raw
            print("OK")
    return data


# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS  (fig2–fig13, fig16–fig17)
# ═══════════════════════════════════════════════════════════════════════
def _savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _invert_score(val):
    """Effective AUC (ideal 0.5) → 0-100 score."""
    return max(0.0, (1.0 - (val - 0.5) / 0.5)) * 100


def _utility_score(synth_val, real_val, lower_better=False):
    """Utility metric → 0-100 score (100 = matches real baseline)."""
    if lower_better:
        return min(100.0, (real_val / synth_val) * 100) if synth_val > 0 else 100.0
    return min(100.0, (synth_val / real_val) * 100) if real_val > 0 else 100.0


def _place_bar_labels(ax, bars, means, sds, baseline, fmt, yrange, fontsize=13):
    """Place data labels above bars, or inside if they would overlap the baseline."""
    zone = yrange * 0.04
    offset = yrange * 0.012

    for bar, m, sd in zip(bars, means, sds):
        cx = bar.get_x() + bar.get_width() / 2
        bar_top = m + sd
        above_y = bar_top + offset

        overlap = False
        if baseline is not None:
            text_lo = above_y - zone * 0.2
            text_hi = above_y + zone
            overlap = (baseline >= text_lo) and (baseline <= text_hi)

        if overlap and m > yrange * 0.1:
            inside_y = m - yrange * 0.02
            ax.text(cx, inside_y, f"{m:{fmt}}",
                    ha="center", va="top", fontsize=fontsize - 1,
                    fontweight="bold", color="white", zorder=6)
        else:
            ax.text(cx, above_y, f"{m:{fmt}}",
                    ha="center", va="bottom", fontsize=fontsize,
                    fontweight="bold", zorder=6)


def _aggregate_oulad_shap(raw_imp):
    """Merge OULAD one-hot features to parent categories, use human names."""
    result = {}
    cat_sums = {}
    for feat, val in raw_imp.items():
        if "_" in feat:
            prefix = feat.split("_")[0]
            cat_sums[prefix] = cat_sums.get(prefix, 0.0) + val
        else:
            if feat == "x0":
                continue
            result[OULAD_NUM.get(feat, feat)] = val
    for prefix, total in cat_sums.items():
        result[OULAD_CAT.get(prefix, f"cat_{prefix}")] = total
    return result


def _aggregate_assist_shap(raw_imp):
    """Rename ASSISTments features, exclude user_id."""
    result = {}
    for feat, val in raw_imp.items():
        if feat == "x0":
            continue
        result[ASSIST_NUM.get(feat, feat)] = val
    return result


def get_shap_importance(data, dataset, synth, task, source="tstr"):
    key = f"{source}_importance"
    all_imp = []
    for seed in SEEDS:
        imp = data[seed][dataset]["synthesizers"][synth]["shap"][task][key]
        all_imp.append(imp)
    all_keys = set()
    for imp in all_imp:
        all_keys.update(imp.keys())
    mean_imp = {k: np.mean([d.get(k, 0.0) for d in all_imp]) for k in all_keys}
    if dataset == "oulad":
        return _aggregate_oulad_shap(mean_imp)
    return _aggregate_assist_shap(mean_imp)


def get_trtr_importance(data, dataset, task):
    all_imp = []
    for seed in SEEDS:
        for synth in SYNTHESIZERS:
            imp = data[seed][dataset]["synthesizers"][synth]["shap"][task][
                "trtr_importance"
            ]
            all_imp.append(imp)
    all_keys = set()
    for imp in all_imp:
        all_keys.update(imp.keys())
    mean_imp = {k: np.mean([d.get(k, 0.0) for d in all_imp]) for k in all_keys}
    if dataset == "oulad":
        return _aggregate_oulad_shap(mean_imp)
    return _aggregate_assist_shap(mean_imp)


# ═══════════════════════════════════════════════════════════════════════
# FIG 2–3: Classification Utility
# ═══════════════════════════════════════════════════════════════════════
def plot_cls_utility(data, dataset, fig_num):
    fig, ax = plt.subplots(figsize=(8, 6))

    trtr = np.array(
        [data[s][dataset]["synthesizers"][SYNTHESIZERS[0]]["utility"][
            "classification"]["trtr_rf_auc"] for s in SEEDS]
    )
    synth_vals = {
        syn: np.array([data[s][dataset]["synthesizers"][syn]["utility"][
            "classification"]["rf_auc"] for s in SEEDS])
        for syn in SYNTHESIZERS
    }

    labels = ["Real\n(TRTR)", "Gaussian\nCopula", "CTGAN", "TabDDPM"]
    means = [trtr.mean()] + [synth_vals[s].mean() for s in SYNTHESIZERS]
    sds = [trtr.std()] + [synth_vals[s].std() for s in SYNTHESIZERS]
    colors = [C_REAL] + [C_SYNTH[s] for s in SYNTHESIZERS]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, zorder=3,
                  error_kw=dict(lw=1.5, capthick=1.2))

    bl = means[0]
    x_ext = 0.35
    ax.plot([-x_ext, len(labels) - 1 + x_ext], [bl, bl],
            color="red", linestyle="--", linewidth=1.8, zorder=5,
            label=f"TRTR Baseline ({bl:.3f})")

    ymax = max(m + sd for m, sd in zip(means, sds)) + 0.06
    ylim_top = min(1.15, ymax)
    ax.set_ylim(0, ylim_top)

    _place_bar_labels(ax, bars, means, sds, bl, ".3f", ylim_top)

    ax.set_ylabel("AUC-ROC", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=13)
    ax.set_axisbelow(True)

    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4–5: Regression Utility
# ═══════════════════════════════════════════════════════════════════════
def plot_reg_utility(data, dataset, fig_num):
    fig, ax = plt.subplots(figsize=(8, 6))

    trtr = np.array(
        [data[s][dataset]["synthesizers"][SYNTHESIZERS[0]]["utility"][
            "regression"]["trtr_rf_mae"] for s in SEEDS]
    )
    synth_vals = {
        syn: np.array([data[s][dataset]["synthesizers"][syn]["utility"][
            "regression"]["rf_mae"] for s in SEEDS])
        for syn in SYNTHESIZERS
    }

    labels = ["Real\n(TRTR)", "Gaussian\nCopula", "CTGAN", "TabDDPM"]
    means = [trtr.mean()] + [synth_vals[s].mean() for s in SYNTHESIZERS]
    sds = [trtr.std()] + [synth_vals[s].std() for s in SYNTHESIZERS]
    colors = [C_REAL] + [C_SYNTH[s] for s in SYNTHESIZERS]
    fmt = ".2f" if dataset == "oulad" else ".3f"

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.8, zorder=3,
                  error_kw=dict(lw=1.5, capthick=1.2))

    bl = means[0]
    x_ext = 0.35
    ax.plot([-x_ext, len(labels) - 1 + x_ext], [bl, bl],
            color="red", linestyle="--", linewidth=1.8, zorder=5,
            label=f"TRTR Baseline ({bl:{fmt}})")

    ymax_data = max(m + sd for m, sd in zip(means, sds))
    yrange = ymax_data * 1.15
    ax.set_ylim(0, yrange)

    _place_bar_labels(ax, bars, means, sds, bl, fmt, yrange)

    ax.set_ylabel("Mean Absolute Error", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left" if max(means) == means[0] else "upper right",
              framealpha=0.9, fontsize=13)
    ax.set_axisbelow(True)

    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: SDMetrics Quality Scores
# ═══════════════════════════════════════════════════════════════════════
def plot_sdmetrics(data, fig_num):
    fig, ax = plt.subplots(figsize=(8, 6))
    bw = 0.35
    sp = 1.15
    ylim_top = 110
    offset = ylim_top * 0.012

    for j, ds in enumerate(DATASETS):
        pos, ms, ss = [], [], []
        for i, syn in enumerate(SYNTHESIZERS):
            vals = [data[s][ds]["synthesizers"][syn]["sdmetrics"][
                "overall_score"] for s in SEEDS]
            ms.append(np.mean(vals) * 100)
            ss.append(np.std(vals) * 100)
            pos.append(i * sp + (j - 0.5) * bw)
        bars = ax.bar(pos, ms, bw, yerr=ss, capsize=4,
                      color=C_DATASET[ds], edgecolor="black", linewidth=0.6,
                      label=DATASET_LABELS[ds], zorder=3,
                      error_kw=dict(lw=1.2, capthick=1))
        for bar, m, sd in zip(bars, ms, ss):
            bar_top = m + sd
            ax.text(bar.get_x() + bw / 2, bar_top + offset,
                    f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="black", zorder=6)

    ax.set_ylabel("Quality Score (%)", fontsize=15, fontweight="bold")
    ax.set_xticks([i * sp for i in range(len(SYNTHESIZERS))])
    ax.set_xticklabels([SYNTH_LABELS[s] for s in SYNTHESIZERS], fontsize=14)
    ax.set_ylim(0, ylim_top)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=13)
    ax.set_axisbelow(True)

    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: MIA Overall
# ═══════════════════════════════════════════════════════════════════════
def plot_mia_overall(data, fig_num):
    fig, ax = plt.subplots(figsize=(8, 6))
    bw = 0.35
    sp = 1.15

    for j, ds in enumerate(DATASETS):
        pos, ms, ss = [], [], []
        for i, syn in enumerate(SYNTHESIZERS):
            vals = [data[s][ds]["synthesizers"][syn]["mia"][
                "worst_case_effective_auc"] for s in SEEDS]
            ms.append(np.mean(vals))
            ss.append(np.std(vals))
            pos.append(i * sp + (j - 0.5) * bw)
        bars = ax.bar(pos, ms, bw, yerr=ss, capsize=4,
                      color=C_DATASET[ds], edgecolor="black", linewidth=0.6,
                      label=DATASET_LABELS[ds], zorder=3,
                      error_kw=dict(lw=1.2, capthick=1))
        for bar, m, sd in zip(bars, ms, ss):
            ax.text(bar.get_x() + bw / 2, m + max(sd, 0.003) + 0.006,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", zorder=6)

    ax.axhline(y=0.5, color="#029E73", linestyle="--", linewidth=2,
               alpha=0.8, label="Ideal Privacy (0.50)", zorder=5)

    ax.set_ylabel("MIA Effective AUC", fontsize=15, fontweight="bold")
    ax.set_xticks([i * sp for i in range(len(SYNTHESIZERS))])
    ax.set_xticklabels([SYNTH_LABELS[s] for s in SYNTHESIZERS], fontsize=14)
    ax.set_ylim(0.45, 0.60)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=13)
    ax.set_axisbelow(True)

    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 8–9: MIA Per-Attacker
# ═══════════════════════════════════════════════════════════════════════
def plot_mia_per_attacker(data, dataset, fig_num):
    fig, ax = plt.subplots(figsize=(8, 6))
    bw = 0.25
    sp = 1.15

    for j, atk in enumerate(ATTACKERS):
        pos, ms, ss = [], [], []
        for i, syn in enumerate(SYNTHESIZERS):
            vals = [data[s][dataset]["synthesizers"][syn]["mia"][
                "attackers"][atk]["effective_auc"] for s in SEEDS]
            ms.append(np.mean(vals))
            ss.append(np.std(vals))
            pos.append(i * sp + (j - 1) * bw)
        bars = ax.bar(pos, ms, bw, yerr=ss, capsize=3,
                      color=C_ATTACKER[atk], edgecolor="black", linewidth=0.6,
                      label=ATTACKER_LABELS[atk], zorder=3,
                      error_kw=dict(lw=1.2, capthick=1))
        for bar, m, sd in zip(bars, ms, ss):
            ax.text(bar.get_x() + bw / 2, m + max(sd, 0.003) + 0.008,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", zorder=6)

    ax.axhline(y=0.5, color="#029E73", linestyle="--", linewidth=2,
               alpha=0.8, label="Ideal Privacy (0.50)", zorder=5)

    ax.set_ylabel("MIA Effective AUC", fontsize=15, fontweight="bold")
    ax.set_xticks([i * sp for i in range(len(SYNTHESIZERS))])
    ax.set_xticklabels([SYNTH_LABELS[s] for s in SYNTHESIZERS], fontsize=14)
    ax.set_ylim(0.45, 0.60)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=13)
    ax.set_axisbelow(True)

    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 10–13: SHAP Feature Importance Bars
# ═══════════════════════════════════════════════════════════════════════
def plot_shap_bars(data, dataset, task, fig_num):
    """Single-panel SHAP bar chart for one dataset and one task."""
    top_n = TOP_N_SHAP if dataset == "oulad" else None

    trtr_imp = get_trtr_importance(data, dataset, task)
    tstr_imps = {
        syn: get_shap_importance(data, dataset, syn, task, "tstr")
        for syn in SYNTHESIZERS
    }

    sorted_feats = sorted(trtr_imp.keys(),
                          key=lambda f: trtr_imp[f], reverse=True)
    if top_n:
        sorted_feats = sorted_feats[:top_n]

    n_f = len(sorted_feats)
    fig_h = max(5, 0.7 * n_f + 1.5) if dataset == "oulad" else max(4.5, 0.9 * n_f + 1.5)
    fig_w = 8 if dataset == "oulad" else 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sources = ["trtr"] + list(SYNTHESIZERS)
    src_labels = {
        "trtr": "TRTR (Real)",
        "gaussian_copula": "TSTR (Gaussian Copula)",
        "ctgan": "TSTR (CTGAN)",
        "tabddpm": "TSTR (TabDDPM)",
    }
    src_colors = {
        "trtr": C_REAL,
        "gaussian_copula": C_SYNTH["gaussian_copula"],
        "ctgan": C_SYNTH["ctgan"],
        "tabddpm": C_SYNTH["tabddpm"],
    }

    n_src = len(sources)
    bh = 0.8 / n_src
    y_pos = np.arange(n_f)

    for si, src in enumerate(sources):
        imp = trtr_imp if src == "trtr" else tstr_imps[src]
        vals = [imp.get(f, 0.0) for f in sorted_feats]
        y = y_pos + (si - n_src / 2 + 0.5) * bh
        ax.barh(y, vals, bh * 0.92,
                color=src_colors[src], edgecolor="black",
                linewidth=0.3, zorder=3,
                label=src_labels[src])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if dataset == "assistments":
        ax.set_xlim(0, 0.14)
        ax.set_xticks([0, 0.04, 0.08, 0.12])

    ax.legend(loc="lower right", framealpha=0.9, fontsize=13)

    fig.tight_layout()
    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 16–17: Multi-Objective Performance Heatmap
# ═══════════════════════════════════════════════════════════════════════
def plot_performance_heatmap(data, dataset, fig_num):
    columns = ["Quality", "Realism", "Privacy", "Cls. AUC", "Regr. MAE"]
    matrix = np.zeros((3, 5))

    for i, syn in enumerate(SYNTHESIZERS):
        vals = [data[s][dataset]["synthesizers"][syn]["sdmetrics"][
            "overall_score"] for s in SEEDS]
        matrix[i, 0] = np.mean(vals) * 100

        vals = [data[s][dataset]["synthesizers"][syn]["c2st"][
            "effective_auc"] for s in SEEDS]
        matrix[i, 1] = _invert_score(np.mean(vals))

        vals = [data[s][dataset]["synthesizers"][syn]["mia"][
            "worst_case_effective_auc"] for s in SEEDS]
        matrix[i, 2] = _invert_score(np.mean(vals))

        trtr = np.mean([data[s][dataset]["synthesizers"][syn]["utility"][
            "classification"]["trtr_rf_auc"] for s in SEEDS])
        tstr = np.mean([data[s][dataset]["synthesizers"][syn]["utility"][
            "classification"]["rf_auc"] for s in SEEDS])
        matrix[i, 3] = _utility_score(tstr, trtr)

        trtr = np.mean([data[s][dataset]["synthesizers"][syn]["utility"][
            "regression"]["trtr_rf_mae"] for s in SEEDS])
        tstr = np.mean([data[s][dataset]["synthesizers"][syn]["utility"][
            "regression"]["rf_mae"] for s in SEEDS])
        matrix[i, 4] = _utility_score(tstr, trtr, lower_better=True)

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
    _savefig(fig, f"fig{fig_num}.png")


# ═══════════════════════════════════════════════════════════════════════
# BEESWARM HELPERS  (fig14–fig15)
# ═══════════════════════════════════════════════════════════════════════
def _infer_feature_spec(df):
    cat_cols = [c for c in df.columns if c in CATEGORICAL_COLS]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


def _build_preprocessor(num_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent",
                                         fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False)),
            ]), cat_cols)
        )
    return ColumnTransformer(transformers, remainder="drop")


def _get_feature_names(preprocessor):
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            if hasattr(ohe, "get_feature_names_out"):
                names.extend(ohe.get_feature_names_out(cols))
            else:
                names.extend(ohe.get_feature_names(cols))
    return names


def _aggregate_onehot_shap(feature_names, shap_values, cat_cols):
    """Aggregate one-hot SHAP values back to parent categorical features."""
    importance_dict = {}
    for i, fname in enumerate(feature_names):
        parent = None
        for cat in cat_cols:
            if fname.startswith(f"{cat}_"):
                parent = cat
                break
        key = parent if parent else fname
        importance_dict.setdefault(key, []).append(shap_values[:, i])

    aggregated = {}
    for key, vals in importance_dict.items():
        aggregated[key] = np.sum(vals, axis=0) if len(vals) > 1 else vals[0]
    return aggregated


def _compute_shap_for_seed(seed, data_source):
    """Train RF + compute SHAP for one seed and data source."""
    import shap as shap_lib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    seed_dir = BASE_DIR / f"seed_{seed}" / "oulad"
    data_path = seed_dir / "data.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    df_all = pd.read_parquet(data_path)
    df_real_train = df_all[df_all["split"] == "real_train"].drop(
        columns=["split", "synthesizer"]).reset_index(drop=True)
    df_real_test = df_all[df_all["split"] == "real_test"].drop(
        columns=["split", "synthesizer"]).reset_index(drop=True)

    if data_source == "trtr":
        df_train = df_real_train
    else:
        df_train = df_all[
            (df_all["split"] == "synthetic_train") &
            (df_all["synthesizer"] == data_source)
        ].drop(columns=["split", "synthesizer"]).reset_index(drop=True)

    X_train = df_train.drop(columns=ID_COLS + TARGET_COLS, errors="ignore")
    y_train = df_train[CLASS_TARGET]
    X_test = df_real_test.drop(columns=ID_COLS + TARGET_COLS, errors="ignore")

    num_cols, cat_cols = _infer_feature_spec(X_train)
    preprocessor = _build_preprocessor(num_cols, cat_cols)
    model = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=BEESWARM_N_ESTIMATORS, random_state=seed, n_jobs=1))
    ])
    model.fit(X_train, y_train)

    X_test_tf = preprocessor.transform(X_test)
    if hasattr(X_test_tf, "toarray"):
        X_test_tf = X_test_tf.toarray()

    feature_names = _get_feature_names(preprocessor)

    rng = np.random.default_rng(seed)
    n_samples = min(BEESWARM_N_SHAP_SAMPLES, X_test_tf.shape[0])
    idx = rng.choice(X_test_tf.shape[0], size=n_samples, replace=False)
    X_shap = X_test_tf[idx]

    explainer = shap_lib.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_shap, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    aggregated = _aggregate_onehot_shap(feature_names, shap_values, cat_cols)
    return aggregated, X_shap, list(aggregated.keys())


def _load_or_compute_seed(seed, model_type):
    """Return cached SHAP data for one seed, computing and saving if absent."""
    cache_file = CACHE_DIR / f"seed_{seed}_{model_type}.pkl"
    if cache_file.exists():
        print(f"Loading cached {model_type}…", end=" ", flush=True)
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    print(f"Computing {model_type}…", end=" ", flush=True)
    result = _compute_shap_for_seed(seed, model_type)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    return result


# ═══════════════════════════════════════════════════════════════════════
# FIG 14–15: SHAP Beeswarm Plots
# ═══════════════════════════════════════════════════════════════════════
def _plot_beeswarm(exp, title, xlabel, save_path):
    """Render a single beeswarm with publication-quality fonts."""
    import shap as shap_lib

    plt.rcParams.update({
        "font.size": 23,
        "axes.labelsize": 23,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
    })
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.sca(ax)
    shap_lib.plots.beeswarm(exp, max_display=BEESWARM_MAX_DISPLAY,
                             show=False, plot_size=None)
    ax.set_title(title, fontsize=23, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=23)
    ax.tick_params(axis="both", labelsize=21)
    for label in ax.get_yticklabels():
        label.set_fontsize(24)
    for label in ax.get_xticklabels():
        label.set_fontsize(21)
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_fontsize() < 21:
            item.set_fontsize(21)
    plt.tight_layout()
    fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {Path(save_path).name}")


def generate_beeswarm_figures():
    """Aggregate SHAP values across seeds and save fig14 + fig15."""
    import shap as shap_lib  # noqa: F401 (confirm import succeeds)

    print(f"\n  Seeds: {SEEDS}")
    print(f"  SHAP samples/seed: {BEESWARM_N_SHAP_SAMPLES}  |  "
          f"RF trees: {BEESWARM_N_ESTIMATORS}  |  "
          f"Cache: {CACHE_DIR}")

    all_shap_real: dict[str, list] = {}
    all_shap_synth: dict[str, list] = {}
    feature_names = None

    for seed in SEEDS:
        print(f"  Seed {seed}: ", end="", flush=True)
        shap_real, _, feat_real = _load_or_compute_seed(seed, "trtr")
        print("✓", end=" | ", flush=True)
        shap_synth, _, _ = _load_or_compute_seed(seed, "tabddpm")
        print("✓")

        if feature_names is None:
            feature_names = feat_real

        for feat in feature_names:
            all_shap_real.setdefault(feat, []).append(shap_real[feat])
            all_shap_synth.setdefault(feat, []).append(shap_synth[feat])

        del shap_real, shap_synth
        gc.collect()

    # Concatenate across seeds
    for feat in feature_names:
        all_shap_real[feat] = np.concatenate(all_shap_real[feat])
        all_shap_synth[feat] = np.concatenate(all_shap_synth[feat])

    n_total = len(all_shap_real[feature_names[0]])
    print(f"\n  Total SHAP samples: {n_total}  |  Features: {len(feature_names)}")

    # Sort by mean absolute SHAP (TRTR) and pick top N
    mean_abs = {k: np.mean(np.abs(v)) for k, v in all_shap_real.items()}
    top_features = sorted(mean_abs, key=mean_abs.__getitem__,
                          reverse=True)[:BEESWARM_MAX_DISPLAY]

    shap_real_arr = np.column_stack([all_shap_real[f] for f in top_features])
    shap_synth_arr = np.column_stack([all_shap_synth[f] for f in top_features])

    import shap as shap_lib
    exp_real = shap_lib.Explanation(values=shap_real_arr,
                                     feature_names=top_features)
    exp_synth = shap_lib.Explanation(values=shap_synth_arr,
                                      feature_names=top_features)

    _plot_beeswarm(
        exp_real,
        title="OULAD — Classification (Dropout): TRTR",
        xlabel="SHAP value (impact on model output)",
        save_path=OUT_DIR / "fig14.png",
    )
    _plot_beeswarm(
        exp_synth,
        title="OULAD — Classification (Dropout): TSTR (TabDDPM)",
        xlabel="SHAP value (impact on model output)",
        save_path=OUT_DIR / "fig15.png",
    )


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate all Synthla-Edu V2 publication figures (fig2–fig17).")
    parser.add_argument(
        "--skip-beeswarm", action="store_true",
        help="Skip fig14/fig15 SHAP beeswarm (saves ~1 min if cache is cold).")
    args = parser.parse_args()

    print("=" * 65)
    print("  Synthla-Edu V2 — All Publication Figures (fig2–fig17)")
    print("=" * 65)

    os.makedirs(OUT_DIR, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    setup_rcparams()

    # ------------------------------------------------------------------
    # Load per-seed results (used by fig2–fig13, fig16–fig17)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading per-seed results.json …")
    data = load_all_results()

    # ------------------------------------------------------------------
    # Utility figures (fig2–fig5)
    # ------------------------------------------------------------------
    print("\n[2/4] Utility figures (fig2–fig5) …")
    plot_cls_utility(data, "oulad", 2)
    plot_cls_utility(data, "assistments", 3)
    plot_reg_utility(data, "oulad", 4)
    plot_reg_utility(data, "assistments", 5)

    # ------------------------------------------------------------------
    # Fidelity & privacy figures (fig6–fig9)
    # ------------------------------------------------------------------
    print("\n[3/4] Fidelity & privacy figures (fig6–fig9) …")
    plot_sdmetrics(data, 6)
    plot_mia_overall(data, 7)
    plot_mia_per_attacker(data, "oulad", 8)
    plot_mia_per_attacker(data, "assistments", 9)

    # ------------------------------------------------------------------
    # SHAP bar charts + heatmaps (fig10–fig13, fig16–fig17)
    # ------------------------------------------------------------------
    print("\n[4/4] SHAP bar charts & heatmaps (fig10–fig13, fig16–fig17) …")
    plot_shap_bars(data, "oulad", "classification", 10)
    plot_shap_bars(data, "oulad", "regression", 11)
    plot_shap_bars(data, "assistments", "classification", 12)
    plot_shap_bars(data, "assistments", "regression", 13)
    plot_performance_heatmap(data, "oulad", 16)
    plot_performance_heatmap(data, "assistments", 17)

    # ------------------------------------------------------------------
    # SHAP beeswarm (fig14–fig15)
    # ------------------------------------------------------------------
    if args.skip_beeswarm:
        print("\n[skipped] Beeswarm figures (--skip-beeswarm flag set).")
    else:
        print("\n[+] SHAP beeswarm figures (fig14–fig15) …")
        generate_beeswarm_figures()

    print(f"\n{'=' * 65}")
    total = 14 if not args.skip_beeswarm else 12
    print(f"  Done — {total} figures saved to {OUT_DIR}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    # Check if we have enough class diversity for AUC
    if metric == "auc" and len(np.unique(y_true)) < 2:
        return {
            "metric": metric,
            "n_boot": int(n_boot),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "samples": [],
            "reason": "Only one class in y_true; AUC is not defined",
        }
    
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if metric == "auc":
            # Skip bootstrap samples that don't have both classes
            if len(np.unique(yt)) < 2:
                continue
            val = roc_auc_score(yt, yp)
        elif metric == "mae":
            val = mean_absolute_error(yt, yp)
        else:
            raise ValueError(f"Unknown metric '{metric}'")
        stats.append(float(val))

    # Return NaN CI if no valid samples
    if len(stats) == 0:
        return {
            "metric": metric,
            "n_boot": int(n_boot),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "samples": [],
            "reason": "No valid bootstrap samples",
        }

    lo = np.quantile(stats, alpha / 2.0)
    hi = np.quantile(stats, 1.0 - alpha / 2.0)
    return {
        "metric": metric,
        "n_boot": int(n_boot),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "samples": stats,
    }


def paired_permutation_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    metric: str,
    n_perm: int = 5000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired permutation test between two predictors.

    For AUC: swaps predicted probabilities per-sample between A and B.
    For MAE: swaps per-sample absolute errors.
    """
    rng = np.random.default_rng(seed)

    if metric == "auc":
        obs = roc_auc_score(y_true, pred_a) - roc_auc_score(y_true, pred_b)
        diffs = []
        for _ in range(n_perm):
            swap = rng.random(len(y_true)) < 0.5
            a = pred_a.copy()
            b = pred_b.copy()
            a[swap], b[swap] = b[swap], a[swap]
            diffs.append(roc_auc_score(y_true, a) - roc_auc_score(y_true, b))
    elif metric == "mae":
        err_a = np.abs(y_true - pred_a)
        err_b = np.abs(y_true - pred_b)
        obs = err_a.mean() - err_b.mean()
        diffs = []
        for _ in range(n_perm):
            swap = rng.random(len(y_true)) < 0.5
            ea = err_a.copy()
            eb = err_b.copy()
            ea[swap], eb[swap] = eb[swap], ea[swap]
            diffs.append(ea.mean() - eb.mean())
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    diffs = np.asarray(diffs)
    p = float((np.abs(diffs) >= abs(obs)).mean())
    return {
        "metric": metric,
        "n_perm": int(n_perm),
        "observed_delta": float(obs),
        "p_value_two_sided": p,
    }

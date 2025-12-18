from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from .models import make_model
from .preprocess import infer_feature_spec, make_preprocess_pipeline

logger = logging.getLogger(__name__)


def _effective_auc(auc: float) -> float:
    # Ensure AUC is in [0.5, 1.0] by symmetry.
    return float(max(auc, 1.0 - auc))


def knn_distance_features(
    candidates: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    exclude_cols: Optional[List[str]] = None,
    k: int = 5,
) -> pd.DataFrame:
    """Compute KNN distance-to-synthetic features for membership inference.

    Features:
      - d_min: distance to nearest synthetic neighbor
      - d_mean_k: mean distance to k nearest synthetic neighbors
    """
    exclude = exclude_cols or []
    X_syn = synthetic.drop(columns=[c for c in exclude if c in synthetic.columns])
    X_cand = candidates.drop(columns=[c for c in exclude if c in candidates.columns])

    spec = infer_feature_spec(X_syn)
    pre = make_preprocess_pipeline(spec)

    Z_syn = pre.fit_transform(X_syn)
    Z_cand = pre.transform(X_cand)

    nn = NearestNeighbors(n_neighbors=min(k, Z_syn.shape[0]), metric="euclidean", n_jobs=-1)
    nn.fit(Z_syn)
    dists, _ = nn.kneighbors(Z_cand, return_distance=True)

    out = pd.DataFrame(
        {
            "d_min": dists[:, 0],
            "d_mean_k": dists.mean(axis=1),
        }
    )
    return out


def run_mia_worst_case_auc(
    *,
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    exclude_cols: List[str],
    attacker_models: List[str],
    test_size: float = 0.3,
    random_state: int = 0,
    k: int = 5,
) -> Dict[str, Any]:
    """Membership inference via distance-to-synthetic features and multiple attackers.

    - Members: records used to train the synthesizer (real_train)
    - Non-members: holdout real records (real_holdout)

    Attacker input features: KNN distance features computed against the released synthetic dataset.
    Output: worst-case (max) Effective AUC across the provided attacker models.
    """
    # Balance members/non-members to avoid class imbalance inflating AUC
    n = min(len(real_train), len(real_holdout))
    members = real_train.sample(n=n, random_state=random_state).reset_index(drop=True)
    nonmembers = real_holdout.sample(n=n, random_state=random_state).reset_index(drop=True)

    X_mem = knn_distance_features(members, synthetic, exclude_cols=exclude_cols, k=k)
    X_non = knn_distance_features(nonmembers, synthetic, exclude_cols=exclude_cols, k=k)

    X = pd.concat([X_mem, X_non], ignore_index=True)
    y = np.concatenate([np.ones(len(X_mem)), np.zeros(len(X_non))])

    # Try stratified split; fall back to unstratified if classes too small
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        logger.warning(f"MIA stratified split failed (reason: {e}). Falling back to unstratified split.")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

    # Check if training set has both classes; skip if not (data too small/unbalanced)
    if len(np.unique(y_tr)) < 2:
        logger.warning(f"MIA: training set y_tr has only 1 class {np.unique(y_tr)}; skipping attacker models due to insufficient class diversity")
        return {
            "attacker_auc": {},
            "attacker_effective_auc": {},
            "worst_model": None,
            "worst_model_effective_auc": np.nan,
        }

    aucs: Dict[str, float] = {}
    eff_aucs: Dict[str, float] = {}
    for m in attacker_models:
        # attackers are classification models
        est = make_model(m, task="classification", random_state=random_state)
        pipe = Pipeline([("model", est)])
        pipe.fit(X_tr, y_tr)

        # probability of class 1 (member)
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_te)[:, 1]
        else:
            # fallback: decision_function -> sigmoid-ish
            scores = pipe.decision_function(X_te)  # type: ignore
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        auc = roc_auc_score(y_te, y_prob)
        aucs[m] = float(auc)
        eff_aucs[m] = _effective_auc(float(auc))

    worst_model = max(eff_aucs, key=lambda k: eff_aucs[k])
    return {
        "attacker_auc": aucs,
        "attacker_effective_auc": eff_aucs,
        "worst_case_effective_auc": float(eff_aucs[worst_model]),
        "worst_model": worst_model,
        "n_members": int(n),
        "k": int(k),
    }

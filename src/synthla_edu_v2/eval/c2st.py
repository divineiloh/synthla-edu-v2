from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import infer_feature_spec, make_preprocess_pipeline


def _make_c2st_classifier(name: str, random_state: int):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    if name == "xgb":
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise ImportError("xgboost is required for c2st classifier 'xgb'. Install with: pip install xgboost") from e
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown C2ST classifier '{name}'")


def c2st_effective_auc(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    *,
    classifier: str = "rf",
    sample_size: Optional[int] = None,
    test_size: float = 0.3,
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> Dict[str, Any]:
    """Classifier two-sample test (C2ST) with Effective AUC.

    Effective AUC = max(AUC, 1-AUC) so that 0.5 is ideal (indistinguishable) and 1.0 is fully distinguishable.
    """
    n = min(len(real_data), len(synthetic_data))
    if sample_size is not None:
        n = min(n, int(sample_size))

    real = real_data.sample(n=n, random_state=seeds[0]).reset_index(drop=True)
    syn = synthetic_data.sample(n=n, random_state=seeds[0]).reset_index(drop=True)

    X = pd.concat([real, syn], ignore_index=True)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(syn))])

    # Preprocessing inferred once (stable across repeats)
    spec = infer_feature_spec(X)
    pre = make_preprocess_pipeline(spec)

    aucs = []
    eff_aucs = []
    for s in seeds:
        # Try stratified split but fall back to unstratified on errors (small samples)
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=s, stratify=y)
        except ValueError as e:
            import logging

            logging.getLogger("synthla_edu_v2").warning(
                "C2ST stratified split failed (reason: %s). Falling back to unstratified split.",
                str(e),
            )
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=s)

        clf = _make_c2st_classifier(classifier, random_state=s)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        # If training split ends up with a single class after fallback, predict_proba will have single column.
        # In that case, skip this seed with a warning.
        clf_step = pipe.named_steps["clf"]
        classes = getattr(clf_step, "classes_", None)
        if classes is None or len(classes) < 2:
            import logging

            logging.getLogger("synthla_edu_v2").warning(
                "C2ST: classifier trained on single class %s for seed %s; skipping this fold",
                classes.tolist() if classes is not None else classes,
                s,
            )
            aucs.append(float("nan"))
            eff_aucs.append(float("nan"))
            continue

        proba = pipe.predict_proba(X_te)
        # Handle edge cases where predict_proba returns a single column
        if proba.shape[1] == 1:
            # If only one column is present, try to map it to class '1' if possible.
            if 1 in classes:
                idx1 = list(classes).index(1)
                y_prob = proba[:, idx1]
            else:
                # fallback: treat column as prob of class 0; probability of class 1 is (1 - prob_col)
                y_prob = 1.0 - proba[:, 0]
        else:
            # Normal case: find index of class 1
            idx1 = list(classes).index(1) if 1 in classes else 1
            y_prob = proba[:, idx1]

        auc = roc_auc_score(y_te, y_prob)
        eff = max(auc, 1.0 - auc)
        aucs.append(float(auc))
        eff_aucs.append(float(eff))

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "effective_auc_mean": float(np.mean(eff_aucs)),
        "effective_auc_std": float(np.std(eff_aucs, ddof=1)) if len(eff_aucs) > 1 else 0.0,
        "seeds": seeds,
        "classifier": classifier,
        "n_per_class": int(n),
    }

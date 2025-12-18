from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

from .models import make_model
from .preprocess import infer_feature_spec, make_preprocess_pipeline


@dataclass
class UtilityResult:
    metrics: Dict[str, float]
    predictions: Dict[str, np.ndarray]  # model_name -> predicted prob (cls) or y_pred (reg)
    y_true: np.ndarray


import logging


def run_utility(
    *,
    task: str,
    target_col: str,
    id_cols: List[str],
    feature_drop_cols: List[str],
    model_names: List[str],
    syn_train: pd.DataFrame,
    real_test: pd.DataFrame,
    random_state: int,
) -> UtilityResult:
    if task not in ["classification", "regression"]:
        raise ValueError(f"task must be classification or regression, got {task}")

    logger = logging.getLogger("synthla_edu_v2")

    drop_cols = set(id_cols + [target_col] + feature_drop_cols)

    X_train = syn_train.drop(columns=[c for c in drop_cols if c in syn_train.columns])
    y_train = syn_train[target_col].values

    X_test = real_test.drop(columns=[c for c in drop_cols if c in real_test.columns])
    y_test = real_test[target_col].values

    # If classification, ensure train and test contain at least two classes
    if task == "classification":
        classes_train = np.unique(y_train)
        classes_test = np.unique(y_test)
        if len(classes_train) < 2:
            logger.warning(
                "Skipping classification task '%s' because training data contains a single class: %s",
                target_col,
                classes_train.tolist(),
            )
            return UtilityResult(metrics={}, predictions={}, y_true=y_test)
        if len(classes_test) < 2:
            logger.warning(
                "Skipping classification task '%s' because test data contains a single class: %s",
                target_col,
                classes_test.tolist(),
            )
            return UtilityResult(metrics={}, predictions={}, y_true=y_test)

    # Preprocessing inferred from train columns only (stable column ordering)
    spec = infer_feature_spec(X_train)
    pre = make_preprocess_pipeline(spec)

    metrics: Dict[str, float] = {}
    preds: Dict[str, np.ndarray] = {}

    for mname in model_names:
        est = make_model(mname, task=task, random_state=random_state)
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)

        if task == "classification":
            # We use class-1 probability for AUC
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            metrics[f"{mname}_auc"] = float(auc)
            preds[mname] = y_prob
        else:
            y_pred = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            metrics[f"{mname}_mae"] = float(mae)
            preds[mname] = y_pred

    # Mean metric across the specified models
    if task == "classification":
        mean_auc = np.mean([metrics[f"{m}_auc"] for m in model_names if f"{m}_auc" in metrics])
        metrics["mean_auc"] = float(mean_auc) if len(metrics) > 0 else float("nan")
    else:
        mean_mae = np.mean([metrics[f"{m}_mae"] for m in model_names if f"{m}_mae" in metrics])
        metrics["mean_mae"] = float(mean_mae) if len(metrics) > 0 else float("nan")

    return UtilityResult(metrics=metrics, predictions=preds, y_true=y_test)

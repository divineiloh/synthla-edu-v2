from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def make_model(model_name: str, *, task: str, random_state: int) -> Any:
    """Factory for downstream utility and attacker models."""
    if task == "classification":
        if model_name == "logreg":
            return LogisticRegression(max_iter=2000, n_jobs=-1)
        if model_name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )
        if model_name == "xgb":
            try:
                import xgboost as xgb  # type: ignore
            except Exception as e:
                raise ImportError("xgboost is required for model 'xgb'. Install with: pip install xgboost") from e
            return xgb.XGBClassifier(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown classification model '{model_name}'")
    if task == "regression":
        if model_name == "ridge":
            return Ridge(alpha=1.0, random_state=random_state)
        if model_name == "rf_reg":
            return RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown regression model '{model_name}'")
    raise ValueError(f"Unknown task '{task}'")

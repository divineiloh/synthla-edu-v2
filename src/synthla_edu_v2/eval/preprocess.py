from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]


def infer_feature_spec(df: pd.DataFrame, *, exclude_cols: Optional[List[str]] = None) -> FeatureSpec:
    exclude = set(exclude_cols or [])
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if str(df[c].dtype) in ["category", "object", "string"]:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return FeatureSpec(numeric_cols=num_cols, categorical_cols=cat_cols)


def make_preprocess_pipeline(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric_cols),
            ("cat", categorical_pipe, spec.categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

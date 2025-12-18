from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame


def simple_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    stratify_col: Optional[str] = None,
) -> SplitResult:
    strat = df[stratify_col] if stratify_col else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return SplitResult(train=train_df.reset_index(drop=True), test=test_df.reset_index(drop=True))


def group_train_test_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    test_size: float,
    random_state: int,
) -> SplitResult:
    """Split rows by grouping variable to avoid entity leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col].values
    idx = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(idx, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return SplitResult(train=train_df, test=test_df)

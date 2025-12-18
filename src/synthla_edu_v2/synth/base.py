from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class FitMeta:
    train_rows: int
    train_cols: int
    extra: Dict[str, Any]


class BaseSynthesizer(ABC):
    name: str

    def __init__(self, *, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.fit_meta: Optional[FitMeta] = None

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> "BaseSynthesizer":
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

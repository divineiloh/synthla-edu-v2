from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseSynthesizer, FitMeta


def _detect_metadata(df: pd.DataFrame):
    try:
        from sdv.metadata import SingleTableMetadata  # type: ignore
    except Exception as e:
        raise ImportError(
            "SDV is required for Gaussian Copula and CTGAN. Install with: pip install sdv"
        ) from e

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata


class GaussianCopulaSynth(BaseSynthesizer):
    name = "gaussian_copula"

    def __init__(self, *, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params=params)
        self._model = None

    def fit(self, df_train: pd.DataFrame) -> "GaussianCopulaSynth":
        metadata = _detect_metadata(df_train)
        try:
            from sdv.single_table import GaussianCopulaSynthesizer  # type: ignore
        except Exception:
            # legacy SDV versions
            from sdv.tabular import GaussianCopula  # type: ignore

            self._model = GaussianCopula()
            self._model.fit(df_train)
            self.fit_meta = FitMeta(train_rows=len(df_train), train_cols=df_train.shape[1], extra={"backend": "sdv.tabular"})
            return self

        self._model = GaussianCopulaSynthesizer(metadata, **self.params)
        self._model.fit(df_train)
        self.fit_meta = FitMeta(train_rows=len(df_train), train_cols=df_train.shape[1], extra={"backend": "sdv.single_table"})
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("GaussianCopulaSynth must be fit() before sample().")
        return self._model.sample(n)


class CTGANSynth(BaseSynthesizer):
    name = "ctgan"

    def __init__(self, *, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params=params)
        self._model = None

    def fit(self, df_train: pd.DataFrame) -> "CTGANSynth":
        metadata = _detect_metadata(df_train)
        try:
            from sdv.single_table import CTGANSynthesizer  # type: ignore
        except Exception:
            # legacy SDV versions
            from sdv.tabular import CTGAN  # type: ignore

            self._model = CTGAN(**self.params)
            self._model.fit(df_train)
            self.fit_meta = FitMeta(train_rows=len(df_train), train_cols=df_train.shape[1], extra={"backend": "sdv.tabular"})
            return self

        self._model = CTGANSynthesizer(metadata, **self.params)
        self._model.fit(df_train)
        self.fit_meta = FitMeta(train_rows=len(df_train), train_cols=df_train.shape[1], extra={"backend": "sdv.single_table"})
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("CTGANSynth must be fit() before sample().")
        return self._model.sample(n)

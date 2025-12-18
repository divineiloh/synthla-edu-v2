from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseSynthesizer, FitMeta


class TabDDPMSynth(BaseSynthesizer):
    """Diffusion-based tabular synthesizer via Synthcity's 'ddpm' plugin.

    Notes:
      * This wrapper expects the input dataframe to include all features and targets.
      * The plugin generates a full synthetic dataframe with the same columns.
    """

    name = "tabddpm"

    def __init__(self, *, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params=params)
        self._plugin = None

    def fit(self, df_train: pd.DataFrame) -> "TabDDPMSynth":
        try:
            from synthcity.plugins import Plugins  # type: ignore
        except Exception as e:
            raise ImportError(
                "synthcity is required for TabDDPM. Install with: pip install synthcity"
            ) from e

        # Synthcity plugins can fit directly on a DataFrame
        self._plugin = Plugins().get("ddpm", **self.params)
        self._plugin.fit(df_train)

        self.fit_meta = FitMeta(train_rows=len(df_train), train_cols=df_train.shape[1], extra={"backend": "synthcity.ddpm"})
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._plugin is None:
            raise RuntimeError("TabDDPMSynth must be fit() before sample().")
        # generate() returns a DataLoader; convert to dataframe()
        syn_loader = self._plugin.generate(count=n)
        return syn_loader.dataframe()

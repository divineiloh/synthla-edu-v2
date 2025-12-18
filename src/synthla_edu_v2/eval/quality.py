from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def compute_sdmetrics_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    *,
    metadata_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute SDMetrics single-table quality score.

    Returns a dict containing:
      - overall_score
      - properties (per-property scores, if available)
    """
    try:
        from sdmetrics.reports.single_table import QualityReport  # type: ignore
    except Exception as e:
        raise ImportError(
            "sdmetrics is required for quality evaluation. Install with: pip install sdmetrics"
        ) from e

    report = QualityReport()
    if metadata_dict is None:
        # allow sdmetrics to infer metadata, but explicit metadata is preferred
        report.generate(real_data=real_data, synthetic_data=synthetic_data)
    else:
        report.generate(real_data=real_data, synthetic_data=synthetic_data, metadata=metadata_dict)

    out: Dict[str, Any] = {"overall_score": float(report.get_score())}
    try:
        details = report.get_details()
        out["details"] = details.to_dict() if hasattr(details, "to_dict") else details
    except Exception:
        pass
    return out

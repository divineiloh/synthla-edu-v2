from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..utils import ensure_dir, save_json


def write_table_csv(out_dir: str | Path, name: str, rows: List[Dict[str, Any]]) -> Path:
    out_dir = ensure_dir(out_dir)
    df = pd.DataFrame(rows)
    path = Path(out_dir) / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def write_json(out_dir: str | Path, name: str, obj: Any) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / f"{name}.json"
    save_json(path, obj)
    return path

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """Set python / numpy seeds (and torch, if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch is optional
        pass


def now_ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def df_signature(df: pd.DataFrame) -> Dict[str, Any]:
    """Lightweight dataframe signature for provenance tracking."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "null_frac": {c: float(df[c].isna().mean()) for c in df.columns},
    }


import logging
from logging.handlers import RotatingFileHandler


def timer() -> Tuple[float, Any]:
    start = time.time()

    def _stop() -> float:
        return time.time() - start

    return start, _stop


def configure_logging(out_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure a logger that writes to both stdout and a run.log file in out_dir.

    Returns the configured logger.
    """
    p = ensure_dir(out_dir)
    logger = logging.getLogger("synthla_edu_v2")
    logger.setLevel(level)

    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(p / "run.log", maxBytes=10_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def asdict_safe(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts where possible."""
    try:
        return asdict(obj)  # dataclass
    except Exception:
        return obj

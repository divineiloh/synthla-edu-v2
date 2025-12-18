from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # "oulad" | "assistments"
    raw_path: str  # directory containing raw files
    processed_path: str  # directory for processed outputs
    # dataset-specific parameters
    params: Dict[str, Any]


@dataclass(frozen=True)
class SynthSpec:
    name: str  # "gaussian_copula" | "ctgan" | "tabddpm"
    params: Dict[str, Any]


@dataclass(frozen=True)
class SplitSpec:
    test_size: float
    random_state: int
    stratify_col: Optional[str]  # only for classification target where appropriate


@dataclass(frozen=True)
class UtilityTaskSpec:
    name: str  # "classification" | "regression"
    target_col: str
    id_cols: List[str]
    feature_drop_cols: List[str]
    models: List[str]  # e.g. ["logreg", "rf"] or ["ridge", "rf_reg"]
    # optional per-task overrides
    params: Dict[str, Any]


@dataclass(frozen=True)
class EvalSpec:
    sdmetrics: Dict[str, Any]
    c2st: Dict[str, Any]
    mia: Dict[str, Any]
    bootstrap: Dict[str, Any]
    permutation_test: Dict[str, Any]


@dataclass(frozen=True)
class RunSpec:
    out_dir: str
    seed: int
    datasets: List[DatasetSpec]
    synthesizers: List[SynthSpec]
    split: SplitSpec
    utility_tasks: List[UtilityTaskSpec]
    evaluation: EvalSpec


def load_config(path: str | Path) -> RunSpec:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())

    datasets = [DatasetSpec(**d) for d in cfg["datasets"]]
    synthesizers = [SynthSpec(**s) for s in cfg["synthesizers"]]
    split = SplitSpec(**cfg["split"])
    utility_tasks = [UtilityTaskSpec(**t) for t in cfg["utility_tasks"]]
    evaluation = EvalSpec(**cfg["evaluation"])

    return RunSpec(
        out_dir=cfg["out_dir"],
        seed=int(cfg["seed"]),
        datasets=datasets,
        synthesizers=synthesizers,
        split=split,
        utility_tasks=utility_tasks,
        evaluation=evaluation,
    )

# -*- coding: utf-8 -*-
from __future__ import annotations

# Patch torch.amp for fastai compatibility with PyTorch 2.2.2
# This must happen BEFORE any synthcity imports
import sys
import torch
import torch.cuda.amp as _cuda_amp
import torch.cuda.amp.grad_scaler as _grad_scaler_module
torch.amp.GradScaler = _cuda_amp.GradScaler
torch.amp.grad_scaler = _grad_scaler_module
sys.modules['torch.amp.grad_scaler'] = _grad_scaler_module

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# Suppress expected warnings that don't indicate errors
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
warnings.filterwarnings('ignore', category=UserWarning, message='pkg_resources is deprecated')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*IDGenerator.*renamed.*IndexGenerator.*')

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
import sys
import platform
import time
import random
import os
import torch

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))


def remove_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def remove_glob(dir_path: Path, pattern: str) -> None:
    try:
        for p in dir_path.glob(pattern):
            if p.is_file():
                remove_if_exists(p)
    except Exception:
        pass


# -------------------------------
# Preprocessing helpers
# -------------------------------
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


# -------------------------------
# Dataset builders
# -------------------------------
DEFAULT_ASSIST_KEEP_COLS = [
    "user_id",
    "problem_id",
    "skill_id",
    "original",
    "attempt_count",
    "ms_first_response",
    "tutor_mode",
    "answer_type",
    "type",
    "hint_count",
    "hint_total",
    "overlap_time",
    "template_id",
    "first_action",
    "bottom_hint",
    "opportunity",
    "opportunity_original",
    "position",
    "correct",
]


def _find_assistments_csv(raw_dir: Path) -> Path:
    candidates = list(raw_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def build_assistments_table(raw_dir: str | Path, *, encoding: str = "ISO-8859-15") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw_dir = Path(raw_dir)
    path = _find_assistments_csv(raw_dir)
    df = pd.read_csv(path, low_memory=False, encoding=encoding)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Minimal required columns for KISS runner
    required = ["user_id", "correct"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"ASSISTments missing required columns: {missing_required}")

    # Keep only available expected columns
    keep_cols = [c for c in DEFAULT_ASSIST_KEEP_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # Ensure binary interaction target
    df["correct"] = df["correct"].astype(int)

    # Student-level aggregation (default for V2): one row per student
    grp = df.groupby("user_id")
    out = pd.DataFrame({
        "user_id": grp.size().index.astype(int),
        "n_interactions": grp.size().values.astype(float),
        "student_pct_correct": grp["correct"].mean().values.astype(float),
        "unique_skills": (grp["skill_id"].nunique().values.astype(float) if "skill_id" in df.columns else grp.size().values.astype(float)),
        "hint_rate": (grp["hint_count"].mean().values.astype(float) if "hint_count" in df.columns else grp.size().values.astype(float)),
        "avg_attempts": (grp["attempt_count"].mean().values.astype(float) if "attempt_count" in df.columns else grp.size().values.astype(float)),
        "avg_response_time": (grp["ms_first_response"].mean().values.astype(float) if "ms_first_response" in df.columns else grp.size().values.astype(float)),
    })

    # Binary classification target: high_engagement (independent from student_pct_correct)
    # Uses median split on n_interactions to avoid deterministic relationship with regression target
    engagement_threshold = out["n_interactions"].median()
    out["high_engagement"] = (out["n_interactions"] >= engagement_threshold).astype(int)
    
    # Verify independence: check correlation between features and dropped n_interactions
    # This ensures the classification task is not trivial via proxy reconstruction
    feature_cols = [c for c in out.columns if c not in ["user_id", "high_engagement", "student_pct_correct", "n_interactions"]]
    if feature_cols:
        correlations = out[feature_cols].corrwith(out["n_interactions"]).abs()
        max_corr = correlations.max()
        if max_corr > 0.8:
            print(f"WARNING: High correlation detected between n_interactions and features (max={max_corr:.3f})")
            print(f"Top correlated: {correlations.nlargest(3).to_dict()}")
            print("This may create implicit leakage in classification task.")
    
    # CRITICAL: Drop n_interactions after creating target to prevent trivial prediction
    out = out.drop(columns=["n_interactions"])
    out["user_id"] = out["user_id"].astype("int64")

    schema = {
        "id_cols": ["user_id"],
        "group_col": "user_id",
        "target_cols": ["high_engagement", "student_pct_correct"],
        "categorical_cols": [],
    }
    return out, schema


def aggregate_assistments_student_level(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("user_id")
    out = pd.DataFrame(
        {
            "user_id": grp.size().index.astype(int),
            "n_interactions": grp.size().values.astype(float),
            "n_unique_problems": grp["problem_id"].nunique().values.astype(float) if "problem_id" in df.columns else grp.size().values.astype(float),
            "n_unique_skills": grp["skill_id"].nunique().values.astype(float) if "skill_id" in df.columns else grp.size().values.astype(float),
            "mean_attempt_count": grp["attempt_count"].mean().values if "attempt_count" in df.columns else grp.size().values.astype(float),
            "mean_ms_first_response": grp["ms_first_response"].mean().values if "ms_first_response" in df.columns else grp.size().values.astype(float),
            "mean_hint_count": grp["hint_count"].mean().values if "hint_count" in df.columns else grp.size().values.astype(float),
            "mean_hint_total": grp["hint_total"].mean().values if "hint_total" in df.columns else grp.size().values.astype(float),
            "mean_overlap_time": grp["overlap_time"].mean().values if "overlap_time" in df.columns else grp.size().values.astype(float),
            "student_pct_correct": grp["correct"].mean().values.astype(float),
        }
    )
    return out


OULAD_REQUIRED_FILES = [
    "studentinfo",
    "studentregistration",
    "studentvle",
    "studentassessment",
    "assessments",
    "vle",
]


def _find_csv(raw_dir: Path, stem_lower: str) -> Path:
    candidates = list(raw_dir.rglob("*.csv"))
    for p in candidates:
        name = p.stem.lower()
        if stem_lower in name:
            return p
    raise FileNotFoundError(f"Could not find CSV for '{stem_lower}' under {raw_dir}")


def load_raw_oulad(raw_dir: str | Path, *, skip_keys: Optional[set[str]] = None) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    dfs: Dict[str, pd.DataFrame] = {}
    skip_keys = skip_keys or set()
    for key in OULAD_REQUIRED_FILES:
        if key in skip_keys:
            continue
        path = _find_csv(raw_dir, key)
        dfs[key] = pd.read_csv(path)
    return dfs


def _aggregate_studentvle_streaming(
    studentvle_csv: Path,
    keys: List[str],
    *,
    min_vle_clicks_clip: float = 0.0,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    # Stream the giant file to avoid large contiguous allocations during pandas factorization/groupby.
    # Preserve the original feature set while staying memory-safe.
    import sys
    usecols = [c for c in (keys + ["sum_click", "id_site", "date"]) if c]
    acc_sum: Dict[tuple, float] = {}
    acc_count: Dict[tuple, int] = {}
    
    print(f"[VLE Aggregation] Processing {studentvle_csv.name} in chunks...", flush=True)
    print(f"[VLE Aggregation] Note: This may take 2-5 minutes for large datasets (10M+ rows)", flush=True)

    # Auto-detect ranges from data to avoid hardcoded caps
    site_min, site_max = None, None
    date_min, date_max = None, None
    site_bytes_initial = 64
    date_bytes_initial = 64
    acc_sites_bits: Dict[tuple, bytearray] = {}
    acc_days_bits: Dict[tuple, bytearray] = {}
    
    first_pass = True
    chunk_num = 0

    def _set_bits(bitarr: bytearray, values: np.ndarray, *, offset: int, cap_bits: int) -> None:
        # values may be float with NaN; coerce safely - optimized to avoid Series overhead
        if len(values) == 0:
            return
        # Filter NaN and convert directly
        vals = values[~pd.isna(values)].astype(int)
        for v in vals:
            idx = v + offset
            if idx < 0 or idx >= cap_bits:
                continue
            byte_i = idx >> 3
            bit_i = idx & 7
            need_len = byte_i + 1
            if need_len > len(bitarr):
                bitarr.extend(b"\x00" * (need_len - len(bitarr)))
            bitarr[byte_i] |= (1 << bit_i)

    def _popcount_bytes(bitarr: bytearray) -> int:
        return sum(int(b).bit_count() for b in bitarr)

    for chunk in pd.read_csv(studentvle_csv, usecols=usecols, chunksize=chunksize):
        # First pass: detect ranges
        if first_pass:
            if "id_site" in chunk.columns:
                chunk_site_vals = chunk["id_site"].dropna()
                if len(chunk_site_vals) > 0:
                    site_min = int(chunk_site_vals.min()) if site_min is None else min(site_min, int(chunk_site_vals.min()))
                    site_max = int(chunk_site_vals.max()) if site_max is None else max(site_max, int(chunk_site_vals.max()))
            if "date" in chunk.columns:
                chunk_date_vals = chunk["date"].dropna()
                if len(chunk_date_vals) > 0:
                    date_min = int(chunk_date_vals.min()) if date_min is None else min(date_min, int(chunk_date_vals.min()))
                    date_max = int(chunk_date_vals.max()) if date_max is None else max(date_max, int(chunk_date_vals.max()))
        
        if min_vle_clicks_clip > 0.0:
            chunk["sum_click"] = chunk["sum_click"].clip(lower=min_vle_clicks_clip)

        # Sum + count
        grp_clicks = chunk.groupby(keys, sort=False)["sum_click"]
        sums = grp_clicks.sum()
        counts = grp_clicks.size()
        for k, v in sums.items():
            acc_sum[k] = acc_sum.get(k, 0.0) + float(v)
        for k, v in counts.items():
            acc_count[k] = acc_count.get(k, 0) + int(v)

        # After first chunk, compute bit capacities
        if first_pass:
            first_pass = False
            # Add 10% buffer for safety
            site_offset = 0 if site_min is None else -site_min
            site_cap_bits = 1 if site_max is None else int((site_max - (site_min or 0)) * 1.1) + 100
            date_offset = 0 if date_min is None else -date_min
            date_cap_bits = 1 if date_max is None else int((date_max - (date_min or 0)) * 1.1) + 100

        # Distinct sites
        if "id_site" in chunk.columns and site_min is not None:
            sites = chunk.groupby(keys, sort=False)["id_site"].unique()
            for k, arr in sites.items():
                bitarr = acc_sites_bits.get(k)
                if bitarr is None:
                    bitarr = bytearray(site_bytes_initial)
                    acc_sites_bits[k] = bitarr
                _set_bits(bitarr, np.asarray(arr), offset=site_offset, cap_bits=site_cap_bits)

        # Distinct days
        if "date" in chunk.columns and date_min is not None:
            days = chunk.groupby(keys, sort=False)["date"].unique()
            for k, arr in days.items():
                bitarr = acc_days_bits.get(k)
                if bitarr is None:
                    bitarr = bytearray(date_bytes_initial)
                    acc_days_bits[k] = bitarr
                _set_bits(bitarr, np.asarray(arr), offset=date_offset, cap_bits=date_cap_bits)

    print(f"\n[VLE Aggregation] Finalizing aggregation for {len(acc_sum):,} student groups...", flush=True)
    rows = []
    for k, total in acc_sum.items():
        nrec = acc_count.get(k, 0)
        nsites = _popcount_bytes(acc_sites_bits.get(k, bytearray()))
        ndays = _popcount_bytes(acc_days_bits.get(k, bytearray()))
        rows.append((*k, total, nrec, nsites, ndays))

    vle_feat = pd.DataFrame(
        rows,
        columns=[*keys, "total_vle_clicks", "n_vle_records", "n_vle_sites", "n_vle_days"],
    )
    vle_feat["mean_vle_clicks"] = vle_feat["total_vle_clicks"] / vle_feat["n_vle_records"].replace(0, np.nan)
    vle_feat["clicks_per_active_day"] = vle_feat["total_vle_clicks"] / vle_feat["n_vle_days"].replace(0, np.nan)
    print(f"[VLE Aggregation] Complete! Generated {len(vle_feat):,} feature rows", flush=True)
    return vle_feat


def build_oulad_student_table(raw_dir: str | Path, *, min_vle_clicks_clip: float = 0.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Avoid loading the massive studentVle table all at once (stream it instead).
    dfs = load_raw_oulad(raw_dir, skip_keys={"studentvle"})
    info = dfs["studentinfo"].copy()
    reg = dfs["studentregistration"].copy()
    sass = dfs["studentassessment"].copy()
    ass = dfs["assessments"].copy()

    keys = ["code_module", "code_presentation", "id_student"]

    # Target: 1 = Fail/Withdrawn, 0 = Pass/Distinction
    info["dropout"] = info["final_result"].astype(str).str.lower().isin(["withdrawn", "fail"]).astype(int)

    # Reduce memory pressure for large groupbys (studentVle can be ~10M rows)
    # - sort=False prevents pandas from allocating extra arrays to reorder group labels
    # - observed=True is only used when group keys are categorical
    # NOTE: Avoid casting the huge studentVle keys to categorical on low-memory machines;
    # doing so can trigger large temporary allocations during dtype inference.
    for c in ["code_module", "code_presentation"]:
        for df_ in (reg, sass, ass):
            if c in df_.columns and df_[c].dtype == object:
                df_[c] = df_[c].astype("category")

    reg_feat = (
        reg.groupby(keys, as_index=False, sort=False, observed=True)
        .agg(
            date_registration=("date_registration", "min"),
            date_unregistration=("date_unregistration", "min"),
        )
    )
    reg_feat["is_unregistered"] = reg_feat["date_unregistration"].notna().astype(int)

    studentvle_csv = _find_csv(Path(raw_dir), "studentvle")
    vle_feat = _aggregate_studentvle_streaming(studentvle_csv, keys, min_vle_clicks_clip=min_vle_clicks_clip)

    # Fix: First attach (code_module, code_presentation, weight) from assessments to studentAssessment
    # This prevents many-to-many join that would duplicate rows
    ass_meta = ass[["id_assessment", "code_module", "code_presentation", "weight"]].copy()
    sass = sass.merge(ass_meta, on="id_assessment", how="left")
    
    # Now aggregate grades by (code_module, code_presentation, id_student)
    sass["score_x_weight"] = sass["score"] * sass["weight"]
    grade_feat = (
        sass.groupby(keys, as_index=False, sort=False, observed=True)
        .agg(
            n_assessments=("id_assessment", "nunique"),
            total_weight=("weight", "sum"),
            weighted_score_sum=("score_x_weight", "sum"),
            mean_score=("score", "mean"),
        )
    )
    grade_feat["final_grade"] = grade_feat["weighted_score_sum"] / grade_feat["total_weight"].replace(0, np.nan)

    df = info.merge(reg_feat, on=keys, how="left").merge(vle_feat, on=keys, how="left").merge(grade_feat, on=keys, how="left")

    # Drop the source of the target to prevent leakage
    if "final_result" in df.columns:
        df.drop(columns=["final_result"], inplace=True)
    
    # Drop intermediate grade calculation columns to prevent regression target leakage
    # (final_grade = weighted_score_sum / total_weight is deterministic)
    # mean_score is highly correlated with final_grade (r≈0.92) and should be excluded
    leakage_cols = ["weighted_score_sum", "total_weight", "mean_score"]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    # Verify independence: check correlation between NUMERIC features and target columns
    target_cols = ["dropout", "final_grade"]
    id_cols = keys
    # Only check numeric features to avoid type errors with categorical columns
    numeric_feature_cols = [c for c in df.columns if c not in target_cols + id_cols 
                            and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_feature_cols and "final_grade" in df.columns:
        # Check final_grade correlation (most likely to have leakage from assessment features)
        correlations = df[numeric_feature_cols].corrwith(df["final_grade"]).abs()
        max_corr = correlations.max()
        if max_corr > 0.8:
            max_feat = correlations.idxmax()
            print(f"WARNING: High correlation detected between '{max_feat}' and 'final_grade': {max_corr:.3f}")
            print(f"  This may indicate target leakage. Consider removing '{max_feat}' if deterministic.")
        # Also check dropout correlation (binary target, may have lower correlations)
        if "dropout" in df.columns:
            dropout_corr = df[numeric_feature_cols].corrwith(df["dropout"]).abs()
            max_dropout_corr = dropout_corr.max()
            if max_dropout_corr > 0.8:
                max_dropout_feat = dropout_corr.idxmax()
                print(f"WARNING: High correlation detected between '{max_dropout_feat}' and 'dropout': {max_dropout_corr:.3f}")
                print(f"  This may indicate target leakage. Consider removing '{max_dropout_feat}' if deterministic.")

    cat_cols = [
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    for c in df.columns:
        if c in keys or c in cat_cols:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(float)

    vle_cols = ["total_vle_clicks", "mean_vle_clicks", "n_vle_records", "n_vle_sites", "n_vle_days", "clicks_per_active_day"]
    for c in vle_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    assess_cols = ["n_assessments", "total_weight", "weighted_score_sum", "mean_score", "final_grade"]
    for c in assess_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    if "date_unregistration" in df.columns:
        df["date_unregistration"] = df["date_unregistration"].fillna(-999.0)

    for c in cat_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].cat.add_categories(["Unknown"]).fillna("Unknown")

    schema = {
        "id_cols": keys,
        "group_col": "id_student",  # Ensure grouped splitting to prevent student leakage
        "target_cols": ["dropout", "final_grade"],
        "categorical_cols": [c for c in cat_cols if c in df.columns],
    }
    return df, schema


# -------------------------------
# Splitting helpers
# -------------------------------
@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame


def simple_train_test_split(df: pd.DataFrame, *, test_size: float, random_state: int, stratify_col: Optional[str] = None) -> SplitResult:
    strat = df[stratify_col] if stratify_col else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return SplitResult(train=train_df.reset_index(drop=True), test=test_df.reset_index(drop=True))


def group_train_test_split(df: pd.DataFrame, *, group_col: str, test_size: float, random_state: int) -> SplitResult:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col].values
    idx = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(idx, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return SplitResult(train=train_df, test=test_df)


# -------------------------------
# Synthesizer (Gaussian Copula, CTGAN, TabDDPM)
# -------------------------------
class GaussianCopulaSynth:
    name = "gaussian_copula"

    def __init__(self, **kwargs) -> None:
        self.params = kwargs
        self._model = None
        self._categorical_categories: Dict[str, List[str]] = {}

    def fit(self, df_train: pd.DataFrame) -> "GaussianCopulaSynth":
        from sdv.metadata import SingleTableMetadata  # type: ignore
        from sdv.single_table import GaussianCopulaSynthesizer  # type: ignore

        # Remember which columns were categorical so we can restore dtype consistency
        self._categorical_categories = {}
        for col in df_train.columns:
            if isinstance(df_train[col].dtype, pd.CategoricalDtype):
                try:
                    self._categorical_categories[col] = [str(v) for v in df_train[col].cat.categories]
                except Exception:
                    pass
        
        # WORKAROUND: SDV 1.0+ categorical transformer bug with special characters
        # Issue: RDT's FrequencyEncoder.map_labels fails on category values containing '<=' or '>='
        # Error: KeyError when mapping '55<=' in age_band column
        # Solution: Sanitize all categorical columns before fit, restore after sample
        # Scope: Applied to ALL object/categorical columns for consistency
        # Tested: Correctly restores original values (e.g., '55<=', '0-35', '35-55')
        df_fixed = df_train.copy()
        for col in df_fixed.columns:
            if isinstance(df_fixed[col].dtype, pd.CategoricalDtype) or df_fixed[col].dtype == 'object':
                df_fixed[col] = df_fixed[col].astype(str).replace({'55<=': '55_plus', '0-35': '0_35', '35-55': '35_55'})
        
        md = SingleTableMetadata()
        md.detect_from_dataframe(df_fixed)
        self._model = GaussianCopulaSynthesizer(md, **self.params)
        self._model.fit(df_fixed)
        return self

    def sample(self, n: int, *, random_state: Optional[int] = None) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit() must be called before sample().")
        # SDV 1.0+ supports random_state for reproducibility
        if random_state is not None:
            set_seed(random_state)
            try:
                synthetic = self._model.sample(n, random_state=random_state)
            except TypeError:
                # Fallback for older SDV versions
                synthetic = self._model.sample(n)
        else:
            synthetic = self._model.sample(n)
        # Restore original category names
        for col in synthetic.columns:
            if synthetic[col].dtype == 'object' or isinstance(synthetic[col].dtype, pd.CategoricalDtype):
                synthetic[col] = synthetic[col].astype(str).replace({'55_plus': '55<=', '0_35': '0-35', '35_55': '35-55'})

        # Restore pandas categorical dtype where appropriate (downstream consumers may rely on it)
        for col, base_categories in self._categorical_categories.items():
            if col not in synthetic.columns:
                continue
            values = synthetic[col].astype(str)
            base_set = set(base_categories)
            extra = sorted(set(values.unique()) - base_set)
            categories = base_categories + extra
            synthetic[col] = pd.Categorical(values, categories=categories)
        return synthetic


class CTGANSynth:
    name = "ctgan"

    def __init__(self, **kwargs) -> None:
        self.params = {k: v for k, v in kwargs.items() if k in ["epochs", "batch_size", "generator_dim", "discriminator_dim"]}
        self._model = None
        self._categorical_categories: Dict[str, List[str]] = {}

    def fit(self, df_train: pd.DataFrame) -> "CTGANSynth":
        from sdv.metadata import SingleTableMetadata  # type: ignore
        from sdv.single_table import CTGANSynthesizer  # type: ignore

        # Remember which columns were categorical so we can restore dtype consistency
        self._categorical_categories = {}
        for col in df_train.columns:
            if isinstance(df_train[col].dtype, pd.CategoricalDtype):
                try:
                    self._categorical_categories[col] = [str(v) for v in df_train[col].cat.categories]
                except Exception:
                    pass
        
        # Fix problematic category values that SDV can't handle
        df_fixed = df_train.copy()
        for col in df_fixed.columns:
            if isinstance(df_fixed[col].dtype, pd.CategoricalDtype) or df_fixed[col].dtype == 'object':
                # Replace '<=' and '>=' in category names (SDV's categorical transformer has a bug with these)
                df_fixed[col] = df_fixed[col].astype(str).replace({'55<=': '55_plus', '0-35': '0_35', '35-55': '35_55'})
        
        md = SingleTableMetadata()
        md.detect_from_dataframe(df_fixed)
        self._model = CTGANSynthesizer(md, epochs=self.params.get("epochs", 300), batch_size=self.params.get("batch_size", 500))
        self._model.fit(df_fixed)
        return self

    def sample(self, n: int, *, random_state: Optional[int] = None) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit() must be called before sample().")
        # SDV 1.0+ supports random_state for reproducibility
        if random_state is not None:
            set_seed(random_state)
            try:
                synthetic = self._model.sample(n, random_state=random_state)
            except TypeError:
                # Fallback for older SDV versions
                synthetic = self._model.sample(n)
        else:
            synthetic = self._model.sample(n)
        # Restore original category names
        for col in synthetic.columns:
            if synthetic[col].dtype == 'object' or isinstance(synthetic[col].dtype, pd.CategoricalDtype):
                synthetic[col] = synthetic[col].astype(str).replace({'55_plus': '55<=', '0_35': '0-35', '35_55': '35-55'})

        # Restore pandas categorical dtype where appropriate
        for col, base_categories in self._categorical_categories.items():
            if col not in synthetic.columns:
                continue
            values = synthetic[col].astype(str)
            base_set = set(base_categories)
            extra = sorted(set(values.unique()) - base_set)
            categories = base_categories + extra
            synthetic[col] = pd.Categorical(values, categories=categories)
        return synthetic


class TabDDPMSynth:
    name = "tabddpm"

    def __init__(self, **kwargs) -> None:
        # Set stable defaults for TabDDPM to avoid NaN issues
        # NOTE: TabDDPM can be unstable on datasets with:
        #   - Extreme outliers (>99.5th percentile) → auto-clipped in fit()
        #   - High missing data rates (>10%) → auto-imputed in fit()
        #   - Small sample sizes (<500 rows) → may produce poor results
        #   - High cardinality categoricals (>100 categories) → consider encoding
        self.params = {
            "n_iter": kwargs.get("n_iter", 2000), # More iterations for convergence with lower LR
            "lr": 0.0002,  # Lower learning rate for stability (was 0.001)
            "num_timesteps": 1000,  # More timesteps for better quality (was 100)
            "batch_size": 1024,  # Larger batch size for stability
            "gaussian_loss_type": "mse",
            "scheduler": "cosine",  # Cosine scheduler is more stable
            "model_type": "mlp",
            "weight_decay": 1e-4,
        }
        # Override with user params
        self.params.update(kwargs)
        self._model = None

    def fit(self, df_train: pd.DataFrame) -> "TabDDPMSynth":
        try:
            # Import inside method to avoid circular imports/optional dependency issues
            from synthcity.plugins.core.dataloader import GenericDataLoader  # type: ignore
            from synthcity.plugins import Plugins  # type: ignore
            
            # Preprocess: ensure all numeric cols are float, no extreme values
            df_prep = df_train.copy()
            
            # Fill NaNs to prevent loss=nan
            for col in df_prep.columns:
                if df_prep[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df_prep[col]):
                        df_prep[col] = df_prep[col].fillna(df_prep[col].median())
                    else:
                        df_prep[col] = df_prep[col].fillna(df_prep[col].mode()[0])

            for col in df_prep.columns:
                if pd.api.types.is_numeric_dtype(df_prep[col]):
                    # Clip extreme values to avoid numerical instability
                    df_prep[col] = df_prep[col].astype(float)
                    # Clip to 0.5th and 99.5th percentiles to remove extreme outliers
                    q_low, q_high = df_prep[col].quantile([0.005, 0.995])
                    df_prep[col] = df_prep[col].clip(lower=q_low, upper=q_high)
            
            loader = GenericDataLoader(df_prep)
            
            self._model = Plugins().get("ddpm", **self.params)
            self._model.fit(loader)
        except Exception as e:
            raise RuntimeError(f"TabDDPM fitting failed: {e}. Ensure synthcity>=0.2.11 and compatible torch are installed.")
        return self

    def sample(self, n: int, *, random_state: Optional[int] = None) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit() must be called before sample().")
        
        # Set seed for reproducibility if provided
        if random_state is not None:
            set_seed(random_state)
        
        # TabDDPM can be unstable with sampling_patience parameter
        # Use simple generate() call with batch sampling for large n
        try:
            if n > 5000:
                # For large samples, use batching to avoid memory issues
                batch_size = 1000
                batches = []
                remaining = n
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    batch_samples = self._model.generate(count=current_batch)
                    batches.append(batch_samples.to_pandas() if hasattr(batch_samples, "to_pandas") else batch_samples.dataframe())
                    remaining -= current_batch
                samples = pd.concat(batches, ignore_index=True)
            else:
                # For smaller samples, generate directly
                samples = self._model.generate(count=n)
                samples = samples.to_pandas() if hasattr(samples, "to_pandas") else samples.dataframe()
            
            # CRITICAL: Validate TabDDPM output for publication quality
            # TabDDPM can produce NaN/inf values during training instability
            if samples.isna().sum().sum() > 0:
                nan_cols = samples.columns[samples.isna().any()].tolist()
                raise RuntimeError(f"TabDDPM produced NaN values in columns: {nan_cols}. "
                                 f"This indicates training instability. Increase n_iter (current: {self.params.get('n_iter', 'unknown')}) "
                                 f"or remove --quick flag for stable results.")
            
            if np.isinf(samples.select_dtypes(include=[np.number]).values).any():
                raise RuntimeError(f"TabDDPM produced infinite values. This indicates numerical instability. "
                                 f"Try increasing n_iter or adjusting learning rate.")
            
            return samples
        except RuntimeError:
            # Re-raise our validation errors
            raise
        except Exception as e:
            raise RuntimeError(f"TabDDPM sampling failed: {e}. This can occur with complex datasets. Try increasing n_iter or reducing num_timesteps.")

# -------------------------------
# Evaluations: SDMetrics Quality, C2ST, MIA
# -------------------------------

def sdmetrics_quality(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, Any]:
    try:
        from sdmetrics.reports.single_table import QualityReport  # type: ignore
    except Exception as e:
        raise ImportError("sdmetrics is required. Install with: pip install sdmetrics") from e

    # Detect metadata using SDV for compatibility with newer sdmetrics requiring metadata
    metadata_dict: Optional[Dict[str, Any]] = None
    try:
        from sdv.metadata import SingleTableMetadata  # type: ignore

        md = SingleTableMetadata()
        md.detect_from_dataframe(real)
        metadata_dict = md.to_dict()
    except Exception:
        metadata_dict = None

    report = QualityReport()
    if metadata_dict is not None:
        report.generate(real_data=real, synthetic_data=syn, metadata=metadata_dict)
    else:
        # Fallback for older sdmetrics that allow inferring metadata internally
        report.generate(real_data=real, synthetic_data=syn)

    out: Dict[str, Any] = {"overall_score": float(report.get_score())}
    try:
        details = report.get_details()
        out["details"] = details.to_dict() if hasattr(details, "to_dict") else details
    except Exception:
        pass
    return out


def c2st_effective_auc(real_test: pd.DataFrame, synthetic_train: pd.DataFrame, *, exclude_cols: Optional[List[str]] = None, test_size: float = 0.3, seed: int = 0) -> Dict[str, Any]:
    # Compare REAL TEST vs SYNTHETIC TRAIN only (leakage-safe)
    # Remove ID columns and any other excluded columns to prevent trivial discrimination
    exclude = exclude_cols or []
    real_test_clean = real_test.drop(columns=[c for c in exclude if c in real_test.columns], errors='ignore')
    synthetic_train_clean = synthetic_train.drop(columns=[c for c in exclude if c in synthetic_train.columns], errors='ignore')
    
    n = min(len(real_test_clean), len(synthetic_train_clean))
    real_sample = real_test_clean.sample(n=n, random_state=seed).reset_index(drop=True)
    synthetic_sample = synthetic_train_clean.sample(n=n, random_state=seed).reset_index(drop=True)

    X = pd.concat([real_sample, synthetic_sample], ignore_index=True)
    y = np.concatenate([np.ones(len(real_sample)), np.zeros(len(synthetic_sample))])

    spec = infer_feature_spec(X)
    preprocessor = make_preprocess_pipeline(spec)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Random Forest with 300 trees (sufficient for stable performance on binary classification)
    # No hyperparameter tuning: RF is robust to defaults for discrimination tasks (Hastie et al., 2009)
    classifier = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    y_prob = proba[:, 1] if proba.shape[1] > 1 else (1.0 - proba[:, 0])
    auc = roc_auc_score(y_test, y_prob)
    effective_auc = max(auc, 1.0 - auc)
    return {"effective_auc": float(effective_auc), "classifier": "random_forest", "n_per_class": int(n)}


def mia_worst_case_effective_auc(real_train: pd.DataFrame, real_holdout: pd.DataFrame, synthetic: pd.DataFrame, *, exclude_cols: Optional[List[str]] = None, test_size: float = 0.3, random_state: int = 0, k: int = 5) -> Dict[str, Any]:
    """Membership Inference Attack (MIA) for privacy evaluation.
    
    Tests if an attacker can infer whether a record was in the training set used
    to generate synthetic data. Uses distance-based features (k-NN to synthetic)
    with multiple attack models (LR, RF, optional XGBoost).
    
    EFFECTIVE AUC INTERPRETATION:
    - AUC = 0.5: Attacker cannot predict membership (IDEAL for privacy)
    - AUC = 1.0 or 0.0: Perfect membership inference (BAD for privacy)
    - Effective AUC = max(AUC, 1-AUC): Handles label-flipping ambiguity
      - Ensures metric represents worst-case attack performance
      - Always >= 0.5 (interpretable as attack success rate)
    
    WORST-CASE REPORTING:
    - Returns the maximum effective AUC across all attackers (LR, RF, XGB)
    - Conservative evaluation: assumes adversary uses best available attack
    
    For publication: Report worst-case effective AUC and interpret distance from 0.5
    as measure of privacy risk (how easily attackers can infer membership).
    """
    n = min(len(real_train), len(real_holdout))
    members = real_train.sample(n=n, random_state=random_state).reset_index(drop=True)
    nonmembers = real_holdout.sample(n=n, random_state=random_state).reset_index(drop=True)

    exclude = exclude_cols or []
    X_synthetic = synthetic.drop(columns=[c for c in exclude if c in synthetic.columns])
    X_members = members.drop(columns=[c for c in exclude if c in members.columns])
    X_nonmembers = nonmembers.drop(columns=[c for c in exclude if c in nonmembers.columns])

    spec = infer_feature_spec(X_synthetic)
    preprocessor = make_preprocess_pipeline(spec)

    Z_synthetic = preprocessor.fit_transform(X_synthetic)
    Z_members = preprocessor.transform(X_members)
    Z_nonmembers = preprocessor.transform(X_nonmembers)

    knn = NearestNeighbors(n_neighbors=min(k, Z_synthetic.shape[0]), metric="euclidean", n_jobs=-1)
    knn.fit(Z_synthetic)
    distances_members, _ = knn.kneighbors(Z_members, return_distance=True)
    distances_nonmembers, _ = knn.kneighbors(Z_nonmembers, return_distance=True)

    X = pd.DataFrame({
        "d_min": np.concatenate([distances_members[:, 0], distances_nonmembers[:, 0]]),
        "d_mean_k": np.concatenate([distances_members.mean(axis=1), distances_nonmembers.mean(axis=1)]),
    })
    y = np.concatenate([np.ones(len(Z_members)), np.zeros(len(Z_nonmembers))])

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Multi-attacker: Logistic Regression, Random Forest, optional XGBoost; worst-case effective AUC
    results: Dict[str, Any] = {"attackers": {}}

    lr = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc = float(roc_auc_score(y_test, lr_prob))
    # Effective AUC = max(AUC, 1-AUC): worst-case attack performance
    lr_eff = float(max(lr_auc, 1.0 - lr_auc))
    results["attackers"]["logistic_regression"] = {"auc": lr_auc, "effective_auc": lr_eff}

    # Random Forest with 300 trees (sufficient for convergence, diminishing returns beyond 300)
    # Standard configuration for membership inference attacks (Shokri et al., 2017)
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_prob))
    # Effective AUC = max(AUC, 1-AUC): worst-case attack performance
    rf_eff = float(max(rf_auc, 1.0 - rf_auc))
    results["attackers"]["random_forest"] = {"auc": rf_auc, "effective_auc": rf_eff}

    if _HAS_XGB:
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, n_jobs=-1, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        xgb_auc = float(roc_auc_score(y_test, xgb_prob))
        # Effective AUC = max(AUC, 1-AUC): worst-case attack performance
        xgb_eff = float(max(xgb_auc, 1.0 - xgb_auc))
        results["attackers"]["xgboost"] = {"auc": xgb_auc, "effective_auc": xgb_eff}

    # Worst-case privacy risk: maximum effective AUC across all attack models
    worst = max(v["effective_auc"] for v in results["attackers"].values())
    results["worst_case_effective_auc"] = float(worst)
    results["n_members"] = int(n)
    results["knn_neighbors"] = int(k)
    return results

def bootstrap_ci(metric_vector: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, random_state: int = 0) -> Dict[str, float]:
    """Bootstrap confidence intervals with warnings for small sample sizes.
    
    Minimum recommended sample size: 30 observations
    Small samples (n < 50) use reduced bootstrap iterations to avoid overfitting.
    """
    n = len(metric_vector)
    
    # Warn for small samples
    if n < 30:
        print(f"WARNING: Small sample size (n={n}) may produce unreliable bootstrap CIs. Recommend n >= 30.")
    
    # Reduce bootstrap iterations for very small samples
    if n < 50:
        n_boot = min(n_boot, 500)
    
    rng = np.random.default_rng(random_state)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(float(np.mean(metric_vector[idx])))
    boot = np.array(boot)
    return {
        "mean": float(np.mean(boot)), 
        "ci_low": float(np.quantile(boot, alpha/2)), 
        "ci_high": float(np.quantile(boot, 1 - alpha/2)),
        "n_samples": int(n),
        "n_bootstrap": int(n_boot)
    }

def tstr_utility(real_train: pd.DataFrame, real_test: pd.DataFrame, synthetic_train: pd.DataFrame, *, class_target: str, reg_target: str, seed: int = 0) -> Dict[str, Any]:
    """TSTR (Train-on-Synthetic-Test-on-Real) utility evaluation with TRTR ceiling.
    
    TSTR METHODOLOGY (Primary Utility Metric):
    - Train models (RF, LR for classification; RF, Ridge for regression) on SYNTHETIC data
    - Test on REAL held-out data
    - Measures: How useful is the synthetic data for downstream ML tasks?
    - Interpretation: Higher AUC/lower MAE = better synthetic data utility
    
    TRTR METHODOLOGY (Performance Ceiling):
    - Train models on REAL training data
    - Test on REAL held-out data
    - Measures: Best possible performance with real data
    - Interpretation: Upper bound for TSTR metrics (synthetic can't exceed real)
    
    UTILITY GAP ANALYSIS:
    - Compare TSTR vs TRTR to quantify utility loss from using synthetic data
    - Example: TSTR AUC=0.85, TRTR AUC=0.90 → 5.6% utility gap
    - Smaller gap = better synthetic data quality for ML applications
    
    For publication: Report both TSTR (primary) and TRTR (ceiling) with utility gap.
    """
    def split_X_y(df: pd.DataFrame, target: str, all_targets: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        # Drop ALL target columns to prevent cross-target leakage
        drop_cols = [c for c in all_targets if c in df.columns]
        X = df.drop(columns=drop_cols)
        y = df[target].values
        return X, y

    # Classification: RF + LR, train on synthetic, test on real
    all_targets = [class_target, reg_target]
    X_syn_cls, y_syn_cls = split_X_y(synthetic_train, class_target, all_targets)
    X_real_te_cls, y_real_te_cls = split_X_y(real_test, class_target, all_targets)

    spec_cls = infer_feature_spec(pd.concat([X_syn_cls, X_real_te_cls], axis=0))

    rf_cls = Pipeline([("pre", make_preprocess_pipeline(spec_cls)), ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))])
    rf_cls.fit(X_syn_cls, y_syn_cls)
    rf_prob = rf_cls.predict_proba(X_real_te_cls)[:, 1]
    rf_auc = float(roc_auc_score(y_real_te_cls, rf_prob))
    # Compute per-sample log loss for statistical tests
    rf_logloss_vec = -np.log(np.clip(rf_prob * y_real_te_cls + (1 - rf_prob) * (1 - y_real_te_cls), 1e-15, 1.0))
    # Bootstrap CI for AUC: resample test set and recompute AUC
    rng_rf = np.random.default_rng(seed)
    rf_auc_boots = []
    for _ in range(1000):
        idx = rng_rf.integers(0, len(y_real_te_cls), size=len(y_real_te_cls))
        rf_auc_boots.append(float(roc_auc_score(y_real_te_cls[idx], rf_prob[idx])))
    rf_auc_ci = {"mean": float(np.mean(rf_auc_boots)), "ci_low": float(np.quantile(rf_auc_boots, 0.025)), "ci_high": float(np.quantile(rf_auc_boots, 0.975))}

    lr_cls = Pipeline([("pre", make_preprocess_pipeline(spec_cls)), ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed))])
    lr_cls.fit(X_syn_cls, y_syn_cls)
    lr_prob = lr_cls.predict_proba(X_real_te_cls)[:, 1]
    lr_auc = float(roc_auc_score(y_real_te_cls, lr_prob))
    # Compute per-sample log loss for statistical tests
    lr_logloss_vec = -np.log(np.clip(lr_prob * y_real_te_cls + (1 - lr_prob) * (1 - y_real_te_cls), 1e-15, 1.0))
    # Bootstrap CI for AUC: resample test set and recompute AUC
    rng_lr = np.random.default_rng(seed + 1)
    lr_auc_boots = []
    for _ in range(1000):
        idx = rng_lr.integers(0, len(y_real_te_cls), size=len(y_real_te_cls))
        lr_auc_boots.append(float(roc_auc_score(y_real_te_cls[idx], lr_prob[idx])))
    lr_auc_ci = {"mean": float(np.mean(lr_auc_boots)), "ci_low": float(np.quantile(lr_auc_boots, 0.025)), "ci_high": float(np.quantile(lr_auc_boots, 0.975))}

    # Regression: RFReg + Ridge, train on synthetic, test on real
    X_syn_reg, y_syn_reg = split_X_y(synthetic_train, reg_target, all_targets)
    X_real_te_reg, y_real_te_reg = split_X_y(real_test, reg_target, all_targets)

    spec_reg = infer_feature_spec(pd.concat([X_syn_reg, X_real_te_reg], axis=0))

    rf_reg = Pipeline([("pre", make_preprocess_pipeline(spec_reg)), ("reg", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1))])
    rf_reg.fit(X_syn_reg, y_syn_reg)
    rf_pred = rf_reg.predict(X_real_te_reg)
    rf_mae_vec = np.abs(rf_pred - y_real_te_reg)
    rf_mae = float(np.mean(rf_mae_vec))
    rf_mae_ci = bootstrap_ci(rf_mae_vec, random_state=seed)

    ridge_reg = Pipeline([("pre", make_preprocess_pipeline(spec_reg)), ("reg", Ridge(alpha=1.0))])
    ridge_reg.fit(X_syn_reg, y_syn_reg)
    ridge_pred = ridge_reg.predict(X_real_te_reg)
    ridge_mae_vec = np.abs(ridge_pred - y_real_te_reg)
    ridge_mae = float(np.mean(ridge_mae_vec))
    ridge_mae_ci = bootstrap_ci(ridge_mae_vec, random_state=seed)

    # TRTR ceiling: train on real_train, test on real_test
    X_real_tr_cls, y_real_tr_cls = split_X_y(real_train, class_target, all_targets)
    rf_trtr = Pipeline([("pre", make_preprocess_pipeline(spec_cls)), ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))])
    rf_trtr.fit(X_real_tr_cls, y_real_tr_cls)
    rf_trtr_prob = rf_trtr.predict_proba(X_real_te_cls)[:, 1]
    rf_trtr_auc = float(roc_auc_score(y_real_te_cls, rf_trtr_prob))
    # Compute TRTR per-sample log loss for statistical comparison with TSTR
    rf_trtr_logloss_vec = -np.log(np.clip(rf_trtr_prob * y_real_te_cls + (1 - rf_trtr_prob) * (1 - y_real_te_cls), 1e-15, 1.0))

    lr_trtr = Pipeline([("pre", make_preprocess_pipeline(spec_cls)), ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed))])
    lr_trtr.fit(X_real_tr_cls, y_real_tr_cls)
    lr_trtr_prob = lr_trtr.predict_proba(X_real_te_cls)[:, 1]
    lr_trtr_auc = float(roc_auc_score(y_real_te_cls, lr_trtr_prob))
    # Compute TRTR per-sample log loss for statistical comparison with TSTR
    lr_trtr_logloss_vec = -np.log(np.clip(lr_trtr_prob * y_real_te_cls + (1 - lr_trtr_prob) * (1 - y_real_te_cls), 1e-15, 1.0))

    X_real_tr_reg, y_real_tr_reg = split_X_y(real_train, reg_target, all_targets)
    rf_reg_trtr = Pipeline([("pre", make_preprocess_pipeline(spec_reg)), ("reg", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1))])
    rf_reg_trtr.fit(X_real_tr_reg, y_real_tr_reg)
    rf_trtr_pred = rf_reg_trtr.predict(X_real_te_reg)
    rf_trtr_mae_vec = np.abs(rf_trtr_pred - y_real_te_reg)
    rf_trtr_mae = float(np.mean(rf_trtr_mae_vec))

    ridge_reg_trtr = Pipeline([("pre", make_preprocess_pipeline(spec_reg)), ("reg", Ridge(alpha=1.0))])
    ridge_reg_trtr.fit(X_real_tr_reg, y_real_tr_reg)
    ridge_trtr_pred = ridge_reg_trtr.predict(X_real_te_reg)
    ridge_trtr_mae_vec = np.abs(ridge_trtr_pred - y_real_te_reg)
    ridge_trtr_mae = float(np.mean(ridge_trtr_mae_vec))

    return {
        "classification": {
            # TSTR metrics (train on synthetic, test on real) - PRIMARY UTILITY METRICS
            "rf_auc": rf_auc, "lr_auc": lr_auc, "mean_auc": float(np.mean([rf_auc, lr_auc])),
            "rf_auc_ci": rf_auc_ci, "lr_auc_ci": lr_auc_ci,
            # TRTR metrics (train on real, test on real) - PERFORMANCE CEILING
            "trtr_rf_auc": rf_trtr_auc, "trtr_lr_auc": lr_trtr_auc,
        },
        "regression": {
            # TSTR metrics (train on synthetic, test on real) - PRIMARY UTILITY METRICS
            "rf_mae": rf_mae, "ridge_mae": ridge_mae, "mean_mae": float(np.mean([rf_mae, ridge_mae])),
            "rf_mae_ci": rf_mae_ci, "ridge_mae_ci": ridge_mae_ci,
            # TRTR metrics (train on real, test on real) - PERFORMANCE CEILING
            "trtr_rf_mae": rf_trtr_mae, "trtr_ridge_mae": ridge_trtr_mae,
        },
        "per_sample": {
            "cls_logloss_rf": rf_logloss_vec.tolist(),
            "cls_logloss_lr": lr_logloss_vec.tolist(),
            "reg_abs_err_rf": rf_mae_vec.tolist(),
            "reg_abs_err_ridge": ridge_mae_vec.tolist(),
            # TRTR per-sample losses for statistical comparison
            "cls_logloss_rf_trtr": rf_trtr_logloss_vec.tolist(),
            "cls_logloss_lr_trtr": lr_trtr_logloss_vec.tolist(),
            "reg_abs_err_rf_trtr": rf_trtr_mae_vec.tolist(),
            "reg_abs_err_ridge_trtr": ridge_trtr_mae_vec.tolist(),
        }
    }

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples.
    
    Cohen's d interpretation (standard benchmarks):
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    
    For publication: Report alongside p-value to distinguish statistical
    significance (is there a difference?) from practical significance
    (does the difference matter?).
    """
    diff = a - b
    pooled_std = np.std(diff, ddof=1)
    if pooled_std == 0:
        return 0.0
    return float(np.mean(diff) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Human-readable interpretation of Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_permutation_test(a: np.ndarray, b: np.ndarray, *, n_perm: int = 10000, random_state: int = 0) -> Dict[str, float]:
    """Paired permutation test with 10,000 permutations for p-value resolution of 0.0001.
    
    Following best practices (Phipson & Smyth, 2010), we use 10,000 permutations
    to ensure sufficient statistical power and p-value resolution for publication.
    
    Returns both statistical significance (p-value) and practical significance
    (Cohen's d effect size) for comprehensive interpretation.
    """
    rng = np.random.default_rng(random_state)
    diff_obs = float(np.mean(a - b))
    diffs = []
    for _ in range(n_perm):
        sign = rng.choice([-1, 1], size=len(a))
        diffs.append(float(np.mean(sign * (a - b))))
    diffs = np.array(diffs)
    p = float(np.mean(np.abs(diffs) >= np.abs(diff_obs)))
    
    # Compute effect size for practical significance
    effect_size = cohens_d(a, b)
    
    return {
        "p_value": p, 
        "mean_diff": diff_obs, 
        "cohens_d": effect_size,
        "effect_interpretation": interpret_cohens_d(effect_size),
        "n_permutations": n_perm
    }


def plot_main_results(out_dir: Path) -> Path:
    import glob
    
    sd_files = list(out_dir.glob("sdmetrics__*.json"))
    c2_files = list(out_dir.glob("c2st__*.json"))
    mia_files = list(out_dir.glob("mia__*.json"))
    
    if not (sd_files and c2_files and mia_files):
        return out_dir
    
    sd_path = sd_files[0]
    c2_path = c2_files[0]
    mia_path = mia_files[0]
    synth_name = sd_path.stem.split("__")[1] if "__" in sd_path.stem else "unknown"

    if not (sd_path.exists() and c2_path.exists() and mia_path.exists()):
        return out_dir

    try:
        sd = json.loads(sd_path.read_text())
        c2 = json.loads(c2_path.read_text())
        mia = json.loads(mia_path.read_text())
    except Exception:
        return out_dir

    labels = ["Quality", "Realism (Eff. AUC)", "Privacy (Eff. AUC)"]
    values = [
        float(sd.get("overall_score", 0.0)) * 100.0,
        float(c2.get("effective_auc_mean", c2.get("auc_mean", 0.0))) * 100.0,
        float(mia.get("worst_case_effective_auc", 0.0)) * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"]) 
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_title(f"SYNTHLA-EDU V2: {synth_name.upper()}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    out_png = out_dir / "metrics_summary.png"
    try:
        fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)
    return out_png


def plot_model_comparison(dataset_dir: str | Path) -> Path:
    """Compare all synthesizer models for a given dataset.
    Prefers consolidated results.json (run-all output); falls back to per-synth JSONs.
    """
    dataset_dir = Path(dataset_dir)
    results_path = dataset_dir / "results.json"
    models = {}

    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            # Use dynamic synthesizer names from results (not hardcoded)
            for synth_name, metrics in results.get("synthesizers", {}).items():
                sd = metrics.get("sdmetrics", {})
                c2 = metrics.get("c2st", {})
                mia = metrics.get("mia", {})
                # Fallback chain for C2ST key (single-seed may vary)
                c2st_auc = float(c2.get("effective_auc", c2.get("c2st_effective_auc", 0.0)))
                models[synth_name] = {
                    "Quality": float(sd.get("overall_score", 0.0)) * 100.0,
                    "Realism": c2st_auc * 100.0,
                    "Privacy": float(mia.get("worst_case_effective_auc", 0.0)) * 100.0,
                }
        except Exception:
            pass
    else:
        # Fallback: search for individual synthesizer JSON files
        # Note: Uses standard synthesizer names as of publication date
        for synth_name in ["gaussian_copula", "ctgan", "tabddpm"]:
            sd_path = dataset_dir / f"sdmetrics__{synth_name}.json"
            c2_path = dataset_dir / f"c2st__{synth_name}.json"
            mia_path = dataset_dir / f"mia__{synth_name}.json"
            if not (sd_path.exists() and c2_path.exists() and mia_path.exists()):
                continue
            try:
                sd = json.loads(sd_path.read_text())
                c2 = json.loads(c2_path.read_text())
                mia = json.loads(mia_path.read_text())
                models[synth_name] = {
                    "Quality": float(sd.get("overall_score", 0.0)) * 100.0,
                    "Realism": float(c2.get("effective_auc_mean", c2.get("auc_mean", 0.0))) * 100.0,
                    "Privacy": float(mia.get("worst_case_effective_auc", 0.0)) * 100.0,
                }
            except Exception:
                continue

    if not models:
        return dataset_dir

    metrics = ["Quality", "Realism", "Privacy"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for i, (model_name, values) in enumerate(sorted(models.items())):
        vals = [values[m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model_name.upper(), color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    dataset_name = dataset_dir.name.replace("_runs", "").upper()
    ax.set_title(f"SYNTHLA-EDU V2: Model Comparison ({dataset_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    fig.tight_layout()

    out_png = dataset_dir / "model_comparison.png"
    try:
        fig.savefig(str(out_png), dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)
    return out_png


def create_cross_dataset_visualizations(
    figures_dir: Path,
    all_results: Dict[str, Dict[str, Any]],
    all_train_data: Dict[str, pd.DataFrame],
    all_synthetic_data: Dict[str, Dict[str, pd.DataFrame]]
) -> List[Path]:
    """Generate 10 publication-ready visualizations (separate per dataset where applicable).
    
    Figures generated:
        fig1: Classification Utility - OULAD
        fig2: Classification Utility - ASSISTMENTS
        fig3: Regression Utility - OULAD  
        fig4: Regression Utility - ASSISTMENTS
        fig5: Statistical Quality (SDMetrics)
        fig6: Privacy Preservation (MIA)
        fig7: Performance Heatmap - OULAD
        fig8: Performance Heatmap - ASSISTMENTS
        fig9: Per-Attacker Privacy - OULAD
        fig10: Per-Attacker Privacy - ASSISTMENTS
    
    Args:
        figures_dir: Directory to save figures
        all_results: Dict of {dataset_name: results_dict}
        all_train_data: Dict of {dataset_name: train_dataframe}
        all_synthetic_data: Dict of {dataset_name: {synth_name: synth_df}}
    
    Returns:
        List of saved figure paths
    """
    
    def score_from_effective_auc(eff_auc: float) -> float:
        """Convert effective AUC [0.5, 1.0] to score [100, 0] where 0.5→100 (ideal)."""
        return 200.0 * (1.0 - float(eff_auc))
    
    def score_from_mae(syn_mae: float, real_mae: float) -> float:
        """Convert MAE to relative score where matching/beating real→100."""
        syn_mae = max(float(syn_mae), 1e-12)
        real_mae = max(float(real_mae), 1e-12)
        return 100.0 * min(1.0, real_mae / syn_mae)
    
    ensure_dir(figures_dir)
    saved_figs = []
    
    # Publication-quality settings
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })
    
    # Color-blind friendly palette
    colors = {
        'gaussian_copula': '#DE8F05', 
        'ctgan': '#029E73',
        'tabddpm': '#CC78BC',
        'oulad': '#0173B2',
        'assistments': '#E69F00'
    }
    
    datasets = list(all_results.keys())
    synth_names = ['gaussian_copula', 'ctgan', 'tabddpm']
    
    # --- Figures 1-2: Classification Utility (Separate per dataset) ---
    for ds_idx, dataset in enumerate(datasets, 1):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            results = all_results[dataset]
            synths = results['synthesizers']
            
            # Get baseline and synthetic AUCs
            baseline_auc = synths[synth_names[0]]['utility']['classification']['trtr_rf_auc']
            models = ['Real'] + [s.replace('_', ' ').title() for s in synth_names]
            aucs = [baseline_auc] + [synths[s]['utility']['classification']['rf_auc'] for s in synth_names]
            
            bars = ax.bar(models, aucs, color=['#999999'] + [colors[s] for s in synth_names], 
                         edgecolor='black', linewidth=1.2)
            
            # Draw baseline with offset to avoid contact with bars
            bar_width = bars[0].get_width()
            x_min = bars[0].get_x() - bar_width * 0.3
            x_max = bars[-1].get_x() + bars[-1].get_width() + bar_width * 0.3
            
            ax.plot([x_min, x_max], [baseline_auc, baseline_auc], 
                    color='red', linestyle='--', linewidth=2.5, 
                    label=f'Real Baseline ({baseline_auc:.3f})', alpha=0.8, zorder=10)
            
            # Dynamic label offset
            for bar, val in zip(bars, aucs):
                if val < baseline_auc and (baseline_auc - val) < 0.02:
                    y_offset = 0.022
                elif abs(val - baseline_auc) < 0.005:
                    y_offset = 0.015
                else:
                    y_offset = 0.008
                
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
            ax.set_xlabel('', fontsize=13, fontweight='bold')
            ax.set_title(f'Classification Utility - {dataset.upper()} Dataset', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_ylim(0.5, 1.05)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
            ax.tick_params(axis='x', labelsize=13)
            
            fig.tight_layout()
            path = figures_dir / f'fig{ds_idx}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(path)
        except Exception as e:
            print(f"Warning: Could not create Figure {ds_idx} (Classification {dataset}): {e}")
    
    # --- Figures 3-4: Regression Utility (Separate per dataset) ---
    for ds_idx, dataset in enumerate(datasets, 3):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            results = all_results[dataset]
            synths = results['synthesizers']
            
            baseline_mae = synths[synth_names[0]]['utility']['regression']['trtr_ridge_mae']
            models = ['Real'] + [s.replace('_', ' ').title() for s in synth_names]
            maes = [baseline_mae] + [synths[s]['utility']['regression']['ridge_mae'] for s in synth_names]
            
            bars = ax.bar(models, maes, color=['#999999'] + [colors[s] for s in synth_names], 
                         edgecolor='black', linewidth=1.2)
            
            # Draw baseline with offset
            bar_width = bars[0].get_width()
            x_min = bars[0].get_x() - bar_width * 0.3
            x_max = bars[-1].get_x() + bars[-1].get_width() + bar_width * 0.3
            
            ax.plot([x_min, x_max], [baseline_mae, baseline_mae], 
                    color='red', linestyle='--', linewidth=2.5, 
                    label=f'Real Baseline ({baseline_mae:.2f})', alpha=0.8, zorder=10)
            
            # Dynamic offset based on data scale
            max_mae = max(maes)
            for bar, val in zip(bars, maes):
                distance_to_baseline = abs(val - baseline_mae)
                if distance_to_baseline < (max_mae * 0.05):
                    y_offset = max_mae * 0.025
                elif distance_to_baseline < (max_mae * 0.15):
                    y_offset = max_mae * 0.015
                else:
                    y_offset = max_mae * 0.01
                
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset, 
                       f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Mean Absolute Error', fontsize=13, fontweight='bold')
            ax.set_xlabel('', fontsize=13, fontweight='bold')
            ax.set_title(f'Regression Utility - {dataset.upper()} Dataset', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
            ax.tick_params(axis='x', labelsize=13)
            
            fig.tight_layout()
            path = figures_dir / f'fig{ds_idx}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(path)
        except Exception as e:
            print(f"Warning: Could not create Figure {ds_idx} (Regression {dataset}): {e}")
    
    # --- Figure 5: Statistical Quality (SDMetrics) ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(synth_names)) * 1.15
        width = 0.4
        
        for idx, dataset in enumerate(datasets):
            results = all_results[dataset]
            synths = results['synthesizers']
            quality_scores = [synths[s]['sdmetrics']['overall_score'] * 100 for s in synth_names]
            
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, quality_scores, width, 
                         label=dataset.upper(), color=colors[dataset], 
                         edgecolor='black', linewidth=1.2)
            
            for bar, val in zip(bars, quality_scores):
                if val > 95:
                    y_offset = -5
                    va = 'top'
                else:
                    y_offset = 1
                    va = 'bottom'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset, 
                       f'{val:.1f}%', ha='center', va=va, fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Quality Score (%)', fontsize=13, fontweight='bold')
        ax.set_xlabel('', fontsize=13, fontweight='bold')
        ax.set_title('Statistical Quality (SDMetrics)', fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in synth_names])
        ax.set_ylim(0, 100)
        ax.legend(frameon=True, shadow=True, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=13)
        
        fig.tight_layout()
        path = figures_dir / 'fig5.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(path)
    except Exception as e:
        print(f"Warning: Could not create Figure 5 (Quality): {e}")
    
    # --- Figure 6: Privacy Preservation (MIA) ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(synth_names)) * 1.15
        width = 0.4
        
        for idx, dataset in enumerate(datasets):
            results = all_results[dataset]
            synths = results['synthesizers']
            mia_scores = [synths[s]['mia']['worst_case_effective_auc'] for s in synth_names]
            
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, mia_scores, width, 
                         label=dataset.upper(), color=colors[dataset], 
                         edgecolor='black', linewidth=1.2)
            
            for bar, val in zip(bars, mia_scores):
                if abs(val - 0.5) < 0.02:
                    y_offset = 0.025
                else:
                    y_offset = 0.01
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2.5, 
                  label='Ideal Privacy (0.5)', alpha=0.8, zorder=10)
        ax.set_ylabel('MIA Effective AUC', fontsize=13, fontweight='bold')
        ax.set_xlabel('', fontsize=13, fontweight='bold')
        ax.set_title('Privacy Preservation: Membership Inference Attack', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in synth_names])
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=True, shadow=True, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=13)
        
        fig.tight_layout()
        path = figures_dir / 'fig6.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(path)
    except Exception as e:
        print(f"Warning: Could not create Figure 6 (Privacy): {e}")
    
    # --- Figures 7-8: Performance Heatmap (Separate per dataset) ---
    for ds_idx, dataset in enumerate(datasets, 7):
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            row_labels = []
            metrics_data = []
            metric_names = ['Quality', 'Realism', 'Privacy', 'Class. AUC', 'Regr. MAE']
            
            results = all_results[dataset]
            synths = results['synthesizers']
            baseline_mae = synths[synth_names[0]]['utility']['regression']['trtr_ridge_mae']
            
            for s in synth_names:
                # Format label: Gaussian Copula on two lines
                if s == 'gaussian_copula':
                    label = 'Gaussian\nCopula'
                else:
                    label = s.replace('_', ' ').title()
                row_labels.append(label)
                
                row = [
                    synths[s]['sdmetrics']['overall_score'] * 100,
                    score_from_effective_auc(synths[s]['c2st']['effective_auc']),
                    score_from_effective_auc(synths[s]['mia']['worst_case_effective_auc']),
                    synths[s]['utility']['classification']['rf_auc'] * 100,
                    score_from_mae(synths[s]['utility']['regression']['ridge_mae'], baseline_mae)
                ]
                metrics_data.append(row)
            
            metrics_array = np.array(metrics_data)
            
            im = ax.imshow(metrics_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            
            ax.set_xticks(np.arange(len(metric_names)))
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_xticklabels(metric_names, fontsize=13)
            ax.set_yticklabels(row_labels, fontsize=13)
            
            for i in range(len(row_labels)):
                for j in range(len(metric_names)):
                    ax.text(j, i, f'{metrics_array[i, j]:.1f}', 
                           ha='center', va='center', color='black', fontsize=11, fontweight='bold')
            
            ax.set_title(f'Performance Heatmap - {dataset.upper()} Dataset', 
                        fontsize=12, fontweight='bold', pad=15)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Score (%)', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            ax.set_xlabel('', fontsize=13, fontweight='bold')
            
            fig.tight_layout()
            path = figures_dir / f'fig{ds_idx}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(path)
        except Exception as e:
            print(f"Warning: Could not create Figure {ds_idx} (Heatmap {dataset}): {e}")
    
    # --- Figures 9-10: Per-Attacker Privacy (Separate per dataset) ---
    attacker_colors = {
        'logistic_regression': '#0173B2',
        'random_forest': '#E69F00',
        'xgboost': '#029E73'
    }
    
    for ds_idx, dataset in enumerate(datasets, 9):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            synths = all_results[dataset]['synthesizers']
            
            attackers = ['logistic_regression', 'random_forest', 'xgboost']
            if 'xgboost' not in synths[synth_names[0]]['mia']['attackers']:
                attackers = ['logistic_regression', 'random_forest']
            
            x = np.arange(len(synth_names)) * 1.15
            width = 0.25 if len(attackers) == 3 else 0.35
            
            for i, attacker in enumerate(attackers):
                scores = [synths[s]['mia']['attackers'][attacker]['effective_auc'] for s in synth_names]
                offset = (i - len(attackers)/2 + 0.5) * width
                bars = ax.bar(x + offset, scores, width, 
                            label=attacker.replace('_', ' ').title(), 
                            color=attacker_colors[attacker],
                            edgecolor='black', linewidth=1.2)
                
                for bar, score in zip(bars, scores):
                    if abs(score - 0.5) < 0.02:
                        y_offset = 0.025
                    else:
                        y_offset = 0.01
                    
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset, 
                           f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2.5, 
                      label='Ideal Privacy (0.5)', alpha=0.8, zorder=10)
            ax.set_ylabel('MIA Effective AUC', fontsize=13, fontweight='bold')
            ax.set_xlabel('', fontsize=13, fontweight='bold')
            ax.set_title(f'Privacy: Per-Attacker MIA Performance - {dataset.upper()} Dataset', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace('_', ' ').title() for s in synth_names])
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', labelsize=13)
            
            fig.tight_layout()
            path = figures_dir / f'fig{ds_idx}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(path)
        except Exception as e:
            print(f"Warning: Could not create Figure {ds_idx} (Per-Attacker {dataset}): {e}")
    
    # Reset to defaults
    plt.rcParams.update(plt.rcParamsDefault)
    
    return saved_figs


# -------------------------------
# Orchestration
# -------------------------------

def build_dataset(name: str, raw_dir: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    name_l = name.lower()
    raw_path = Path(raw_dir)
    if name_l == "assistments":
        # Try dataset-specific subfolder first, then direct path
        if (raw_path / "assistments").exists():
            return build_assistments_table(raw_path / "assistments")
        return build_assistments_table(raw_path)
    if name_l == "oulad":
        # Try dataset-specific subfolder first, then direct path
        if (raw_path / "oulad").exists():
            return build_oulad_student_table(raw_path / "oulad")
        return build_oulad_student_table(raw_path)
    raise ValueError(f"Unknown dataset '{name}'")


def split_dataset(df: pd.DataFrame, schema: Dict[str, Any], *, test_size: float, seed: int, stratify_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "group_col" in schema:
        res = group_train_test_split(df, group_col=schema["group_col"], test_size=test_size, random_state=seed)
        return res.train, res.test
    res = simple_train_test_split(df, test_size=test_size, random_state=seed, stratify_col=stratify_col)
    return res.train, res.test


def run_single(
    dataset: str,
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    test_size: float = 0.3,
    seed: int = 0,
    synthesizer_name: str = "gaussian_copula",
    aggregate_assistments: bool = False,
    quick: bool = False,
) -> Path:
    print("\n" + "="*70)
    print(f"SYNTHLA-EDU V2: Single Run")
    print("="*70)
    print(f"Dataset: {dataset.upper()} | Synthesizer: {synthesizer_name.upper()} | Quick: {quick}")
    print("="*70 + "\n")
    
    if quick:
        print("!" * 70)
        print("WARNING: QUICK MODE ENABLED - Results NOT suitable for publication!")
        print("Quick mode uses reduced training (CTGAN: 100 epochs, TabDDPM: 300 iter)")
        print("and may produce poor quality metrics (e.g., C2ST near 1.0).")
        print("Remove --quick flag for evaluation-quality results.")
        print("!" * 70 + "\n")
    
    out_path = ensure_dir(Path(out_dir))
    # Auto-clean old artifacts in this output folder
    print("Step 1/7: Cleaning previous results...")
    remove_glob(out_path, "sdmetrics__*.json")
    remove_glob(out_path, "c2st__*.json")
    remove_glob(out_path, "mia__*.json")
    remove_glob(out_path, "synthetic_train__*.parquet")
    remove_glob(out_path, "real_*.parquet")
    remove_if_exists(out_path / "metrics_summary.png")
    remove_if_exists(out_path / "schema.json")
    print("[OK] Cleanup complete\n")
    
    print("Step 2/7: Loading dataset...")
    df, schema = build_dataset(dataset, raw_dir)
    print(f"[OK] Loaded {len(df):,} rows with {len(df.columns)} columns\n")

    # Save real
    ds_out = out_path
    df.to_parquet(ds_out / "real_full.parquet", index=False)
    write_json(ds_out / "schema.json", schema)

    # Stratify by first target if present
    strat_col = next(iter(schema.get("target_cols", [])), None)
    train_df, test_df = split_dataset(df, schema, test_size=test_size, seed=seed, stratify_col=strat_col)
    train_df.to_parquet(ds_out / "real_train.parquet", index=False)
    test_df.to_parquet(ds_out / "real_test.parquet", index=False)

    # Configure synthesizers with quick mode parameters
    # Parameter choices based on: (1) Original papers, (2) Empirical convergence on educational data
    # CTGAN: 300 epochs standard (Xu et al. 2019), 100 for quick testing (3x speedup)
    # TabDDPM: 1200 iterations standard (Kotelnikov et al. 2023), 300 for quick (4x speedup)
    #   - TabDDPM requires more iterations due to diffusion process (forward+reverse)
    # Gaussian Copula: No iterative training (closed-form estimation)
    print(f"Step 4/7: Training {synthesizer_name.upper()}...")
    if synthesizer_name == "ctgan":
        params = {"epochs": 100 if quick else 300}
        print(f"> CTGAN with {params['epochs']} epochs (standard: 300, quick: 100)")
        synth_obj = CTGANSynth(**params)
    elif synthesizer_name == "tabddpm":
        params = {"n_iter": 300 if quick else 1200}
        print(f"> TabDDPM with {params['n_iter']} iterations (standard: 1200, quick: 300)")
        synth_obj = TabDDPMSynth(**params)
    else:
        print(f"> Gaussian Copula (no iterative training)")
        synth_obj = GaussianCopulaSynth()
    
    synth_name = synth_obj.name
    synth_obj.fit(train_df)
    print(f"> Sampling {len(train_df):,} rows with seed {seed}...")
    synthetic_data = synth_obj.sample(len(train_df), random_state=seed)
    print(f"[OK] Synthesis complete\n")
    synthetic_path = ds_out / f"synthetic_train__{synth_name}.parquet"
    remove_if_exists(synthetic_path)
    synthetic_data.to_parquet(synthetic_path, index=False)

    # ASSISTments is already aggregated to student-level by build_dataset
    test_eval = test_df
    synthetic_eval = synthetic_data

    print("Step 5/7: Running evaluations...")
    print("> SDMetrics Quality Report...")
    quality_metrics = sdmetrics_quality(test_eval, synthetic_eval)
    sd_path = ds_out / f"sdmetrics__{synth_name}.json"
    remove_if_exists(sd_path)
    write_json(sd_path, quality_metrics)

    print("> C2ST Realism Test...")
    # C2ST EXCLUSION POLICY: Exclude both ID and target columns
    # Rationale: C2ST tests distribution fidelity. Including targets would allow trivial
    # discrimination if target distributions differ between real and synthetic.
    # This is a conservative choice that focuses on feature distribution quality.
    c2st_exclude = schema.get("id_cols", []) + schema.get("target_cols", [])
    realism_metrics = c2st_effective_auc(test_df, synthetic_data, exclude_cols=c2st_exclude, test_size=0.3, seed=seed)
    c2_path = ds_out / f"c2st__{synth_name}.json"
    remove_if_exists(c2_path)
    write_json(c2_path, realism_metrics)

    print("> MIA Privacy Attack...")
    # MIA EXCLUSION POLICY: Exclude only ID columns, INCLUDE targets
    # Rationale: MIA simulates a worst-case attacker with access to all non-ID features.
    # Target values may be informative for membership inference in real-world scenarios.
    # This is a conservative privacy evaluation (harder for synthetic data to pass).
    exclude_cols = schema.get("id_cols", [])
    privacy_metrics = mia_worst_case_effective_auc(train_df, test_df, synthetic_data, exclude_cols=exclude_cols, test_size=0.3, random_state=seed, k=5)
    mia_path = ds_out / f"mia__{synth_name}.json"
    remove_if_exists(mia_path)
    write_json(mia_path, privacy_metrics)
    print("[OK] Evaluations complete\n")

    print("Step 6/7: Generating visualizations...")
    # Re-generate summary plot
    remove_if_exists(ds_out / "metrics_summary.png")
    plot_main_results(ds_out)
    print("[OK] Visualization saved\n")

    print("="*70)
    print("Run Complete!")
    print("="*70)
    print(f"Results: {out_path}")
    print("="*70 + "\n")
    
    return out_path


def run_all(raw_dir: str | Path, out_dir: str | Path, *, test_size: float = 0.3, seed: int = 0, quick: bool = False) -> Path:
    base_out = ensure_dir(Path(out_dir))
    datasets = ["oulad", "assistments"]
    # Using GC + CTGAN + TabDDPM
    synthesizers = ["gaussian_copula", "ctgan", "tabddpm"]

    # Set up automatic logging to capture all experiment details
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_out / f"experiment_log_{timestamp}.txt"
    
    # Tee output to both console and log file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = open(log_file, 'w', encoding='utf-8')
    sys.stdout = TeeOutput(original_stdout, log_handle)
    sys.stderr = TeeOutput(original_stderr, log_handle)

    print("\n" + "="*70)
    print("SYNTHLA-EDU V2: Full Experimental Matrix")
    print("="*70)
    print(f"Execution started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Datasets: {len(datasets)} | Synthesizers: {len(synthesizers)} | Quick Mode: {quick}")
    print(f"Total Experiments: {len(datasets) * len(synthesizers)}")
    print("="*70 + "\n")
    
    if quick:
        print("!" * 70)
        print("WARNING: QUICK MODE ENABLED - Results NOT suitable for publication!")
        print("Quick mode uses reduced training (CTGAN: 100 epochs, TabDDPM: 300 iter)")
        print("and may produce poor quality metrics (e.g., C2ST near 1.0).")
        print("Remove --quick flag for evaluation-quality results.")
        print("!" * 70 + "\n")

    # Store results from both datasets for cross-dataset visualization
    all_dataset_results = {}
    all_train_data = {}
    all_synthetic_data = {}

    for dataset_idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*70}")
        print(f"DATASET [{dataset_idx}/{len(datasets)}]: {dataset.upper()}")
        print(f"{'='*70}\n")
        
        ds_out = ensure_dir(base_out / dataset)
        
        # Clean previous results for this dataset - remove ALL old files
        print(f"[{dataset.upper()}] Step 1/7: Cleaning previous results...")
        # Remove specific known files
        for file in ["data.parquet", "results.json", "model_comparison.png"]:
            remove_if_exists(ds_out / file)
        # Remove all parquet, json, and png files (includes all fig*.png)
        remove_glob(ds_out, "*.parquet")
        remove_glob(ds_out, "*.json")
        remove_glob(ds_out, "*.png")
        print(f"[{dataset.upper()}] [OK] Cleanup complete\n")
        
        print(f"[{dataset.upper()}] Step 2/7: Loading and building dataset...")
        df, schema = build_dataset(dataset, raw_dir)
        print(f"[{dataset.upper()}] [OK] Loaded {len(df):,} rows with {len(df.columns)} columns\n")

        print(f"[{dataset.upper()}] Step 3/7: Splitting dataset (train/test)...")
        strat_col = next(iter(schema.get("target_cols", [])), None)
        train_df, test_df = split_dataset(df, schema, test_size=test_size, seed=seed, stratify_col=strat_col)
        print(f"[{dataset.upper()}] [OK] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows\n")

        # Consolidated storage rows
        all_rows = []
        rt_train = train_df.copy(); rt_train["split"] = "real_train"; rt_train["synthesizer"] = "real"
        rt_test = test_df.copy(); rt_test["split"] = "real_test"; rt_test["synthesizer"] = "real"
        all_rows.append(rt_train); all_rows.append(rt_test)

        # Collect environment metadata for reproducibility
        try:
            import pkg_resources
            library_versions = {
                "sdv": pkg_resources.get_distribution("sdv").version,
                "sdmetrics": pkg_resources.get_distribution("sdmetrics").version,
                "synthcity": pkg_resources.get_distribution("synthcity").version,
                "scikit-learn": pkg_resources.get_distribution("scikit-learn").version,
                "torch": pkg_resources.get_distribution("torch").version,
                "pandas": pkg_resources.get_distribution("pandas").version,
                "numpy": pkg_resources.get_distribution("numpy").version,
                "xgboost": pkg_resources.get_distribution("xgboost").version if _HAS_XGB else "not_installed",
            }
        except Exception:
            library_versions = {"error": "Could not retrieve library versions"}
        
        try:
            import psutil
            hardware_info = {
                "cpu_count": os.cpu_count(),
                "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception:
            hardware_info = {"cpu_count": os.cpu_count(), "ram_gb": "unknown", "gpu_available": torch.cuda.is_available()}
        
        results: Dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "quick_mode": quick,
            "split_sizes": {"train": int(len(train_df)), "test": int(len(test_df))},
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "libraries": library_versions,
                "hardware": hardware_info,
            },
            "timestamp": int(time.time()),
            "synthesizers": {},
            "pairwise_tests": {},
        }

        # Targets per dataset
        if dataset == "oulad":
            class_target, reg_target = "dropout", "final_grade"
        else:
            class_target, reg_target = "high_engagement", "student_pct_correct"
        
        # Add dataset metadata for transparency and reproducibility
        results["dataset_metadata"] = {
            "classification_target": class_target,
            "regression_target": reg_target,
            "id_columns": schema.get("id_cols", []),
            "group_column": schema.get("group_col"),
            "n_features": len([c for c in train_df.columns if c not in schema.get("target_cols", []) + schema.get("id_cols", [])]),
            "n_total_columns": len(train_df.columns),
        }
        
        # Document evaluation exclusion policies for paper Methods section
        c2st_exclude_list = schema.get("id_cols", []) + schema.get("target_cols", [])
        mia_exclude_list = schema.get("id_cols", [])
        results["evaluation_policies"] = {
            "c2st_exclusions": {
                "excluded_columns": c2st_exclude_list,
                "rationale": "C2ST evaluates distribution fidelity. Excluding targets prevents trivial discrimination if target distributions differ between real and synthetic data. This conservative choice focuses on feature distribution quality.",
                "n_features_evaluated": len([c for c in train_df.columns if c not in c2st_exclude_list])
            },
            "mia_exclusions": {
                "excluded_columns": mia_exclude_list,
                "rationale": "MIA evaluates worst-case privacy risk. Attackers may have access to all non-ID features including targets in real-world scenarios. This conservative evaluation represents the hardest privacy test.",
                "n_features_evaluated": len([c for c in train_df.columns if c not in mia_exclude_list])
            },
            "design_principle": "Different exclusion policies reflect different threat models: C2ST for fidelity (features only), MIA for privacy (all non-ID data)."
        }
        
        # Print target distribution diagnostics
        print(f"[{dataset.upper()}] Target Distributions:")
        if class_target in train_df.columns:
            class_counts = train_df[class_target].value_counts()
            class_ratio = class_counts.min() / class_counts.max()
            print(f"  - {class_target}: {dict(class_counts)} (balance ratio: {class_ratio:.3f})")
        if reg_target in train_df.columns:
            reg_vals = train_df[reg_target]
            print(f"  - {reg_target}: min={reg_vals.min():.2f}, max={reg_vals.max():.2f}, "
                  f"mean={reg_vals.mean():.2f}, std={reg_vals.std():.2f}")
        print()

        print(f"[{dataset.upper()}] Step 4/7: Training {len(synthesizers)} synthesizers...\n")
        
        # Loop synthesizers (all 3 now that PyTorch 2.9+ supports RMSNorm)
        per_sample_losses: Dict[str, Dict[str, np.ndarray]] = {}
        synthetic_datasets: Dict[str, pd.DataFrame] = {}  # Store for visualizations
        
        for synth_idx, synth_name in enumerate(synthesizers, 1):
            print(f"  [{dataset.upper()}] Synthesizer [{synth_idx}/{len(synthesizers)}]: {synth_name.upper()}")
            print(f"  {'-'*66}")
            
            set_seed(seed) # Ensure deterministic training
            
            # Training parameters based on original papers and empirical convergence
            # CTGAN: Xu et al. (2019) recommends 300 epochs; 100 for quick mode (3x faster)
            # TabDDPM: Kotelnikov et al. (2023) uses 1000-1500 iter; 1200 standard, 300 quick (4x faster)
            # Gaussian Copula: Closed-form (no iterative training)
            params = {}
            if synth_name == "ctgan":
                params["epochs"] = 100 if quick else 300
                print(f"  > Training CTGAN ({params['epochs']} epochs, standard=300)...")
            elif synth_name == "tabddpm":
                params["n_iter"] = 300 if quick else 1200
                print(f"  > Training TabDDPM ({params['n_iter']} iterations, standard=1200)...")
            else:
                print(f"  > Training Gaussian Copula (closed-form estimation)...")

            if synth_name == "ctgan":
                synth_obj = CTGANSynth(**params)
            elif synth_name == "tabddpm":
                synth_obj = TabDDPMSynth(**params)
            else:
                synth_obj = GaussianCopulaSynth(**params)

            # Capture actual fit timing
            fit_start = time.perf_counter()
            synth_obj.fit(train_df)
            fit_time = time.perf_counter() - fit_start
            
            print(f"  > Sampling {len(train_df):,} synthetic rows...")
            
            set_seed(seed) # Ensure deterministic sampling
            
            # Capture actual sample timing
            sample_start = time.perf_counter()
            syn = synth_obj.sample(len(train_df), random_state=seed)
            sample_time = time.perf_counter() - sample_start
            synthetic_datasets[synth_name] = syn  # Store for visualizations
            print(f"  [OK] Synthesis complete\n")

            syn_rows = syn.copy(); syn_rows["split"] = "synthetic_train"; syn_rows["synthesizer"] = synth_name
            all_rows.append(syn_rows)

            print(f"  > Running evaluations...")
            print(f"    - TSTR Utility (classification + regression)...")
            util = tstr_utility(train_df, test_df, syn, class_target=class_target, reg_target=reg_target, seed=seed)
            print(f"    - SDMetrics Quality Report...")
            qual = sdmetrics_quality(test_df, syn)
            print(f"    - C2ST Realism Test...")
            # Exclude both ID and target columns for fair realism comparison
            c2st_exclude = schema.get("id_cols", []) + schema.get("target_cols", [])
            c2 = c2st_effective_auc(test_df, syn, exclude_cols=c2st_exclude, test_size=0.3, seed=seed)
            print(f"    - MIA Privacy Attack...")
            mia = mia_worst_case_effective_auc(train_df, test_df, syn, exclude_cols=schema.get("id_cols", []), test_size=0.3, random_state=seed, k=5)
            print(f"  [OK] Evaluations complete\n")

            per_sample_losses[synth_name] = {
                "cls_logloss": np.array(util["per_sample"]["cls_logloss_rf"]),
                "reg_abs_err": np.array(util["per_sample"]["reg_abs_err_rf"]),
            }

            results["synthesizers"][synth_name] = {
                "sdmetrics": qual, 
                "c2st": c2, 
                "mia": mia, 
                "utility": util,
                "timing": {
                    "fit_seconds": round(fit_time, 2),
                    "sample_seconds": round(sample_time, 2),
                    "total_seconds": round(fit_time + sample_time, 2)
                }
            }

        print(f"[{dataset.upper()}] Step 5/7: Pairwise statistical significance tests...")
        
        # Pairwise synthesizer comparisons (TSTR only)
        pairs = [("ctgan", "tabddpm"), ("ctgan", "gaussian_copula"), ("tabddpm", "gaussian_copula")]
        for a, b in pairs:
            if a in per_sample_losses and b in per_sample_losses:
                print(f"  > Testing {a.upper()} vs {b.upper()} (10,000 permutations)...")
                cls_test = paired_permutation_test(per_sample_losses[a]["cls_logloss"], per_sample_losses[b]["cls_logloss"], n_perm=10000, random_state=seed)
                reg_test = paired_permutation_test(per_sample_losses[a]["reg_abs_err"], per_sample_losses[b]["reg_abs_err"], n_perm=10000, random_state=seed)
                results["pairwise_tests"][f"{a}_vs_{b}"] = {"classification": cls_test, "regression": reg_test}
        
        # TSTR vs TRTR comparisons (utility gap significance)
        print(f"  > Testing TSTR vs TRTR (utility gap significance)...")
        results["tstr_vs_trtr"] = {}
        for synth_name in synthesizers:
            if synth_name in results["synthesizers"]:
                util = results["synthesizers"][synth_name]["utility"]
                # Classification: TSTR vs TRTR
                tstr_cls_rf = np.array(util["per_sample"]["cls_logloss_rf"])
                trtr_cls_rf = np.array(util["per_sample"]["cls_logloss_rf_trtr"])
                cls_rf_test = paired_permutation_test(tstr_cls_rf, trtr_cls_rf, n_perm=10000, random_state=seed)
                
                tstr_cls_lr = np.array(util["per_sample"]["cls_logloss_lr"])
                trtr_cls_lr = np.array(util["per_sample"]["cls_logloss_lr_trtr"])
                cls_lr_test = paired_permutation_test(tstr_cls_lr, trtr_cls_lr, n_perm=10000, random_state=seed)
                
                # Regression: TSTR vs TRTR
                tstr_reg_rf = np.array(util["per_sample"]["reg_abs_err_rf"])
                trtr_reg_rf = np.array(util["per_sample"]["reg_abs_err_rf_trtr"])
                reg_rf_test = paired_permutation_test(tstr_reg_rf, trtr_reg_rf, n_perm=10000, random_state=seed)
                
                tstr_reg_ridge = np.array(util["per_sample"]["reg_abs_err_ridge"])
                trtr_reg_ridge = np.array(util["per_sample"]["reg_abs_err_ridge_trtr"])
                reg_ridge_test = paired_permutation_test(tstr_reg_ridge, trtr_reg_ridge, n_perm=10000, random_state=seed)
                
                results["tstr_vs_trtr"][synth_name] = {
                    "classification_rf": cls_rf_test,
                    "classification_lr": cls_lr_test,
                    "regression_rf": reg_rf_test,
                    "regression_ridge": reg_ridge_test
                }
        
        # Apply Bonferroni correction for multiple testing
        # Total tests: 
        # - Pairwise comparisons: 3 pairs × 2 metrics = 6
        # - TSTR vs TRTR: 3 synthesizers × 4 tests (cls_rf, cls_lr, reg_rf, reg_ridge) = 12
        # Total = 18 hypothesis tests
        all_p_values = []
        for pair_key, tests in results["pairwise_tests"].items():
            all_p_values.append(tests["classification"]["p_value"])
            all_p_values.append(tests["regression"]["p_value"])
        for synth_key, tests in results["tstr_vs_trtr"].items():
            all_p_values.append(tests["classification_rf"]["p_value"])
            all_p_values.append(tests["classification_lr"]["p_value"])
            all_p_values.append(tests["regression_rf"]["p_value"])
            all_p_values.append(tests["regression_ridge"]["p_value"])
        
        n_tests = len(all_p_values)
        adjusted_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
        
        results["multiple_testing_correction"] = {
            "method": "bonferroni",
            "original_alpha": 0.05,
            "adjusted_alpha": adjusted_alpha,
            "n_tests": n_tests,
            "test_breakdown": {
                "pairwise_comparisons": len(results["pairwise_tests"]) * 2,
                "tstr_vs_trtr": len(results["tstr_vs_trtr"]) * 4
            },
            "interpretation": f"Reject H0 if p < {adjusted_alpha:.4f} (Bonferroni-corrected threshold)",
            "rationale": "Controls family-wise error rate at α=0.05 across all hypothesis tests (pairwise + utility gap)"
        }
        
        # Mark which tests are significant after correction
        for pair_key, tests in results["pairwise_tests"].items():
            tests["classification"]["significant_bonferroni"] = tests["classification"]["p_value"] < adjusted_alpha
            tests["regression"]["significant_bonferroni"] = tests["regression"]["p_value"] < adjusted_alpha
        
        for synth_key, tests in results["tstr_vs_trtr"].items():
            tests["classification_rf"]["significant_bonferroni"] = tests["classification_rf"]["p_value"] < adjusted_alpha
            tests["classification_lr"]["significant_bonferroni"] = tests["classification_lr"]["p_value"] < adjusted_alpha
            tests["regression_rf"]["significant_bonferroni"] = tests["regression_rf"]["p_value"] < adjusted_alpha
            tests["regression_ridge"]["significant_bonferroni"] = tests["regression_ridge"]["p_value"] < adjusted_alpha
        
        print(f"[{dataset.upper()}] [OK] Statistical tests complete:")
        print(f"[{dataset.upper()}]   - Pairwise comparisons: {len(results['pairwise_tests']) * 2} tests")
        print(f"[{dataset.upper()}]   - TSTR vs TRTR: {len(results['tstr_vs_trtr']) * 4} tests")
        print(f"[{dataset.upper()}]   - Bonferroni-corrected α={adjusted_alpha:.4f}\n")

        print(f"[{dataset.upper()}] Step 6/7: Saving consolidated results...")
        data_parquet = pd.concat(all_rows, axis=0, ignore_index=True)
        data_parquet.to_parquet(ds_out / "data.parquet", index=False)
        write_json(ds_out / "results.json", results)
        print(f"[{dataset.upper()}] [OK] Saved data.parquet ({len(data_parquet):,} rows)")
        print(f"[{dataset.upper()}] [OK] Saved results.json\n")

        # Store for cross-dataset visualizations
        all_dataset_results[dataset] = results
        all_train_data[dataset] = train_df
        all_synthetic_data[dataset] = synthetic_datasets
        
        # Print per-dataset summary
        print(f"[{dataset.upper()}] {'='*70}")
        print(f"[{dataset.upper()}] SUMMARY: Synthesizer Performance")
        print(f"[{dataset.upper()}] {'='*70}")
        for synth_name in results["synthesizers"]:
            s = results["synthesizers"][synth_name]
            util_auc = s["utility"]["classification"]["mean_auc"]
            util_mae = s["utility"]["regression"]["mean_mae"]
            quality = s["sdmetrics"]["overall_score"]
            c2st = s["c2st"]["effective_auc"]
            mia = s["mia"]["worst_case_effective_auc"]
            print(f"[{dataset.upper()}] {synth_name:20s}: Utility AUC={util_auc:.3f}, MAE={util_mae:.2f} | "
                  f"Quality={quality:.3f}, C2ST={c2st:.3f}, MIA={mia:.3f}")
        print(f"[{dataset.upper()}] {'='*70}")
        print(f"[{dataset.upper()}] DATASET COMPLETE")
        print(f"[{dataset.upper()}] {'='*70}\n")

    # Generate cross-dataset visualizations after both datasets complete
    print("\n" + "="*70)
    print("Step 7/7: Generating Cross-Dataset Publication Visualizations")
    print("="*70)
    
    figures_dir = ensure_dir(base_out / "figures")
    
    # Clean previous figures to ensure consistent numbering (fig1-fig11)
    print("Cleaning previous figures...")
    remove_glob(figures_dir, "*.png")
    print("[OK] Cleanup complete\n")
    
    print("Creating 11 gold-standard cross-dataset comparison figures (fig1-fig11)...")
    print("  Note: Generates fig1-fig11 only. Stale fig9_computational_efficiency.png,")
    print("        fig10_per_attacker_privacy.png, fig12_correlation_matrices.png will be removed.")
    saved_figures = create_cross_dataset_visualizations(
        figures_dir, 
        all_dataset_results, 
        all_train_data, 
        all_synthetic_data
    )
    
    print(f"\n[OK] Generated {len(saved_figures)} publication-quality visualizations:")
    for fig_path in saved_figures:
        print(f"  - {fig_path.name}")

    # Calculate total execution time
    import datetime
    end_time = datetime.datetime.now()
    print(f"\nExecution completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "="*70)
    print("SYNTHLA-EDU V2: All Experiments Complete")
    print("="*70)
    print(f"Results saved to: {base_out}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Log file saved to: {log_file}")
    print("="*70 + "\n")
    
    # Close log file and restore stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()
    
    print(f"✓ Complete execution log saved to: {log_file}")
    
    return base_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file SYNTHLA-EDU runner (KISS)")
    parser.add_argument("--dataset", type=str, choices=["oulad", "assistments"], required=False)
    parser.add_argument("--raw-dir", type=str, required=False, help="Path to raw CSV folder for the dataset")
    parser.add_argument("--out-dir", type=str, required=False, help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthesizer", type=str, choices=["gaussian_copula", "ctgan", "tabddpm"], default="gaussian_copula", help="Synthesizer model to use")
    parser.add_argument("--run-all", action="store_true", help="Run full 2x3 matrix and write consolidated outputs per dataset")
    parser.add_argument("--quick", action="store_true", help="Reduce compute (fewer CTGAN epochs and TabDDPM iterations)")
    parser.add_argument("--aggregate-assistments", action="store_true", help="Aggregate ASSISTments to student-level for evaluation (if not already aggregated)")
    parser.add_argument("--compare", type=str, default=None, help="Dataset directory to generate model comparison chart from results.json")
    args = parser.parse_args()

    if args.run_all:
        if not (args.raw_dir and args.out_dir):
            parser.error("--raw-dir and --out-dir are required for --run-all")
        out = run_all(args.raw_dir, args.out_dir, test_size=args.test_size, seed=args.seed, quick=args.quick)
        print(f"Run-all completed. Outputs at: {out}")
        return

    if args.compare:
        comparison_path = plot_model_comparison(args.compare)
        print(f"Model comparison saved to: {comparison_path}")
        return
    
    if not (args.dataset and args.raw_dir and args.out_dir):
        parser.error("--dataset, --raw-dir, and --out-dir are required when not using --compare")

    out = run_single(
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        test_size=args.test_size,
        seed=args.seed,
        synthesizer_name=args.synthesizer,
        aggregate_assistments=args.aggregate_assistments,
        quick=args.quick,
    )
    print(f"Done. Results saved to: {out}")


if __name__ == "__main__":
    main()

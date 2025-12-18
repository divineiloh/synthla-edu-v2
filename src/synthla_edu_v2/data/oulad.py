from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


OULAD_REQUIRED_FILES = [
    "studentinfo",
    "studentregistration",
    "studentvle",
    "studentassessment",
    "assessments",
    "vle",
]


def _find_csv(raw_dir: Path, stem_lower: str) -> Path:
    # Search for a CSV whose filename (lowercased, no extension) contains stem_lower
    candidates = list(raw_dir.rglob("*.csv"))
    for p in candidates:
        name = p.stem.lower()
        if stem_lower in name:
            return p
    raise FileNotFoundError(f"Could not find CSV for '{stem_lower}' under {raw_dir}")


def load_raw_oulad(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    dfs: Dict[str, pd.DataFrame] = {}
    for key in OULAD_REQUIRED_FILES:
        path = _find_csv(raw_dir, key)
        dfs[key] = pd.read_csv(path)
    return dfs


def build_oulad_student_table(
    raw_dir: str | Path,
    *,
    min_vle_clicks_clip: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build a student-level tabular dataset from OULAD.

    Targets (per SYNTHLA-EDU V2 spec):
      - dropout: binary (Withdrawn=1 else 0)
      - final_grade: continuous (weighted mean of assessment scores)
    """
    dfs = load_raw_oulad(raw_dir)
    info = dfs["studentinfo"].copy()
    reg = dfs["studentregistration"].copy()
    svle = dfs["studentvle"].copy()
    sass = dfs["studentassessment"].copy()
    ass = dfs["assessments"].copy()

    # Base keys
    keys = ["code_module", "code_presentation", "id_student"]

    # --- Targets ---
    info["dropout"] = (info["final_result"].astype(str).str.lower() == "withdrawn").astype(int)

    # --- Registration features ---
    # date_registration can be negative (pre-course) in OULAD; keep as numeric.
    reg_feat = (
        reg.groupby(keys, as_index=False)
        .agg(
            date_registration=("date_registration", "min"),
            date_unregistration=("date_unregistration", "min"),
        )
    )
    reg_feat["is_unregistered"] = reg_feat["date_unregistration"].notna().astype(int)

    # --- VLE features ---
    svle["sum_click"] = svle["sum_click"].clip(lower=min_vle_clicks_clip)
    vle_feat = (
        svle.groupby(keys, as_index=False)
        .agg(
            total_vle_clicks=("sum_click", "sum"),
            mean_vle_clicks=("sum_click", "mean"),
            n_vle_records=("sum_click", "size"),
            n_vle_sites=("id_site", "nunique"),
            n_vle_days=("date", "nunique"),
        )
    )
    # Normalize by active days for stability
    vle_feat["clicks_per_active_day"] = vle_feat["total_vle_clicks"] / vle_feat["n_vle_days"].replace(0, np.nan)

    # --- Assessment / grade features ---
    # First add course keys to studentAssessment by joining with studentInfo
    sass = sass.merge(info[keys], on="id_student", how="left")
    
    # Join weights from assessments
    ass_weights = ass[keys[:-1] + ["id_assessment", "weight"]].copy()
    # The assessments file is keyed by (code_module, code_presentation, id_assessment)
    sass = sass.merge(ass_weights, on=["code_module", "code_presentation", "id_assessment"], how="left")

    # Weighted grade: sum(score * weight) / sum(weight) over submitted assessments.
    sass["score_x_weight"] = sass["score"] * sass["weight"]
    grade_feat = (
        sass.groupby(keys, as_index=False)
        .agg(
            n_assessments=("id_assessment", "nunique"),
            total_weight=("weight", "sum"),
            weighted_score_sum=("score_x_weight", "sum"),
            mean_score=("score", "mean"),
        )
    )
    grade_feat["final_grade"] = grade_feat["weighted_score_sum"] / grade_feat["total_weight"].replace(0, np.nan)

    # --- Merge all ---
    df = info.merge(reg_feat, on=keys, how="left").merge(vle_feat, on=keys, how="left").merge(grade_feat, on=keys, how="left")

    # Basic cleanup: cast categorical columns
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

    # Numeric cols: ensure float for downstream models and SDV metadata detection
    for c in df.columns:
        if c in keys or c in cat_cols:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(float)

    # Fill missing values for synthesizers that don't handle NaNs
    # VLE features: 0 if student never accessed VLE
    vle_cols = ["total_vle_clicks", "mean_vle_clicks", "n_vle_records", "n_vle_sites", "n_vle_days", "clicks_per_active_day"]
    for c in vle_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    
    # Assessment features: 0 if student never submitted assessments
    assess_cols = ["n_assessments", "total_weight", "weighted_score_sum", "mean_score", "final_grade"]
    for c in assess_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    
    # Registration: use large negative value for never-unregistered students
    if "date_unregistration" in df.columns:
        df["date_unregistration"] = df["date_unregistration"].fillna(-999.0)
    
    # Categorical: fill with "Unknown" or most frequent
    for c in cat_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].cat.add_categories(["Unknown"]).fillna("Unknown")

    # Provide schema hints (used by other modules)
    schema = {
        "id_cols": keys,
        "target_cols": ["dropout", "final_grade"],
        "categorical_cols": [c for c in cat_cols if c in df.columns],
    }
    return df, schema

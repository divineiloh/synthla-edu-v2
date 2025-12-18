from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_KEEP_COLS = [
    # identifiers
    "user_id",
    "problem_id",
    "skill_id",
    # features
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
    # target
    "correct",
]


def _find_assistments_csv(raw_dir: Path) -> Path:
    # Prefer combined dataset if present, else any CSV under raw_dir.
    candidates = list(raw_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    # heuristic: pick the largest CSV (often the main dataset)
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def load_raw_assistments(raw_dir: str | Path, *, encoding: str = "ISO-8859-15") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    path = _find_assistments_csv(raw_dir)
    return pd.read_csv(path, low_memory=False, encoding=encoding)


def build_assistments_table(
    raw_dir: str | Path,
    *,
    keep_cols: List[str] | None = None,
    encoding: str = "ISO-8859-15",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build an interaction-level tabular dataset from ASSISTments 2009-2010.

    Row unit: student-problem interaction (as provided in ASSISTments2009-2010 skill builder / combined datasets).

    Targets (per SYNTHLA-EDU V2 spec):
      - correct: binary (correct on first attempt; incorrect or asked for help)
      - student_pct_correct: continuous (student-level mean correctness)
    """
    df = load_raw_assistments(raw_dir, encoding=encoding)

    # Standardize column names to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    keep = keep_cols or DEFAULT_KEEP_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"ASSISTments missing expected columns: {missing}. Available: {list(df.columns)[:50]}...")

    df = df[keep].copy()

    # Ensure correct is binary 0/1
    df["correct"] = df["correct"].astype(int)

    # Student-level % correct (regression target), merged back as a column
    student_pct = df.groupby("user_id", as_index=False)["correct"].mean().rename(columns={"correct": "student_pct_correct"})
    df = df.merge(student_pct, on="user_id", how="left")

    # Cast categorical columns (where present)
    cat_cols = []
    for c in ["tutor_mode", "answer_type", "type", "first_action"]:
        if c in df.columns:
            # Convert to integer codes for SDV compatibility
            df[c] = pd.Categorical(df[c]).codes
            cat_cols.append(c)

    # Treat skill_id, problem_id, template_id as numeric codes for SDV compatibility
    for c in ["skill_id", "problem_id", "template_id"]:
        if c in df.columns:
            # Convert to numeric code to avoid SDV categorical transformer issues
            if df[c].dtype == 'object' or str(df[c].dtype) == 'string':
                df[c] = pd.factorize(df[c])[0]  # Convert to integer factorization
            else:
                df[c] = df[c].astype('int64')
            cat_cols.append(c)

    # user_id as id (do not cast to category by default to avoid huge categories), but safe
    df["user_id"] = df["user_id"].astype("int64")

    # Numeric hygiene
    for c in df.columns:
        if c in ["user_id", "correct"]:
            continue
        if str(df[c].dtype) == "Int64":
            df[c] = df[c].astype(float)

    schema = {
        "id_cols": ["user_id", "problem_id"],
        "group_col": "user_id",
        "target_cols": ["correct", "student_pct_correct"],
        "categorical_cols": cat_cols,
    }
    return df, schema


def aggregate_assistments_student_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate interaction-level ASSISTments to student-level features.

    Output columns:
      - user_id
      - n_interactions
      - n_unique_problems
      - n_unique_skills
      - mean_attempt_count, mean_ms_first_response, mean_hint_count, mean_hint_total, mean_overlap_time
      - student_pct_correct (mean of correct)
    """
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

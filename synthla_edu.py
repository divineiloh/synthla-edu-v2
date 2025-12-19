from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))


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

    missing = [c for c in DEFAULT_ASSIST_KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ASSISTments missing expected columns: {missing}")

    df = df[DEFAULT_ASSIST_KEEP_COLS].copy()

    df["correct"] = df["correct"].astype(int)
    student_pct = df.groupby("user_id", as_index=False)["correct"].mean().rename(columns={"correct": "student_pct_correct"})
    df = df.merge(student_pct, on="user_id", how="left")

    cat_cols: List[str] = []
    for c in ["tutor_mode", "answer_type", "type", "first_action"]:
        if c in df.columns:
            df[c] = pd.Categorical(df[c]).codes
            cat_cols.append(c)

    for c in ["skill_id", "problem_id", "template_id"]:
        if c in df.columns:
            if df[c].dtype == "object" or str(df[c].dtype) == "string":
                df[c] = pd.factorize(df[c])[0]
            else:
                df[c] = df[c].astype("int64")
            cat_cols.append(c)

    df["user_id"] = df["user_id"].astype("int64")

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


def load_raw_oulad(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    dfs: Dict[str, pd.DataFrame] = {}
    for key in OULAD_REQUIRED_FILES:
        path = _find_csv(raw_dir, key)
        dfs[key] = pd.read_csv(path)
    return dfs


def build_oulad_student_table(raw_dir: str | Path, *, min_vle_clicks_clip: float = 0.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dfs = load_raw_oulad(raw_dir)
    info = dfs["studentinfo"].copy()
    reg = dfs["studentregistration"].copy()
    svle = dfs["studentvle"].copy()
    sass = dfs["studentassessment"].copy()
    ass = dfs["assessments"].copy()

    keys = ["code_module", "code_presentation", "id_student"]

    info["dropout"] = (info["final_result"].astype(str).str.lower() == "withdrawn").astype(int)

    reg_feat = (
        reg.groupby(keys, as_index=False)
        .agg(
            date_registration=("date_registration", "min"),
            date_unregistration=("date_unregistration", "min"),
        )
    )
    reg_feat["is_unregistered"] = reg_feat["date_unregistration"].notna().astype(int)

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
    vle_feat["clicks_per_active_day"] = vle_feat["total_vle_clicks"] / vle_feat["n_vle_days"].replace(0, np.nan)

    sass = sass.merge(info[keys], on="id_student", how="left")
    ass_weights = ass[keys[:-1] + ["id_assessment", "weight"]].copy()
    sass = sass.merge(ass_weights, on=["code_module", "code_presentation", "id_assessment"], how="left")

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

    df = info.merge(reg_feat, on=keys, how="left").merge(vle_feat, on=keys, how="left").merge(grade_feat, on=keys, how="left")

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
# Synthesizer (Gaussian Copula via SDV)
# -------------------------------
class GaussianCopulaSynth:
    name = "gaussian_copula"

    def __init__(self, **kwargs) -> None:
        self.params = kwargs
        self._model = None

    def fit(self, df_train: pd.DataFrame) -> "GaussianCopulaSynth":
        try:
            from sdv.metadata import SingleTableMetadata  # type: ignore
            from sdv.single_table import GaussianCopulaSynthesizer  # type: ignore
            md = SingleTableMetadata()
            md.detect_from_dataframe(df_train)
            self._model = GaussianCopulaSynthesizer(md, **self.params)
            self._model.fit(df_train)
        except Exception:
            from sdv.tabular import GaussianCopula  # type: ignore
            self._model = GaussianCopula(**self.params)
            self._model.fit(df_train)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit() must be called before sample().")
        return self._model.sample(n)


# -------------------------------
# Evaluations: SDMetrics Quality, C2ST, MIA
# -------------------------------

def sdmetrics_quality(real: pd.DataFrame, syn: pd.DataFrame) -> Dict[str, Any]:
    try:
        from sdmetrics.reports.single_table import QualityReport  # type: ignore
    except Exception as e:
        raise ImportError("sdmetrics is required. Install with: pip install sdmetrics") from e
    report = QualityReport()
    report.generate(real_data=real, synthetic_data=syn)
    out: Dict[str, Any] = {"overall_score": float(report.get_score())}
    try:
        details = report.get_details()
        out["details"] = details.to_dict() if hasattr(details, "to_dict") else details
    except Exception:
        pass
    return out


def c2st_effective_auc(real: pd.DataFrame, syn: pd.DataFrame, *, test_size: float = 0.3, seeds: List[int] = [0, 1, 2, 3, 4]) -> Dict[str, Any]:
    n = min(len(real), len(syn))
    real_s = real.sample(n=n, random_state=seeds[0]).reset_index(drop=True)
    syn_s = syn.sample(n=n, random_state=seeds[0]).reset_index(drop=True)

    X = pd.concat([real_s, syn_s], ignore_index=True)
    y = np.concatenate([np.ones(len(real_s)), np.zeros(len(syn_s))])

    spec = infer_feature_spec(X)
    pre = make_preprocess_pipeline(spec)

    aucs = []
    effs = []
    for s in seeds:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=s, stratify=y)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=s)
        clf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)
        y_prob = proba[:, 1] if proba.shape[1] > 1 else (1.0 - proba[:, 0])
        auc = roc_auc_score(y_te, y_prob)
        eff = max(auc, 1.0 - auc)
        aucs.append(float(auc))
        effs.append(float(eff))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "effective_auc_mean": float(np.mean(effs)),
        "effective_auc_std": float(np.std(effs, ddof=1)) if len(effs) > 1 else 0.0,
        "seeds": seeds,
        "classifier": "rf",
        "n_per_class": int(n),
    }


def mia_worst_case_effective_auc(real_train: pd.DataFrame, real_holdout: pd.DataFrame, syn: pd.DataFrame, *, exclude_cols: Optional[List[str]] = None, test_size: float = 0.3, random_state: int = 0, k: int = 5) -> Dict[str, Any]:
    n = min(len(real_train), len(real_holdout))
    members = real_train.sample(n=n, random_state=random_state).reset_index(drop=True)
    nonmembers = real_holdout.sample(n=n, random_state=random_state).reset_index(drop=True)

    exclude = exclude_cols or []
    X_syn = syn.drop(columns=[c for c in exclude if c in syn.columns])
    X_mem = members.drop(columns=[c for c in exclude if c in members.columns])
    X_non = nonmembers.drop(columns=[c for c in exclude if c in nonmembers.columns])

    spec = infer_feature_spec(X_syn)
    pre = make_preprocess_pipeline(spec)

    Z_syn = pre.fit_transform(X_syn)
    Z_mem = pre.transform(X_mem)
    Z_non = pre.transform(X_non)

    nn = NearestNeighbors(n_neighbors=min(k, Z_syn.shape[0]), metric="euclidean", n_jobs=-1)
    nn.fit(Z_syn)
    d_mem, _ = nn.kneighbors(Z_mem, return_distance=True)
    d_non, _ = nn.kneighbors(Z_non, return_distance=True)

    X = pd.DataFrame({
        "d_min": np.concatenate([d_mem[:, 0], d_non[:, 0]]),
        "d_mean_k": np.concatenate([d_mem.mean(axis=1), d_non.mean(axis=1)]),
    })
    y = np.concatenate([np.ones(len(Z_mem)), np.zeros(len(Z_non))])

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    eff = max(auc, 1.0 - auc)
    return {
        "worst_case_effective_auc": float(eff),
        "attacker": "rf",
        "n_members": int(n),
        "k": int(k),
    }


# -------------------------------
# Orchestration
# -------------------------------

def build_dataset(name: str, raw_dir: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    name_l = name.lower()
    if name_l == "assistments":
        return build_assistments_table(raw_dir)
    if name_l == "oulad":
        return build_oulad_student_table(raw_dir)
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
    aggregate_assistments: bool = False,
) -> Path:
    out_path = ensure_dir(Path(out_dir))
    df, schema = build_dataset(dataset, raw_dir)

    # Save real
    ds_out = ensure_dir(out_path / dataset)
    df.to_parquet(ds_out / "real_full.parquet", index=False)
    write_json(ds_out / "schema.json", schema)

    # Stratify by first target if present
    strat_col = next(iter(schema.get("target_cols", [])), None)
    train_df, test_df = split_dataset(df, schema, test_size=test_size, seed=seed, stratify_col=strat_col)
    train_df.to_parquet(ds_out / "real_train.parquet", index=False)
    test_df.to_parquet(ds_out / "real_test.parquet", index=False)

    synth = GaussianCopulaSynth()
    synth.fit(train_df)
    syn = synth.sample(len(train_df))
    syn_path = ds_out / f"synthetic_train__gaussian_copula.parquet"
    syn.to_parquet(syn_path, index=False)

    # Optionally aggregate ASSISTments to student-level for utility-like evaluation
    test_eval = test_df
    syn_eval = syn
    if aggregate_assistments and dataset.lower() == "assistments":
        test_eval = aggregate_assistments_student_level(test_eval)
        syn_eval = aggregate_assistments_student_level(syn_eval)

    q = sdmetrics_quality(test_eval, syn_eval)
    write_json(ds_out / "sdmetrics__gaussian_copula.json", q)

    c2 = c2st_effective_auc(test_eval, syn_eval, test_size=0.3, seeds=[0, 1, 2, 3, 4])
    write_json(ds_out / "c2st__gaussian_copula.json", c2)

    exclude_cols = schema.get("id_cols", [])
    mia = mia_worst_case_effective_auc(train_df, test_df, syn, exclude_cols=exclude_cols, test_size=0.3, random_state=seed, k=5)
    write_json(ds_out / "mia__gaussian_copula.json", mia)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file SYNTHLA-EDU runner (KISS)")
    parser.add_argument("--dataset", type=str, choices=["oulad", "assistments"], required=True)
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw CSV folder for the dataset")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aggregate-assistments", action="store_true", help="Aggregate ASSISTments to student-level for evaluation")
    args = parser.parse_args()

    out = run_single(
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        test_size=args.test_size,
        seed=args.seed,
        aggregate_assistments=args.aggregate_assistments,
    )
    print(f"Done. Results saved to: {out}")


if __name__ == "__main__":
    main()

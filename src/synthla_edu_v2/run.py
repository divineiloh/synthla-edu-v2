from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import DatasetSpec, RunSpec, SynthSpec, UtilityTaskSpec, load_config
from .data.assistments import build_assistments_table, aggregate_assistments_student_level
from .data.oulad import build_oulad_student_table
from .data.split import group_train_test_split, simple_train_test_split
from .eval.c2st import c2st_effective_auc
from .eval.mia import run_mia_worst_case_auc
from .eval.quality import compute_sdmetrics_quality
from .eval.reporting import write_json, write_table_csv
from .eval.stats import bootstrap_ci, paired_permutation_test
from .eval.utility import run_utility
from .synth.sdv_wrappers import CTGANSynth, GaussianCopulaSynth
from .synth.tabddpm_wrappers import TabDDPMSynth
from .utils import df_signature, ensure_dir, save_json, set_global_seed


def _build_dataset(ds: DatasetSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    name = ds.name.lower()
    if name == "oulad":
        df, schema = build_oulad_student_table(ds.raw_path, **ds.params)
        return df, schema
    if name == "assistments":
        df, schema = build_assistments_table(ds.raw_path, **ds.params)
        return df, schema
    raise ValueError(f"Unknown dataset name '{ds.name}'")


def _split_dataset(df: pd.DataFrame, schema: Dict[str, Any], spec: RunSpec, task_stratify_col: str | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prefer group split if schema defines group_col (ASSISTments).
    if "group_col" in schema:
        res = group_train_test_split(df, group_col=schema["group_col"], test_size=spec.split.test_size, random_state=spec.split.random_state)
        return res.train, res.test
    # Otherwise simple split; allow stratification on task-specific target
    res = simple_train_test_split(df, test_size=spec.split.test_size, random_state=spec.split.random_state, stratify_col=task_stratify_col)
    return res.train, res.test


def _make_synth(s: SynthSpec):
    name = s.name.lower()
    if name == "gaussian_copula":
        return GaussianCopulaSynth(params=s.params)
    if name == "ctgan":
        return CTGANSynth(params=s.params)
    if name == "tabddpm":
        return TabDDPMSynth(params=s.params)
    raise ValueError(f"Unknown synthesizer '{s.name}'")


def run(config_path: str | Path) -> Path:
    spec = load_config(config_path)
    set_global_seed(spec.seed)

    # Ensure out_dir is clean (replace existing results)
    from shutil import rmtree
    out_dir = Path(spec.out_dir)
    if out_dir.exists():
        # remove all files/dirs inside out_dir to avoid duplicates from prior runs
        for child in out_dir.iterdir():
            if child.is_dir():
                rmtree(child)
            else:
                child.unlink()
    out_dir = ensure_dir(out_dir)

    # Logging
    from .utils import configure_logging
    logger = configure_logging(out_dir)
    logger.info(f"Loaded config from {config_path}; resolved out_dir={out_dir}")

    # Store run configuration for provenance
    from dataclasses import asdict
    save_json(out_dir / "config_resolved.json", {"config_path": str(config_path), "spec": asdict(spec)})
    logger.info("Wrote resolved configuration to disk")

    utility_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    c2st_rows: List[Dict[str, Any]] = []
    mia_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    perm_rows: List[Dict[str, Any]] = []

    for ds in spec.datasets:
        df, schema = _build_dataset(ds)

        # Save processed real data
        ds_out = ensure_dir(out_dir / ds.name)
        df.to_parquet(ds_out / "real_full.parquet", index=False)
        save_json(ds_out / "schema.json", schema)
        save_json(ds_out / "real_signature.json", df_signature(df))

        # --- Split (task stratification only applied to non-group datasets) ---
        # We'll create a single split per dataset using stratification on the first classification task target (if applicable).
        strat_col = None
        for t in spec.utility_tasks:
            if t.name == "classification" and t.target_col in df.columns and ds.name.lower() in t.params.get("datasets", [ds.name.lower(), ds.name]):
                strat_col = t.target_col
                break
        train_df, test_df = _split_dataset(df, schema, spec, task_stratify_col=strat_col)

        train_df.to_parquet(ds_out / "real_train.parquet", index=False)
        test_df.to_parquet(ds_out / "real_test.parquet", index=False)

        # For SDMetrics metadata: detect from training set using SDV (if available)
        metadata_dict = None
        try:
            from sdv.metadata import SingleTableMetadata  # type: ignore

            md = SingleTableMetadata()
            md.detect_from_dataframe(train_df)
            metadata_dict = md.to_dict()
        except Exception:
            metadata_dict = None

        # --- Fit and evaluate each synthesizer ---
        synth_predictions_cache: Dict[Tuple[str, str, str, str], Any] = {}  # (synth, task, model, metric) -> preds array

        for synth_spec in spec.synthesizers:
            synth = _make_synth(synth_spec)
            logger.info(f"Fitting synthesizer {synth.name} on dataset {ds.name} (n_train={len(train_df)})")
            synth.fit(train_df)
            logger.info(f"Sampling {len(train_df)} rows from synthesizer {synth.name}")
            syn = synth.sample(len(train_df))
            syn_path = ds_out / f"synthetic_train__{synth.name}.parquet"
            syn.to_parquet(syn_path, index=False)
            logger.info(f"Wrote synthetic train to {syn_path}")

            # Quality (SDMetrics)
            q = compute_sdmetrics_quality(test_df, syn, metadata_dict=metadata_dict)
            quality_rows.append(
                {
                    "dataset": ds.name,
                    "synthesizer": synth.name,
                    "sdmetrics_overall_score": q.get("overall_score"),
                }
            )
            logger.info(f"Computed SDMetrics for synthesizer {synth.name} (overall={q.get('overall_score')})")
            write_json(ds_out, f"sdmetrics__{synth.name}", q)

            # Realism (C2ST Effective AUC)
            logger.info(f"Running C2ST for synthesizer {synth.name}")
            c2 = c2st_effective_auc(test_df, syn, **spec.evaluation.c2st)
            c2st_rows.append(
                {
                    "dataset": ds.name,
                    "synthesizer": synth.name,
                    "c2st_effective_auc_mean": c2.get("effective_auc_mean"),
                    "c2st_effective_auc_std": c2.get("effective_auc_std"),
                    "c2st_n_per_class": c2.get("n_per_class"),
                    "c2st_classifier": c2.get("classifier"),
                }
            )
            write_json(ds_out, f"c2st__{synth.name}", c2)
            logger.info(f"C2ST result (mean={c2.get('effective_auc_mean')}, std={c2.get('effective_auc_std')})")
            # Privacy (MIA worst-case effective AUC)
            exclude_cols = schema.get("id_cols", [])
            logger.info(f"Running MIA for synthesizer {synth.name}")
            mia = run_mia_worst_case_auc(
                real_train=train_df,
                real_holdout=test_df,
                synthetic=syn,
                exclude_cols=exclude_cols,
                attacker_models=spec.evaluation.mia["attackers"],
                test_size=spec.evaluation.mia.get("test_size", 0.3),
                random_state=spec.seed,
                k=spec.evaluation.mia.get("k", 5),
            )
            mia_rows.append(
                {
                    "dataset": ds.name,
                    "synthesizer": synth.name,
                    "mia_worst_case_effective_auc": mia.get("worst_case_effective_auc"),
                    "mia_worst_model": mia.get("worst_model"),
                }
            )
            write_json(ds_out, f"mia__{synth.name}", mia)
            logger.info(f"MIA finished (worst_case_effective_auc={mia.get('worst_case_effective_auc')})")
            # Utility tasks (TSTR)
            for task in spec.utility_tasks:
                # Allow per-task dataset filter
                allowed = task.params.get("datasets", None)
                if allowed is not None:
                    allowed_norm = [str(x).lower() for x in allowed]
                    if ds.name.lower() not in allowed_norm:
                        continue

                syn_for_task = syn
                test_for_task = test_df
                if task.params.get("aggregate") == "assistments_student":
                    syn_for_task = aggregate_assistments_student_level(syn_for_task)
                    test_for_task = aggregate_assistments_student_level(test_for_task)

                res = run_utility(
                    task=task.name,
                    target_col=task.target_col,
                    id_cols=task.id_cols,
                    feature_drop_cols=task.feature_drop_cols,
                    model_names=task.models,
                    syn_train=syn_for_task,
                    real_test=test_for_task,
                    random_state=spec.seed,
                )

                row = {"dataset": ds.name, "synthesizer": synth.name, "task": task.name, "target": task.target_col}
                row.update(res.metrics)
                utility_rows.append(row)

                # Bootstrap CIs on the *mean* metric and per-model metrics
                boot_cfg = spec.evaluation.bootstrap
                if task.name == "classification":
                    metric_key = "mean_auc"
                    metric = "auc"
                else:
                    metric_key = "mean_mae"
                    metric = "mae"

                # Mean metric: use the mean of model predictions? Not defined.
                # Instead, compute CI per model and for the mean of metrics across models (scalar) using bootstrap over samples per model.
                for m in task.models:
                    if m not in res.predictions:
                        logger.warning(
                            "No predictions for model '%s' on task '%s' (dataset=%s, synth=%s); skipping bootstrap and permutation cache",
                            m,
                            task.name,
                            ds.name,
                            synth.name,
                        )
                        continue
                    pred = res.predictions[m]
                    ci = bootstrap_ci(res.y_true, pred, metric=metric, n_boot=boot_cfg["n_boot"], seed=boot_cfg.get("seed", spec.seed), alpha=boot_cfg.get("alpha", 0.05))
                    ci_rows.append(
                        {
                            "dataset": ds.name,
                            "synthesizer": synth.name,
                            "task": task.name,
                            "target": task.target_col,
                            "model": m,
                            "metric": metric,
                            "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"],
                        }
                    )
                    # cache predictions for permutation comparisons
                    synth_predictions_cache[(synth.name, task.name, m, metric)] = (res.y_true, pred)

        # --- Pairwise permutation tests between synthesizers (per dataset, per task, per model) ---
        # Compare all pairs among configured synthesizers.
        synth_names = [s.name.lower() for s in spec.synthesizers]
        for task in spec.utility_tasks:
            allowed = task.params.get("datasets", None)
            if allowed is not None:
                if ds.name.lower() not in [str(x).lower() for x in allowed]:
                    continue

            metric = "auc" if task.name == "classification" else "mae"
            for m in task.models:
                # all pairwise
                for i in range(len(synth_names)):
                    for j in range(i + 1, len(synth_names)):
                        a = synth_names[i]
                        b = synth_names[j]
                        key_a = (a, task.name, m, metric)
                        key_b = (b, task.name, m, metric)
                        if key_a not in synth_predictions_cache or key_b not in synth_predictions_cache:
                            continue
                        y_true_a, pred_a = synth_predictions_cache[key_a]
                        y_true_b, pred_b = synth_predictions_cache[key_b]
                        # sanity: y_true must match
                        if len(y_true_a) != len(y_true_b):
                            continue
                        perm = paired_permutation_test(
                            y_true_a,
                            pred_a.copy(),
                            pred_b.copy(),
                            metric=metric,
                            n_perm=spec.evaluation.permutation_test["n_perm"],
                            seed=spec.evaluation.permutation_test.get("seed", spec.seed),
                        )
                        perm_rows.append(
                            {
                                "dataset": ds.name,
                                "task": task.name,
                                "target": task.target_col,
                                "model": m,
                                "synth_a": a,
                                "synth_b": b,
                                "metric": metric,
                                "observed_delta": perm["observed_delta"],
                                "p_value_two_sided": perm["p_value_two_sided"],
                                "n_perm": perm["n_perm"],
                            }
                        )

    # Write summary tables
    write_table_csv(out_dir, "utility_results", utility_rows)
    write_table_csv(out_dir, "sdmetrics_results", quality_rows)
    write_table_csv(out_dir, "c2st_results", c2st_rows)
    write_table_csv(out_dir, "mia_results", mia_rows)
    write_table_csv(out_dir, "utility_bootstrap_cis", ci_rows)
    write_table_csv(out_dir, "utility_permutation_tests", perm_rows)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SYNTHLA-EDU V2 benchmark.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config, e.g., configs/full.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
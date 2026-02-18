#!/usr/bin/env python3
"""Rebuild Phase 2 organized tables from raw result CSVs.

This script regenerates the analysis-ready tables under:
  results/phase2_organized/tables/

It is intended to be re-run after you:
- re-train models, and/or
- append new rows to results/final_evaluation_metrics.csv,
so that downstream paper assets (LaTeX tables / figures) reflect the latest results.

Run:
  python scripts/build_phase2_organized_tables.py

Inputs:
- results/experiment_progress.csv
- results/final_evaluation_metrics.csv
- (optional) results/comprehensive_results_summary_*.csv

Outputs:
- results/phase2_organized/tables/phase2_*.csv

Notes:
- Baseline rows are normalized to ratio_key="N/A".
- Ratios are represented as strings: 0.5/0.6/…/1.0, plus N/A/auto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE2_DIR = RESULTS_DIR / "phase2_organized"
OUT_DIR = PHASE2_DIR / "tables"

RATIO_METHODS = {"RandomOverSampler", "SMOTE", "RandomUnderSampler", "ADASYN"}
RATIO_GRID = {"0.5", "0.6", "0.7", "0.8", "0.9", "1.0"}
AUTO_METHODS = {"TomekLinks", "NeighbourhoodCleaningRule", "SMOTETomek", "SMOTEENN"}


def ratio_to_key(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return ""
    if s in {"N/A", "auto"}:
        return s
    try:
        f = float(s)
        return f"{f:.1f}"
    except Exception:
        return s


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def build_joined_master() -> pd.DataFrame:
    # -------------------------
    # experiment_progress
    # -------------------------
    prog_path = RESULTS_DIR / "experiment_progress.csv"
    if not prog_path.exists():
        raise FileNotFoundError(f"Not found: {prog_path}")

    prog = pd.read_csv(prog_path, keep_default_na=False)
    prog["ratio_key"] = prog["ratio"].apply(ratio_to_key)

    # Baseline normalization
    mask_none = prog["sampler"].astype(str).eq("none") & prog["ratio_key"].eq("")
    prog.loc[mask_none, "ratio_key"] = "N/A"

    # Rebuild key (sanity)
    prog["experiment_key_rebuilt"] = prog.apply(
        lambda r: f"{r['seed']}_{r['ratio_key']}_{r['pathology_id']}_{r['sampler']}", axis=1
    )

    # Save progress copies
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prog.to_csv(OUT_DIR / "phase2_experiment_progress_all.csv", index=False)

    prog_completed = prog[prog["status"].astype(str) == "completed"].copy()
    prog_skipped = prog[prog["status"].astype(str) == "skipped"].copy()
    prog_completed.to_csv(OUT_DIR / "phase2_experiment_progress_completed.csv", index=False)
    prog_skipped.to_csv(OUT_DIR / "phase2_experiment_progress_skipped.csv", index=False)

    completed_keys = set(prog_completed["experiment_key"].astype(str).tolist())

    # -------------------------
    # final_evaluation_metrics
    # -------------------------
    eval_path = RESULTS_DIR / "final_evaluation_metrics.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Not found: {eval_path}")

    ev = pd.read_csv(eval_path, keep_default_na=False)

    # Normalize ratio_key from requested ratio
    if "sampling_ratio_requested" in ev.columns:
        ev["ratio_key"] = ev["sampling_ratio_requested"].apply(ratio_to_key)
    else:
        ev["ratio_key"] = ""

    # Normalize types
    for c in ["seed", "class_id"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("Int64")

    # Baseline normalization
    mask_none_ev = ev["sampler"].astype(str).eq("none")
    ev.loc[mask_none_ev, "ratio_key"] = "N/A"

    # experiment_key compatible with progress
    def build_key(r):
        try:
            return f"{int(r['seed'])}_{r['ratio_key']}_{int(r['class_id'])}_{r['sampler']}"
        except Exception:
            return ""

    ev["experiment_key"] = ev.apply(build_key, axis=1)

    # timestamp for dedup
    if "timestamp" in ev.columns:
        ev["timestamp_dt"] = pd.to_datetime(ev["timestamp"], errors="coerce")
    else:
        ev["timestamp_dt"] = pd.NaT

    # Normalize pathology display
    if "pathology" in ev.columns:
        ev["pathology_key"] = ev["pathology"].astype(str).str.replace(" ", "_", regex=False)
    else:
        ev["pathology_key"] = ""

    # Save raw copy with keys
    ev.drop(columns=["timestamp_dt"], errors="ignore").to_csv(
        OUT_DIR / "phase2_final_evaluation_metrics_with_keys.csv", index=False
    )

    # -------------------------
    # Reference set (auto methods; ratio=auto) extracted from evaluation metrics
    # -------------------------
    try:
        if "sampling_ratio_requested" in ev.columns:
            auto_rows = ev[
                ev["sampler"].astype(str).isin(AUTO_METHODS)
                & (ev["sampling_ratio_requested"].astype(str) == "auto")
            ].copy()
        else:
            auto_rows = ev[ev["sampler"].astype(str).isin(AUTO_METHODS)].copy()

        auto_rows.drop(columns=["timestamp_dt"], errors="ignore").to_csv(
            OUT_DIR / "phase2_reference_set_auto_methods_auto_ratio_rows.csv", index=False
        )

        # Summary grouped by (pathology, sampler, hyperparams) → mean±std across seeds
        to_numeric(
            auto_rows,
            ["auc", "auprc", "f1_score", "precision", "recall", "balanced_accuracy", "g_mean"],
        )

        # Ensure grouping keys exist
        for c in ["pathology", "class_id", "sampler", "sampling_hyperparams_parsed"]:
            if c not in auto_rows.columns:
                auto_rows[c] = ""

        auto_summary = (
            auto_rows.groupby(["pathology", "class_id", "sampler", "sampling_hyperparams_parsed"], as_index=False)
            .agg(
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                auc_n=("auc", "count"),
                auprc_mean=("auprc", "mean"),
                auprc_std=("auprc", "std"),
                auprc_n=("auprc", "count"),
                f1_mean=("f1_score", "mean"),
                f1_std=("f1_score", "std"),
                f1_n=("f1_score", "count"),
            )
        )
        auto_summary.to_csv(
            OUT_DIR / "phase2_reference_set_auto_methods_auto_ratio_summary_by_hyperparams.csv",
            index=False,
        )

        best_auto = (
            auto_summary.sort_values(
                ["pathology", "sampler", "auc_mean", "auprc_mean"],
                ascending=[True, True, False, False],
            )
            .groupby(["pathology", "sampler"], as_index=False)
            .head(1)
        )
        best_auto.to_csv(
            OUT_DIR / "phase2_reference_set_auto_methods_auto_ratio_best_by_pathology_and_sampler.csv",
            index=False,
        )
    except Exception:
        # Best-effort; do not block the main pipeline.
        pass

    # Split: matched vs extra
    valid_ev = ev[ev["experiment_key"].ne("")].copy()
    ev_in = valid_ev[valid_ev["experiment_key"].isin(completed_keys)].copy()
    ev_extra = valid_ev[~valid_ev["experiment_key"].isin(completed_keys)].copy()

    ev_in = ev_in.sort_values(["timestamp_dt"]).drop_duplicates(subset=["experiment_key"], keep="last")

    ev_in.drop(columns=["timestamp_dt"], errors="ignore").to_csv(
        OUT_DIR / "phase2_final_evaluation_metrics_ratio_grid_dedup.csv", index=False
    )
    ev_extra.drop(columns=["timestamp_dt"], errors="ignore").to_csv(
        OUT_DIR / "phase2_final_evaluation_metrics_extra_not_in_progress_completed.csv", index=False
    )

    # -------------------------
    # Join
    # -------------------------
    joined = prog_completed.merge(
        ev_in.drop(columns=["timestamp_dt"], errors="ignore"),
        on="experiment_key",
        how="left",
        suffixes=("_progress", "_eval"),
    )

    joined["pathology_display"] = joined["pathology_name"].astype(str).str.replace("_", " ", regex=False)

    joined.to_csv(OUT_DIR / "phase2_joined_progress_plus_eval_ratio_grid.csv", index=False)

    # Missing eval list
    if "auc" in joined.columns:
        joined_auc = pd.to_numeric(joined["auc"], errors="coerce")
        missing = joined[joined_auc.isna()].copy()
    else:
        missing = joined.copy()

    missing.to_csv(OUT_DIR / "phase2_missing_eval_metrics_for_completed_experiments.csv", index=False)

    # Optional: copy AUC-only summary if present
    for p in sorted(RESULTS_DIR.glob("comprehensive_results_summary_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        # Keep only the latest for now
        try:
            pd.read_csv(p, nrows=1)
            pd.read_csv(p, keep_default_na=False).to_csv(OUT_DIR / "phase2_comprehensive_results_summary_completed_auc_only.csv", index=False)
            break
        except Exception:
            continue

    return joined


def build_family_map() -> pd.DataFrame:
    family_map = {
        "none": ("baseline", False),
        "RandomOverSampler": ("oversampling", True),
        "SMOTE": ("oversampling", True),
        "ADASYN": ("oversampling", True),
        "RandomUnderSampler": ("undersampling", True),
        "TomekLinks": ("cleaning", False),
        "NeighbourhoodCleaningRule": ("cleaning", False),
        "SMOTETomek": ("hybrid", False),
        "SMOTEENN": ("hybrid", False),
    }

    # Build from observed samplers when possible
    joined = pd.read_csv(OUT_DIR / "phase2_joined_progress_plus_eval_ratio_grid.csv", keep_default_na=False)
    samplers = sorted(joined["sampler_progress"].astype(str).unique().tolist())

    rows = []
    for s in samplers:
        fam, ctrl = family_map.get(s, ("unknown", False))
        rows.append({"sampler": s, "family": fam, "ratio_controllable": ctrl})

    fam_df = pd.DataFrame(rows)
    fam_df.to_csv(OUT_DIR / "phase2_sampler_family_map.csv", index=False)
    return fam_df


def aggregate_tables(joined: pd.DataFrame, fam_df: pd.DataFrame) -> None:
    # Normalize numeric
    to_numeric(joined, ["auc", "auprc", "f1_score", "precision", "recall", "balanced_accuracy", "g_mean", "auc_score"])

    # Use only rows with evaluation
    df = joined.dropna(subset=["auc"]).copy()

    path_col = "pathology_display"
    sampler_col = "sampler_progress"
    ratio_col = "ratio_key_progress"

    # -------------------------
    # Baseline
    # -------------------------
    baseline = df[(df[sampler_col].astype(str) == "none") & (df[ratio_col].astype(str) == "N/A")].copy()
    base = (
        baseline.groupby(path_col, as_index=False)
        .agg(
            baseline_auc_mean=("auc", "mean"),
            baseline_auc_std=("auc", "std"),
            baseline_auc_n=("auc", "count"),
            baseline_auprc_mean=("auprc", "mean"),
            baseline_auprc_std=("auprc", "std"),
            baseline_auprc_n=("auprc", "count"),
            baseline_f1_mean=("f1_score", "mean"),
            baseline_f1_std=("f1_score", "std"),
            baseline_f1_n=("f1_score", "count"),
        )
    )
    base.to_csv(OUT_DIR / "phase2_baseline_metrics_by_pathology.csv", index=False)

    # -------------------------
    # Stage1 summary (baseline + ratio=0.5)
    # -------------------------
    stage1 = df[df[ratio_col].astype(str).isin(["N/A", "0.5"])].copy()
    s1 = (
        stage1.groupby([path_col, sampler_col], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_n=("auc", "count"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auprc_n=("auprc", "count"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
            f1_n=("f1_score", "count"),
        )
    )
    s1.to_csv(OUT_DIR / "phase2_stage1_auc_auprc_f1_summary_by_pathology_and_sampler.csv", index=False)

    # Best sampler by pathology (stage1)
    s1_nb = s1[s1[sampler_col].astype(str) != "none"].copy()
    best_s1 = (
        s1_nb.sort_values([path_col, "auc_mean", "auprc_mean"], ascending=[True, False, False])
        .groupby(path_col, as_index=False)
        .head(1)
    )
    best_s1.to_csv(OUT_DIR / "phase2_stage1_best_sampler_by_pathology.csv", index=False)

    # Improvements vs baseline (stage1)
    base_small = base[[path_col, "baseline_auc_mean", "baseline_auprc_mean", "baseline_f1_mean"]]
    s1_imp = s1.merge(base_small, on=path_col, how="left")
    s1_imp["auc_improvement_vs_baseline"] = s1_imp["auc_mean"] - s1_imp["baseline_auc_mean"]
    s1_imp["auprc_improvement_vs_baseline"] = s1_imp["auprc_mean"] - s1_imp["baseline_auprc_mean"]
    s1_imp["f1_improvement_vs_baseline"] = s1_imp["f1_mean"] - s1_imp["baseline_f1_mean"]
    s1_imp["is_baseline"] = s1_imp[sampler_col].astype(str).eq("none")
    s1_imp.to_csv(OUT_DIR / "phase2_stage1_improvement_vs_baseline_by_pathology_and_sampler.csv", index=False)

    # Best sampler vs baseline (stage1)
    best_s1_vs = (
        s1_imp[~s1_imp["is_baseline"]]
        .sort_values([path_col, "auc_mean", "auprc_mean"], ascending=[True, False, False])
        .groupby(path_col, as_index=False)
        .head(1)
    )
    best_s1_vs.to_csv(OUT_DIR / "phase2_stage1_best_sampler_vs_baseline.csv", index=False)

    # Robust variants (auc_n >= 2/3)
    for n in [2, 3]:
        sub = s1_nb[s1_nb["auc_n"] >= n].copy()
        best = (
            sub.sort_values([path_col, "auc_mean", "auprc_mean"], ascending=[True, False, False])
            .groupby(path_col, as_index=False)
            .head(1)
        )
        best.to_csv(OUT_DIR / f"phase2_stage1_best_sampler_vs_baseline_auc_n_ge_{n}.csv", index=False)

    # -------------------------
    # Stage2 ratio-grid summary
    # -------------------------
    stage2 = df[df[sampler_col].astype(str).isin(RATIO_METHODS) & df[ratio_col].astype(str).isin(RATIO_GRID)].copy()

    s2 = (
        stage2.groupby([path_col, sampler_col, ratio_col], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_n=("auc", "count"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auprc_n=("auprc", "count"),
        )
    )
    s2.to_csv(OUT_DIR / "phase2_stage2_auc_auprc_summary_by_pathology_sampler_ratio.csv", index=False)

    # Best ratio per pathology+sampler
    best_ratio = (
        s2.sort_values([path_col, sampler_col, "auc_mean", "auprc_mean"], ascending=[True, True, False, False])
        .groupby([path_col, sampler_col], as_index=False)
        .head(1)
    )
    best_ratio.to_csv(OUT_DIR / "phase2_stage2_best_ratio_by_pathology_and_sampler.csv", index=False)

    # Improvement best ratio vs 0.5
    base05 = (
        s2[s2[ratio_col].astype(str) == "0.5"]
        .rename(
            columns={
                "auc_mean": "auc_mean_at_0.5",
                "auc_std": "auc_std_at_0.5",
                "auc_n": "auc_n_at_0.5",
                "auprc_mean": "auprc_mean_at_0.5",
                "auprc_std": "auprc_std_at_0.5",
                "auprc_n": "auprc_n_at_0.5",
            }
        )
        .drop(columns=[ratio_col])
    )
    s2_imp = best_ratio.merge(base05, on=[path_col, sampler_col], how="left")
    s2_imp["auc_improvement_vs_0.5"] = s2_imp["auc_mean"] - s2_imp["auc_mean_at_0.5"]
    s2_imp["auprc_improvement_vs_0.5"] = s2_imp["auprc_mean"] - s2_imp["auprc_mean_at_0.5"]
    s2_imp.to_csv(OUT_DIR / "phase2_stage2_improvement_best_ratio_vs_0.5_by_pathology_and_sampler.csv", index=False)

    # Requested vs achieved ratio summary
    ra = joined[joined[sampler_col].astype(str).isin(RATIO_METHODS) & joined[ratio_col].astype(str).isin(RATIO_GRID)].copy()
    ra["requested_ratio"] = pd.to_numeric(ra[ratio_col], errors="coerce")
    ra["achieved_ratio"] = pd.to_numeric(ra.get("sampling_ratio_achieved"), errors="coerce")
    ra = ra.dropna(subset=["requested_ratio", "achieved_ratio"])
    ra["ratio_abs_error"] = (ra["achieved_ratio"] - ra["requested_ratio"]).abs()

    ra_sum = (
        ra.groupby([path_col, sampler_col, ratio_col], as_index=False)
        .agg(
            achieved_mean=("achieved_ratio", "mean"),
            achieved_std=("achieved_ratio", "std"),
            abs_error_mean=("ratio_abs_error", "mean"),
            abs_error_std=("ratio_abs_error", "std"),
            abs_error_max=("ratio_abs_error", "max"),
            n=("ratio_abs_error", "count"),
        )
    )
    ra_sum.to_csv(OUT_DIR / "phase2_stage2_ratio_achievement_summary.csv", index=False)

    # -------------------------
    # Best overall setting per pathology
    # -------------------------
    all_groups = (
        df.groupby([path_col, sampler_col, ratio_col], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_n=("auc", "count"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auprc_n=("auprc", "count"),
        )
    )

    best_overall = (
        all_groups.sort_values([path_col, "auc_mean", "auprc_mean"], ascending=[True, False, False])
        .groupby(path_col, as_index=False)
        .head(1)
    )
    best_overall = best_overall.merge(base[[path_col, "baseline_auc_mean", "baseline_auprc_mean"]], on=path_col, how="left")
    best_overall["auc_improvement_vs_baseline"] = best_overall["auc_mean"] - best_overall["baseline_auc_mean"]
    best_overall["auprc_improvement_vs_baseline"] = best_overall["auprc_mean"] - best_overall["baseline_auprc_mean"]
    best_overall.to_csv(OUT_DIR / "phase2_best_overall_setting_by_pathology.csv", index=False)

    for n in [2, 3]:
        best_overall[best_overall["auc_n"] >= n].to_csv(
            OUT_DIR / f"phase2_best_overall_setting_by_pathology_auc_n_ge_{n}.csv", index=False
        )

    # -------------------------
    # Family-level summaries
    # -------------------------
    s1f = s1.merge(fam_df, left_on=sampler_col, right_on="sampler", how="left")
    stage1_best_family = (
        s1f[s1f["family"] != "baseline"]
        .sort_values([path_col, "family", "auc_mean", "auprc_mean"], ascending=[True, True, False, False])
        .groupby([path_col, "family"], as_index=False)
        .head(1)
    )
    stage1_best_family.to_csv(OUT_DIR / "phase2_stage1_best_method_within_family_by_pathology.csv", index=False)

    stage1_family_avg = (
        s1f[s1f["family"] != "baseline"]
        .groupby([path_col, "family"], as_index=False)
        .agg(
            auc_mean_of_means=("auc_mean", "mean"),
            auprc_mean_of_means=("auprc_mean", "mean"),
            f1_mean_of_means=("f1_mean", "mean"),
            methods_in_family=(sampler_col, "nunique"),
        )
    )
    stage1_family_avg.to_csv(OUT_DIR / "phase2_stage1_family_average_by_pathology.csv", index=False)

    s2_best = pd.read_csv(OUT_DIR / "phase2_stage2_best_ratio_by_pathology_and_sampler.csv")
    s2f = s2_best.merge(fam_df, left_on=sampler_col, right_on="sampler", how="left")
    stage2_best_family = (
        s2f.sort_values([path_col, "family", "auc_mean", "auprc_mean"], ascending=[True, True, False, False])
        .groupby([path_col, "family"], as_index=False)
        .head(1)
    )
    stage2_best_family.to_csv(OUT_DIR / "phase2_stage2_best_method_within_family_by_pathology.csv", index=False)

    # -------------------------
    # Phase3 candidates (Top-3 settings)
    # -------------------------
    all_groups2 = all_groups.merge(base[[path_col, "baseline_auc_mean", "baseline_auprc_mean"]], on=path_col, how="left")
    all_groups2["auc_improvement_vs_baseline"] = all_groups2["auc_mean"] - all_groups2["baseline_auc_mean"]
    all_groups2["auprc_improvement_vs_baseline"] = all_groups2["auprc_mean"] - all_groups2["baseline_auprc_mean"]
    all_groups2 = all_groups2.merge(fam_df, left_on=sampler_col, right_on="sampler", how="left")

    non_base = all_groups2[~((all_groups2[sampler_col].astype(str) == "none") & (all_groups2[ratio_col].astype(str) == "N/A"))].copy()
    non_base = non_base.sort_values([path_col, "auc_mean", "auprc_mean"], ascending=[True, False, False])

    top3 = non_base.groupby(path_col, as_index=False).head(3)
    top3.to_csv(OUT_DIR / "phase2_phase3_candidates_top3_settings_by_pathology.csv", index=False)

    top3_n2 = non_base[non_base["auc_n"] >= 2].groupby(path_col, as_index=False).head(3)
    top3_n2.to_csv(OUT_DIR / "phase2_phase3_candidates_top3_settings_by_pathology_auc_n_ge_2.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    joined = build_joined_master()
    fam_df = build_family_map()
    aggregate_tables(joined, fam_df)

    print("✅ Phase2 organized tables rebuilt:")
    print(f"  output_dir: {OUT_DIR}")


if __name__ == "__main__":
    main()

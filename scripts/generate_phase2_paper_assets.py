#!/usr/bin/env python3
"""Generate thesis-ready tables/figures from organized Phase 2 results.

Inputs (generated previously):
- results/phase2_organized/tables/phase2_*.csv

Outputs:
- results/phase2_organized/paper_assets/tables/*.tex
- results/phase2_organized/paper_assets/figures/*.png

Run:
  python scripts/generate_phase2_paper_assets.py

Notes:
- This script intentionally uses the *organized* tables as a stable interface.
- When auc_n==1 (std undefined), tables show only mean and keep n explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Use non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['font.size'] = 14


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE2_DIR = PROJECT_ROOT / "results" / "phase2_organized"
TABLES_DIR = PHASE2_DIR / "tables"
ASSETS_DIR = PHASE2_DIR / "paper_assets"
FIG_DIR = ASSETS_DIR / "figures"
TEX_DIR = ASSETS_DIR / "tables"

RATIO_METHODS = ["RandomOverSampler", "SMOTE", "ADASYN", "RandomUnderSampler"]
RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PATH_JP = {
    'Cardiomegaly': '心拡大',
    'Edema': '浮腫',
    'Consolidation': '浸潤影',
    'Atelectasis': '無気肺',
    'Pleural Effusion': '胸水',
    'Average': '平均'
}

SAMPLER_JP = {
    'RandomOverSampler': 'ランダムオーバーサンプリング',
    'SMOTE': 'SMOTE',
    'ADASYN': 'ADASYN',
    'RandomUnderSampler': 'ランダムアンダーサンプリング'
}


def fmt_mean_std(mean: float, std: float | float("nan"), n: int, digits: int = 4) -> str:
    if pd.isna(mean):
        return "-"
    if n is None or pd.isna(n):
        n = 0
    try:
        n_int = int(n)
    except Exception:
        n_int = 0

    if n_int >= 2 and not pd.isna(std):
        return f"{mean:.{digits}f} \\pm {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Return a LaTeX table environment string.

    NOTE:
    - Some pandas versions always emit \\toprule/\\midrule/\\bottomrule.
      To keep the output LaTeX-portable without requiring \\usepackage{booktabs},
      we post-process them into \\hline.
    """
    # escape=False is required so that LaTeX sequences like `\\pm` are preserved.
    # (Text columns should be pre-escaped if they can contain LaTeX special chars.)
    body = df.to_latex(index=False, escape=False)
    body = (
        body.replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )
    # Wrap with table environment
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        + body
        + f"\\caption{{{caption}}}\n"
        + f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)


def build_tables() -> None:
    # -------------------------
    # Baseline table
    # -------------------------
    base = pd.read_csv(TABLES_DIR / "phase2_baseline_metrics_by_pathology.csv")
    out = pd.DataFrame(
        {
            "病変": base["pathology_display"],
            "AUC": [
                fmt_mean_std(m, s, n, 4)
                for m, s, n in zip(
                    base["baseline_auc_mean"],
                    base["baseline_auc_std"],
                    base["baseline_auc_n"],
                )
            ],
            "AUPRC": [
                fmt_mean_std(m, s, n, 4)
                for m, s, n in zip(
                    base["baseline_auprc_mean"],
                    base["baseline_auprc_std"],
                    base["baseline_auprc_n"],
                )
            ],
            "F1": [
                fmt_mean_std(m, s, n, 4)
                for m, s, n in zip(
                    base["baseline_f1_mean"],
                    base["baseline_f1_std"],
                    base["baseline_f1_n"],
                )
            ],
            "n": base["baseline_auc_n"].astype(int),
        }
    )

    tex = to_latex_table(
        out,
        caption="ベースライン性能（CXR8 テスト, 平均±標準偏差）",
        label="tab:phase2_baseline",
    )
    (TEX_DIR / "table_phase2_baseline.tex").write_text(tex, encoding="utf-8")

    # -------------------------
    # Best overall setting table (all)
    # -------------------------
    best_all = pd.read_csv(TABLES_DIR / "phase2_best_overall_setting_by_pathology.csv")
    out_all = pd.DataFrame(
        {
            "病変": best_all["pathology_display"],
            "最良手法": best_all["sampler_progress"],
            "比率": best_all["ratio_key_progress"],
            "AUC": [
                fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
                for m, s, n in zip(best_all["auc_mean"], best_all["auc_std"], best_all["auc_n"])
            ],
            "AUPRC": [
                fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
                for m, s, n in zip(best_all["auprc_mean"], best_all["auprc_std"], best_all["auprc_n"])
            ],
            "ΔAUC(対ベースライン)": best_all["auc_improvement_vs_baseline"].map(lambda x: f"{x:.4f}"),
            "ΔAUPRC(対ベースライン)": best_all["auprc_improvement_vs_baseline"].map(lambda x: f"{x:.4f}"),
        }
    )

    tex_all = to_latex_table(
        out_all,
        caption="病変別の最良設定（全データ, n を併記）",
        label="tab:phase2_best_overall_all",
    )
    (TEX_DIR / "table_phase2_best_overall_all.tex").write_text(tex_all, encoding="utf-8")

    # -------------------------
    # Best overall setting table (robust n>=2)
    # -------------------------
    robust_path = TABLES_DIR / "phase2_best_overall_setting_by_pathology_auc_n_ge_2.csv"
    if robust_path.exists():
        best_r = pd.read_csv(robust_path)
        out_r = pd.DataFrame(
            {
                "病変": best_r["pathology_display"],
                "最良手法": best_r["sampler_progress"],
                "比率": best_r["ratio_key_progress"],
                "AUC": [
                    fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
                    for m, s, n in zip(best_r["auc_mean"], best_r["auc_std"], best_r["auc_n"])
                ],
                "AUPRC": [
                    fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
                    for m, s, n in zip(best_r["auprc_mean"], best_r["auprc_std"], best_r["auprc_n"])
                ],
                "ΔAUC(対ベースライン)": best_r["auc_improvement_vs_baseline"].map(lambda x: f"{x:.4f}"),
                "ΔAUPRC(対ベースライン)": best_r["auprc_improvement_vs_baseline"].map(lambda x: f"{x:.4f}"),
            }
        )
        tex_r = to_latex_table(
            out_r,
            caption="病変別の最良設定（auc\_n ≥ 2 に限定）",
            label="tab:phase2_best_overall_robust",
        )
        (TEX_DIR / "table_phase2_best_overall_robust_auc_n_ge_2.tex").write_text(tex_r, encoding="utf-8")

    # -------------------------
    # Stage2 best ratio per sampler table
    # -------------------------
    s2_best = pd.read_csv(TABLES_DIR / "phase2_stage2_best_ratio_by_pathology_and_sampler.csv")
    s2_best = s2_best[s2_best["sampler_progress"].isin(RATIO_METHODS)].copy()
    s2_out = pd.DataFrame(
        {
            "病変": s2_best["pathology_display"],
            "手法": s2_best["sampler_progress"],
            "最良比率": s2_best["ratio_key_progress"],
            "AUC": [
                fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
                for m, s, n in zip(s2_best["auc_mean"], s2_best["auc_std"], s2_best["auc_n"])
            ],
            "AUPRC": [
                fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
                for m, s, n in zip(s2_best["auprc_mean"], s2_best["auprc_std"], s2_best["auprc_n"])
            ],
        }
    )
    tex_s2 = to_latex_table(
        s2_out,
        caption="比率グリッドにおける手法別の最良比率（CXR8 テスト）",
        label="tab:phase2_stage2_best_ratio",
    )
    (TEX_DIR / "table_phase2_stage2_best_ratio.tex").write_text(tex_s2, encoding="utf-8")

    # -------------------------
    # Phase3 candidate table (robust top3)
    # -------------------------
    cand = TABLES_DIR / "phase2_phase3_candidates_top3_settings_by_pathology_auc_n_ge_2.csv"
    if cand.exists():
        c = pd.read_csv(cand)
        c_out = c[[
            "pathology_display",
            "sampler_progress",
            "ratio_key_progress",
            "auc_mean",
            "auc_std",
            "auc_n",
            "auprc_mean",
            "auprc_std",
            "auprc_n",
            "auc_improvement_vs_baseline",
            "family",
        ]].copy()

        c_out.rename(
            columns={
                "pathology_display": "病変",
                "sampler_progress": "候補手法",
                "ratio_key_progress": "比率",
                "auc_mean": "AUC_mean",
                "auc_std": "AUC_std",
                "auc_n": "n",
                "auprc_mean": "AUPRC_mean",
                "auprc_std": "AUPRC_std",
                "auprc_n": "AUPRC_n",
                "auc_improvement_vs_baseline": "ΔAUC",
                "family": "系統",
            },
            inplace=True,
        )
        # formatting
        c_out["AUC"] = [
            fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
            for m, s, n in zip(c_out["AUC_mean"], c_out["AUC_std"], c_out["n"])
        ]
        c_out["AUPRC"] = [
            fmt_mean_std(m, s, n, 4) + f" (n={int(n)})"
            for m, s, n in zip(c_out["AUPRC_mean"], c_out["AUPRC_std"], c_out["AUPRC_n"])
        ]
        c_out["ΔAUC"] = c_out["ΔAUC"].map(lambda x: f"{x:.4f}")

        c_out = c_out[["病変", "候補手法", "比率", "AUC", "AUPRC", "ΔAUC", "系統"]]

        tex_c = to_latex_table(
            c_out,
            caption="Phase 3 に向けた候補設定（Top-3, auc\_n ≥ 2）",
            label="tab:phase3_candidates",
        )
        (TEX_DIR / "table_phase3_candidates_top3_auc_n_ge_2.tex").write_text(tex_c, encoding="utf-8")


def build_figures() -> None:
    # -------------------------
    # AUC vs ratio per pathology
    # -------------------------
    s2 = pd.read_csv(TABLES_DIR / "phase2_stage2_auc_auprc_summary_by_pathology_sampler_ratio.csv")
    s2 = s2[s2["sampler_progress"].isin(RATIO_METHODS)].copy()
    s2["ratio"] = pd.to_numeric(s2["ratio_key_progress"], errors="coerce")

    for pathology in sorted(s2["pathology_display"].unique().tolist()):
        path_data = s2[s2["pathology_display"] == pathology].copy()
        if path_data.empty:
            continue

        path_jp = PATH_JP.get(pathology, pathology)
        
        fig, ax = plt.subplots(figsize=(10, 8))

        for sampler in RATIO_METHODS:
            sampler_data = path_data[path_data["sampler_progress"] == sampler].sort_values("ratio")
            if sampler_data.empty:
                continue
            # Use Japanese sampler name in legend
            sampler_jp = SAMPLER_JP.get(sampler, sampler)
            ax.errorbar(
                sampler_data["ratio"],
                sampler_data["auc_mean"],
                yerr=sampler_data["auc_std"],
                marker="o",
                linewidth=2.5,
                markersize=8,
                capsize=4,
                label=sampler_jp,
            )

        ax.set_title(f"{path_jp}: 比率とAUCの関係", fontsize=28, fontweight='bold')
        ax.set_xlabel("サンプリング比率", fontsize=24)
        ax.set_ylabel("AUC", fontsize=24)
        ax.set_xticks(RATIOS)
        ax.tick_params(labelsize=20)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=18)
        fig.tight_layout()

        out = FIG_DIR / f"fig_phase2_stage2_auc_vs_ratio_{pathology.replace(' ', '_')}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)

    # -------------------------
    # Best overall improvement vs baseline (bar)
    # -------------------------
    best = pd.read_csv(TABLES_DIR / "phase2_best_overall_setting_by_pathology.csv")
    # Sort or align
    best = best.sort_values("pathology_display")
    best['path_jp'] = best['pathology_display'].map(PATH_JP).fillna(best['pathology_display'])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(best["path_jp"], best["auc_improvement_vs_baseline"], color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("最良設定における AUC 改善量 (対ベースライン)", fontsize=28, fontweight='bold')
    ax.set_ylabel("AUC改善量 (ΔAUC)", fontsize=24)
    ax.tick_params(axis="x", rotation=30, labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_phase2_best_setting_auc_improvement_vs_baseline.png", dpi=300)
    plt.close(fig)

    # -------------------------
    # Ratio achievement scatter (requested vs achieved)
    # -------------------------
    joined = pd.read_csv(TABLES_DIR / "phase2_joined_progress_plus_eval_ratio_grid.csv", keep_default_na=True)
    sub = joined[
        joined["sampler_progress"].isin(RATIO_METHODS)
        & joined["ratio_key_progress"].astype(str).isin([str(r) for r in RATIOS])
    ].copy()

    sub["requested_ratio"] = pd.to_numeric(sub["ratio_key_progress"], errors="coerce")
    sub["achieved_ratio"] = pd.to_numeric(sub["sampling_ratio_achieved"], errors="coerce")
    sub = sub.dropna(subset=["requested_ratio", "achieved_ratio"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for sampler in RATIO_METHODS:
        ss = sub[sub["sampler_progress"] == sampler]
        if ss.empty:
            continue
        # Use Japanese sampler name in legend
        sampler_jp = SAMPLER_JP.get(sampler, sampler)
        ax.scatter(ss["requested_ratio"], ss["achieved_ratio"], s=50, alpha=0.7, label=sampler_jp)

    x = np.array(RATIOS)
    ax.plot(x, x, linestyle="--", color="black", linewidth=1.5, label="理想線（y=x）")
    ax.set_title("要求比率と達成比率の比較", fontsize=22, fontweight='bold')
    ax.set_xlabel("要求比率", fontsize=18)
    ax.set_ylabel("達成比率", fontsize=18)
    ax.set_xticks(RATIOS)
    ax.set_yticks(RATIOS)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_phase2_ratio_requested_vs_achieved_scatter.png", dpi=300)
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    build_tables()
    build_figures()
    print(f"[SUCCESS] Paper assets written to: {ASSETS_DIR}")


if __name__ == "__main__":
    main()

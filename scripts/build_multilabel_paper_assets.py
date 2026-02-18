#!/usr/bin/env python3
"""
Generate LaTeX tables and figures for Chapter 6 (multi-label results).

Input:
  - results/multilabel/experiment_results_summary.csv

Output:
  - results/multilabel/paper_assets/tables/*.tex
  - results/multilabel/paper_assets/figures/*.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "multilabel")
CSV_PATH = os.path.join(RESULTS_DIR, "experiment_results_summary.csv")

OUT_TABLES_DIR = os.path.join(RESULTS_DIR, "paper_assets", "tables")
OUT_FIGS_DIR = os.path.join(RESULTS_DIR, "paper_assets", "figures")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']


APPROACH_NAME_JA = {
    "0": "0（ベースライン）",
    "A": "A（病変別最適サンプリング転用）",
    "B": "B（Manifold Mixup）",
    "C": "C（分布補正WeightedSampler）",
    "D": "D（ML-SMOTE）",
}


LABELS = [
    ("cardiomegaly", "心拡大"),
    ("edema", "浮腫"),
    ("consolidation", "浸潤影"),
    ("atelectasis", "無気肺"),
    ("pleural_effusion", "胸水"),
]


OVERALL_METRICS = [
    "cxr8_macro_auc",
    "cxr8_micro_auc",
    "cxr8_macro_auprc",
    "cxr8_micro_auprc",
]

OVERALL_METRICS_MACRO_ONLY = [
    "cxr8_macro_auc",
    "cxr8_macro_auprc",
]


def _fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _ensure_dirs() -> None:
    os.makedirs(OUT_TABLES_DIR, exist_ok=True)
    os.makedirs(OUT_FIGS_DIR, exist_ok=True)


def _load_df() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing input CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    # Normalize types
    df["approach"] = df["approach"].astype(str)
    df["params"] = df["params"].astype(str)
    return df


def _mark_duplicate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some ML-SMOTE (labelsetIR) experiments exactly reproduce baseline metrics.
    We flag rows that match baseline for the same seed across key metrics.
    """
    df = df.copy()
    df["is_duplicate_baseline"] = False

    base = df[(df["approach"] == "0") & (df["params"] == "{}")].set_index("seed")
    keym = [
        "cxr8_macro_auc",
        "cxr8_micro_auc",
        "cxr8_macro_auprc",
        "cxr8_micro_auprc",
        "cxr8_macro_f1",
        "cxr8_micro_f1",
    ]

    for i, r in df.iterrows():
        seed = r["seed"]
        if seed not in base.index:
            continue
        if all(abs(float(r[m]) - float(base.loc[seed, m])) < 1e-12 for m in keym):
            df.at[i, "is_duplicate_baseline"] = True
    return df


@dataclass(frozen=True)
class GroupStats:
    mean: pd.Series
    std: pd.Series
    n: int


def _group_stats(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> Dict[Tuple[str, ...], GroupStats]:
    out: Dict[Tuple[str, ...], GroupStats] = {}
    gb = df.groupby(group_cols)
    for k, g in gb:
        mean = g[metrics].mean(numeric_only=True)
        std = g[metrics].std(ddof=1, numeric_only=True)
        n = int(g["seed"].nunique()) if "seed" in g.columns else int(len(g))
        key = k if isinstance(k, tuple) else (k,)
        out[key] = GroupStats(mean=mean, std=std, n=n)
    return out


def _select_best_params(df: pd.DataFrame, approach: str, exclude_duplicates: bool = True) -> str:
    sub = df[df["approach"] == approach].copy()
    if exclude_duplicates and approach != "0":
        sub = sub[~sub["is_duplicate_baseline"]]
    g = sub.groupby("params")["cxr8_macro_auc"].mean()
    return str(g.idxmax())


def _write_table_best_by_approach(df: pd.DataFrame) -> str:
    """
    Table: baseline + best setting per approach (excluding duplicate-baseline rows).
    """
    best_params = {a: _select_best_params(df, a, exclude_duplicates=True) for a in sorted(df["approach"].unique())}

    # Compute stats for each chosen group
    rows = []
    for a in ["0", "A", "B", "C", "D"]:
        p = best_params[a]
        sub = df[(df["approach"] == a) & (df["params"] == p)]
        stats = _group_stats(sub, ["approach", "params"], OVERALL_METRICS_MACRO_ONLY)[(a, p)]
        rows.append((a, p, stats))

    out_path = os.path.join(OUT_TABLES_DIR, "table_multilabel_best_by_approach.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        # Keep the table compact to fit on one page.
        # Macro-only table to keep it readable and within margins.
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{6pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.15}\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write("\\begin{tabular}{p{0.35\\linewidth}p{0.35\\linewidth}ll}\n")
        f.write("\\hline\n")
        f.write("手法 & 代表設定（要約） & Macro AUC & Macro AUPRC \\\\\n")
        f.write("\\hline\n")
        for a, p, stats in rows:
            # Summarize params to keep the table readable.
            if a == "0":
                p_tex = "default"
            elif a == "A":
                p_tex = "use\\_phase2\\_defaults=False"
            elif a == "B":
                p_tex = "alpha=2.0, rare\\_weight=2.0"
            elif a == "C":
                p_tex = "mode=sqrt\\_inv, boost=2.0"
            elif a == "D":
                p_tex = "k=5, thr=meanIR"
            else:
                # Fallback: escape underscores at minimum.
                p_tex = str(p).replace("_", "\\_")
            f.write(
                f"{APPROACH_NAME_JA.get(a, a)} & {p_tex} & "
                f"{_fmt_pm(stats.mean['cxr8_macro_auc'], stats.std['cxr8_macro_auc'])} & "
                f"{_fmt_pm(stats.mean['cxr8_macro_auprc'], stats.std['cxr8_macro_auprc'])} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}}\n")
        f.write("\\caption{マルチラベル分類（CXR8テスト）における各approachの代表設定の比較（macro指標，平均$\\pm$標準偏差，3 seed）}\n")
        f.write("\\label{tab:multilabel_best_by_approach}\n")
        f.write("\\end{table}\n")
    return out_path


def _write_table_baseline_vs_bestC_per_label(df: pd.DataFrame) -> str:
    """
    Table: baseline vs best approach C, per-label AUC/AUPRC.
    """
    best_c = _select_best_params(df, "C", exclude_duplicates=True)

    cols = []
    for lab_key, _ in LABELS:
        cols.extend([f"{lab_key}_auc", f"{lab_key}_auprc"])

    base = df[(df["approach"] == "0") & (df["params"] == "{}")]
    cbest = df[(df["approach"] == "C") & (df["params"] == best_c)]

    base_stats = _group_stats(base, ["approach", "params"], cols)[("0", "{}")]
    c_stats = _group_stats(cbest, ["approach", "params"], cols)[("C", best_c)]

    out_path = os.path.join(OUT_TABLES_DIR, "table_multilabel_baseline_vs_bestC_per_label.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("病変 & Baseline AUC & Best-C AUC & Baseline AUPRC & Best-C AUPRC \\\\\n")
        f.write("\\hline\n")
        for lab_key, lab_name in LABELS:
            f.write(
                f"{lab_name} & "
                f"{_fmt_pm(base_stats.mean[f'{lab_key}_auc'], base_stats.std[f'{lab_key}_auc'])} & "
                f"{_fmt_pm(c_stats.mean[f'{lab_key}_auc'], c_stats.std[f'{lab_key}_auc'])} & "
                f"{_fmt_pm(base_stats.mean[f'{lab_key}_auprc'], base_stats.std[f'{lab_key}_auprc'])} & "
                f"{_fmt_pm(c_stats.mean[f'{lab_key}_auprc'], c_stats.std[f'{lab_key}_auprc'])} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{ベースラインとapproach C（最良設定）の病変別性能（CXR8テスト, 平均$\\pm$標準偏差）}\n")
        f.write("\\label{tab:multilabel_baseline_vs_bestC_per_label}\n")
        f.write("\\end{table}\n")
    return out_path


def _plot_best_by_approach_delta(df: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_params = {a: _select_best_params(df, a, exclude_duplicates=True) for a in sorted(df["approach"].unique())}
    base = df[(df["approach"] == "0") & (df["params"] == "{}")][OVERALL_METRICS].mean(numeric_only=True)

    order = ["A", "B", "C", "D"]
    # Simplified labels as requested
    labels = ["Binary Relevance", "Manifold Mixup", "WeightedSampler", "ML-SMOTE"]

    delta_auc = []
    delta_auprc = []
    for a in order:
        p = best_params[a]
        m = df[(df["approach"] == a) & (df["params"] == p)][OVERALL_METRICS].mean(numeric_only=True)
        delta_auc.append(float(m["cxr8_macro_auc"] - base["cxr8_macro_auc"]))
        delta_auprc.append(float(m["cxr8_macro_auprc"] - base["cxr8_macro_auprc"]))

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    # 図16: 「マクロ」を "macro" 表記へ統一
    ax.bar(x - w / 2, delta_auc, width=w, label="Δmacro平均AUC", color='#42A5F5', edgecolor='black', linewidth=1.2)
    ax.bar(x + w / 2, delta_auprc, width=w, label="Δmacro平均AUPRC", color='#66BB6A', edgecolor='black', linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Binary Relevance", "Manifold Mixup", "WeightedSampler", "ML-SMOTE"], fontsize=12, rotation=0, ha="center")
    ax.set_ylabel("ベースラインとの差分", fontsize=16)
    ax.set_title("マルチラベル分類：手法別の改善量（対ベースライン）", fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(OUT_FIGS_DIR, "fig_multilabel_delta_macro_auc_auprc_by_approach.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_bestC_per_label_delta(df: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_c = _select_best_params(df, "C", exclude_duplicates=True)
    base = df[(df["approach"] == "0") & (df["params"] == "{}")].mean(numeric_only=True)
    c = df[(df["approach"] == "C") & (df["params"] == best_c)].mean(numeric_only=True)

    labs = [name for _, name in LABELS]
    d_auc = []
    d_auprc = []
    for lab_key, _ in LABELS:
        d_auc.append(float(c[f"{lab_key}_auc"] - base[f"{lab_key}_auc"]))
        d_auprc.append(float(c[f"{lab_key}_auprc"] - base[f"{lab_key}_auprc"]))

    x = np.arange(len(labs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w / 2, d_auc, width=w, label="ΔAUC")
    ax.bar(x + w / 2, d_auprc, width=w, label="ΔAUPRC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=15, ha="right")
    ax.set_ylabel("差分（提案手法 − ベースライン）", fontsize=16)
    ax.set_title("病変別改善量（提案手法 対 ベースライン）", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(OUT_FIGS_DIR, "fig_multilabel_bestC_delta_per_label.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    _ensure_dirs()
    df = _load_df()
    df = _mark_duplicate_baseline(df)

    t1 = _write_table_best_by_approach(df)
    t2 = _write_table_baseline_vs_bestC_per_label(df)
    f1 = _plot_best_by_approach_delta(df)
    f2 = _plot_bestC_per_label_delta(df)

    print("Wrote:")
    print(" -", t1)
    print(" -", t2)
    print(" -", f1)
    print(" -", f2)


if __name__ == "__main__":
    main()


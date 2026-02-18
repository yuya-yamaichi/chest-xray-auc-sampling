#!/usr/bin/env python3
"""
Summarize evaluation results into organized CSVs for presentation.

Inputs:
- results/final_evaluation_metrics.csv (appended by src/pipelines/evaluate.py)

Outputs under results/evaluation_summary/:
- latest_per_model.csv         # dedup by model_path/model_name, keep latest
- top_by_pathology_auc.csv     # top-1 per pathology by AUC (ties → AUPRC → Recall)
- overall_leaderboard.csv      # sorted by AUC desc
"""

import os
import sys
import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
EVAL_FILE = RESULTS_DIR / 'final_evaluation_metrics.csv'
OUT_DIR = RESULTS_DIR / 'evaluation_summary'


def load_eval() -> pd.DataFrame:
    if not EVAL_FILE.exists():
        print(f"Evaluation file not found: {EVAL_FILE}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(EVAL_FILE)
        return df
    except Exception as e:
        print(f"Failed to read {EVAL_FILE}: {e}")
        return pd.DataFrame()


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')


def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer model_path if available; otherwise model_name
    key = 'model_path' if 'model_path' in df.columns else 'model_name'
    if key not in df.columns:
        return df
    # If timestamp present, sort by it; else rely on file order
    sort_cols = []
    if 'timestamp' in df.columns:
        sort_cols.append('timestamp')
    out = df.copy()
    if sort_cols:
        out = out.sort_values(sort_cols)
    # Keep last occurrence per key
    out = out.drop_duplicates(subset=[key], keep='last')
    return out


def top_by_pathology(df: pd.DataFrame) -> pd.DataFrame:
    if 'pathology' not in df.columns:
        return pd.DataFrame()
    cols = ['auc','auprc','recall']
    coerce_numeric(df, cols)
    df = df.sort_values(['pathology','auc','auprc','recall'], ascending=[True, False, False, False])
    top = df.groupby('pathology', as_index=False).head(1)
    return top


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['auc','auprc','f1_score','precision','recall','balanced_accuracy','g_mean']
    coerce_numeric(df, cols)
    out = df.sort_values(['auc','auprc','recall'], ascending=[False, False, False])
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_eval()
    if df.empty:
        print('No evaluation results to summarize')
        return

    latest = dedup_latest(df)
    latest.to_csv(OUT_DIR / 'latest_per_model.csv', index=False)

    top = top_by_pathology(latest)
    if not top.empty:
        top.to_csv(OUT_DIR / 'top_by_pathology_auc.csv', index=False)

    lb = leaderboard(latest)
    lb.to_csv(OUT_DIR / 'overall_leaderboard.csv', index=False)

    print(f"Summaries written to: {OUT_DIR}")


if __name__ == '__main__':
    main()



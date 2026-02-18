#!/usr/bin/env python3
"""
Build a "clean" analysis view from existing pathology training logs without re-training.

Rules:
- Ratio-grid analysis uses only ratio-controllable methods: RandomOverSampler, SMOTE, RandomUnderSampler
- Requires sampling_applied == True and sampling_strategy_used not in {failed, none}
- Cleaning/Hybrid (TomekLinks, NeighbourhoodCleaningRule, SMOTETomek, SMOTEENN) are kept only when ratio=='auto'
- Writes per-pathology cleaned CSVs and a merged summary under results/clean/
"""

import os
import sys
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
CLEAN_DIR = RESULTS_DIR / 'clean'

PATHOLOGY_LOGS = [
    'training_log_cardiomegaly.csv',
    'training_log_edema.csv',
    'training_log_consolidation.csv',
    'training_log_atelectasis.csv',
    'training_log_pleural effusion.csv',
    'training_log_pleural_effusion.csv',
]

RATIO_METHODS = { 'RandomOverSampler', 'SMOTE', 'RandomUnderSampler', 'ADASYN' }
AUTO_METHODS = { 'TomekLinks', 'NeighbourhoodCleaningRule', 'SMOTETomek', 'SMOTEENN' }


def load_logs() -> pd.DataFrame:
    dfs = []
    for name in PATHOLOGY_LOGS:
        f = RESULTS_DIR / name
        if f.exists():
            try:
                df = pd.read_csv(f)
                df['__source_file'] = name
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def normalize_ratio(v) -> str:
    if pd.isna(v):
        return 'auto'
    s = str(v).strip()
    return s if s else 'auto'


def build_clean_views(df: pd.DataFrame):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f'No logs found to clean under: {RESULTS_DIR}')
        return

    # Normalize fields
    df['sampling_ratio_str'] = df.get('sampling_ratio_requested', 'auto').apply(normalize_ratio)
    if 'sampling_applied' in df.columns:
        df['sampling_applied'] = df['sampling_applied'].map({True: True, False: False, 'True': True, 'False': False}).fillna(False)
    else:
        # Infer: applied if strategy is not failed/none/unknown
        df['sampling_applied'] = ~df['sampling_strategy_used'].isin(['failed','none','unknown'])
    df['sampler'] = df['sampler'].astype(str)
    df['sampling_strategy_used'] = df.get('sampling_strategy_used', '').astype(str)

    # Ratio grid: keep only ratio methods, non-auto ratios, applied==True, not failed/none
    is_ratio_row = df['sampling_ratio_str'].ne('auto')
    is_ratio_method = df['sampler'].isin(RATIO_METHODS)
    is_valid_sampling = df['sampling_applied'] & ~df['sampling_strategy_used'].isin(['failed', 'none', 'unknown'])
    clean_ratio = df[is_ratio_row & is_ratio_method & is_valid_sampling].copy()

    # Auto cleaning/hybrid: keep target methods with sampling applied; treat as auto regardless of requested ratio
    is_auto_method = df['sampler'].isin(AUTO_METHODS)
    clean_auto = df[is_auto_method & df['sampling_applied'] & df['sampling_strategy_used'].eq('auto')].copy()
    # Normalize ratio label to 'auto' for reporting consistency
    if len(clean_auto) > 0:
        clean_auto['sampling_ratio_str'] = 'auto'

    # Merge clean views
    clean_all = pd.concat([clean_ratio, clean_auto], ignore_index=True)

    # Write per-pathology cleaned logs
    wrote = 0
    for name in PATHOLOGY_LOGS:
        src = RESULTS_DIR / name
        if not src.exists():
            continue
        # Determine pathology by filename
        mask = clean_all['__source_file'].eq(name)
        sub = clean_all[mask].drop(columns=['__source_file'])
        if len(sub) == 0:
            continue
        out_name = name.replace(' ', '_')  # normalize spaces
        out_path = CLEAN_DIR / out_name
        sub.to_csv(out_path, index=False)
        wrote += 1

    # Write merged summary
    if len(clean_all) > 0:
        clean_all.drop(columns=['__source_file']).to_csv(CLEAN_DIR / 'merged_clean_results.csv', index=False)

    print(f"Clean views written: {wrote} pathology files; merged rows: {len(clean_all)} → {CLEAN_DIR}")


def main():
    df = load_logs()
    build_clean_views(df)


if __name__ == '__main__':
    main()



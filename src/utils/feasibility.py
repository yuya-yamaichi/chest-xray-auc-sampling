#!/usr/bin/env python3
"""
Feasibility utilities for sampling ratio requests.

Determines whether a requested sampling ratio is achievable for a given
pathology and sampler, based on the original class distribution in the
training set.

Conventions:
- Ratio r means minority_count_target = r * majority_count for oversampling methods.
- For RandomUnderSampler, majority_count_target = minority_count / r.
- Cleaning/Hybrid methods (TomekLinks/NCR/SMOTEENN/SMOTETomek) are treated as
  ratio-noncontrollable; feasibility is considered False when a numeric ratio
  is requested for them.

This module loads dataset root from the YAML config using utils.config.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd

from .config import load_config


PATHOLOGY_COLUMNS = [
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pleural Effusion'
]


def _parse_ratio_input(ratio_input: str) -> float:
    """Parse ratio string to float.

    Accepts formats like "auto", "0.5", "1:2". Raises ValueError for invalid.
    """
    if ratio_input == 'auto':
        raise ValueError("'auto' ratio is not numeric")
    if ':' in ratio_input:
        a, b = ratio_input.split(':', 1)
        return float(a) / float(b)
    return float(ratio_input)


def _load_train_csv_path(config) -> str:
    dataset_root = config.get('dataset.root', '/home/yamaichi/CheXpert/CheXpert-v1.0/')
    return os.path.join(dataset_root, 'train.csv')


def get_original_binary_counts(pathology_id: int) -> Tuple[int, int]:
    """Return counts of class 0 and 1 for the given pathology in the training set.

    Uncertain (-1) and NaN are treated as 0 to match the training-time handling.
    """
    config = load_config()
    csv_path = _load_train_csv_path(config)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CheXpert train.csv not found at: {csv_path}")

    if pathology_id < 0 or pathology_id >= len(PATHOLOGY_COLUMNS):
        raise ValueError(f"Invalid pathology_id: {pathology_id}")

    col = PATHOLOGY_COLUMNS[pathology_id]
    # Read lazily only the target column
    df = pd.read_csv(csv_path, usecols=[col])
    labels = df[col].fillna(0).replace(-1, 0).astype(int)

    num_pos = int((labels == 1).sum())
    num_neg = int((labels == 0).sum())
    return num_neg, num_pos  # returns (class_0_count, class_1_count)


def _minority_majority(count0: int, count1: int) -> Tuple[int, int]:
    if count0 <= count1:
        return count0, count1
    return count1, count0


def check_feasibility(sampler: str, ratio_input: str, pathology_id: int) -> Tuple[bool, str]:
    """Check if requested ratio is feasible for the sampler and pathology.

    Returns (is_feasible, reason). When infeasible, reason explains why.
    """
    # Methods that should not be ratio-controlled in ratio grid
    ratio_noncontrollable = {"TomekLinks", "NeighbourhoodCleaningRule", "SMOTETomek", "SMOTEENN"}

    if sampler == 'none':
        return True, 'sampler=none is ratio-independent'

    if sampler in ratio_noncontrollable:
        return False, f"sampler={sampler} is ratio-noncontrollable; run with ratio=auto in a separate track"

    # Parse ratio
    try:
        r = _parse_ratio_input(ratio_input)
    except Exception:
        return False, f"invalid ratio: {ratio_input}"

    # Load original distribution
    class0, class1 = get_original_binary_counts(pathology_id)
    minority_count, majority_count = _minority_majority(class0, class1)

    if minority_count == 0 or majority_count == 0:
        return False, "degenerate class distribution (one class count is zero)"

    # Feasibility rules
    if sampler in {"RandomOverSampler", "SMOTE", "ADASYN"}:
        desired_minority = int(r * majority_count)
        if desired_minority < minority_count:
            return False, (
                f"oversampling cannot reduce minority: current_minority={minority_count}, "
                f"desired_minority={desired_minority} (r={r:.3f}, majority={majority_count})"
            )
        return True, "feasible for oversampling"

    if sampler in {"RandomUnderSampler"}:
        # desired majority after undersampling
        desired_majority = int(minority_count / r) if r > 0 else majority_count + 1
        if desired_majority > majority_count:
            return False, (
                f"undersampling cannot increase majority: current_majority={majority_count}, "
                f"desired_majority={desired_majority} (r={r:.3f}, minority={minority_count})"
            )
        return True, "feasible for undersampling"

    # Unknown samplers: be conservative
    return False, f"unknown sampler: {sampler}"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check sampling ratio feasibility')
    parser.add_argument('--sampler', required=True)
    parser.add_argument('--ratio', required=True)
    parser.add_argument('--pathology_id', type=int, required=True)
    args = parser.parse_args()

    ok, reason = check_feasibility(args.sampler, args.ratio, args.pathology_id)
    print(reason)
    raise SystemExit(0 if ok else 1)



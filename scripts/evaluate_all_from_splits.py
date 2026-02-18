#!/usr/bin/env python3
"""
Evaluate all models listed in split training logs in a fixed pathology order

Order: cardiomegaly → edema → atelectasis → consolidation → pleural_effusion

Requirements:
- Use src/pipelines/evaluate.py for actual evaluation and result saving
- For ratio-independent samplers, persist their hyperparameters (e.g., NCR params)
- For ratio-based samplers, persist requested ratio

This script imports evaluate.py internals to ensure extra metadata can be
attached to the results row using evaluate.save_results_to_csv_file.
"""

import os
import sys
import csv
import json
import fcntl
import hashlib
from typing import Dict, List, Iterable, Optional


# Resolve project root from this file (scripts/ → project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ensure we can import src.pipelines.evaluate
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Import evaluate helpers
from pipelines.evaluate import (
    evaluate_single_model,
    save_results_to_csv_file,
    read_results_index,
)
from utils.config import load_config


def read_split_csv_loose(csv_path: str) -> Iterable[Dict[str, str]]:
    """
    Read a split training log CSV robustly, handling rows with extra columns
    (e.g., sampling_applied, sampling_hyperparams) that may not exist in the header.

    Strategy: If extra fields appear after 'sampling_strategy_used' and before the
    last column ('sampling_ratio_requested'), map them to known names so that
    downstream saving can persist these parameters.
    """
    if not os.path.exists(csv_path):
        return []

    rows: List[Dict[str, str]] = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return rows

        # Normalize header whitespace
        header = [h.strip() for h in header]
        # Identify anchor indices
        try:
            idx_strategy = header.index('sampling_strategy_used')
        except ValueError:
            idx_strategy = -1

        for raw in reader:
            if not raw:
                continue

            # Trim trailing empty fields introduced by formatting
            while len(raw) > 0 and (raw[-1] is None or str(raw[-1]).strip() == ''):
                # Keep empty last field if header expects it; otherwise drop
                if len(raw) <= len(header):
                    break
                raw.pop()

            # Basic map if equal length
            if len(raw) == len(header):
                row_map = {header[i]: raw[i] for i in range(len(header))}
                rows.append(row_map)
                continue

            # If we have extra fields and know where strategy column is, assume extras
            # were inserted after sampling_strategy_used and before the last header
            if idx_strategy != -1 and len(raw) > len(header):
                row_map: Dict[str, str] = {}
                # map up to strategy (inclusive)
                for i in range(0, min(idx_strategy + 1, len(raw))):
                    if i < len(header):
                        row_map[header[i]] = raw[i]

                # extras between strategy and the last column
                extras_count = len(raw) - len(header)
                extras_names = ['sampling_applied', 'sampling_hyperparams']
                for j in range(extras_count):
                    name = extras_names[j] if j < len(extras_names) else f'extra_field_{j}'
                    src_idx = idx_strategy + 1 + j
                    if src_idx < len(raw):
                        row_map[name] = raw[src_idx]

                # map the final header column from the last value
                # also map any header columns between strategy and last (if any ever appear)
                # here we only expect the last to be 'sampling_ratio_requested'
                if header:
                    row_map[header[-1]] = raw[-1]

                # Map any remaining header fields that are between idx_strategy and last,
                # if they exist in this dataset version (defensive)
                # Example: header = [..., 'sampling_strategy_used', 'X', 'sampling_ratio_requested']
                for k in range(idx_strategy + 1, len(header) - 1):
                    # Align from the tail if possible
                    # This is conservative and won't overwrite already set keys
                    if header[k] not in row_map:
                        # Heuristic: pull from corresponding position counting from end
                        tail_offset = (len(header) - 1) - k
                        src_idx = len(raw) - 1 - tail_offset
                        if 0 <= src_idx < len(raw):
                            row_map[header[k]] = raw[src_idx]

                rows.append(row_map)
                continue

            # Fallback: truncate or pad with empties to header length
            row_map = {header[i]: (raw[i] if i < len(raw) else '') for i in range(len(header))}
            rows.append(row_map)

    return rows


def parse_json_maybe(value: Optional[str]):
    """Try to parse JSON string; return original on failure."""
    if value is None:
        return value
    s = str(value).strip()
    if not s:
        return s
    # Quick gate: must look like a JSON object or array
    if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def evaluate_from_rows(rows: Iterable[Dict[str, str]], default_pathology: Optional[str] = None,
                       dataset_root: Optional[str] = None,
                       use_frontal_only_default: bool = False,
                       skip_if_exists: bool = True):
    """
    Iterate over parsed training-log rows and evaluate each model checkpoint.
    Persist extra metadata into the results CSV via save_results_to_csv_file.
    """
    # Build evaluated index once (if exists)
    results_csv_path = os.path.join(PROJECT_ROOT, 'results', 'final_evaluation_metrics.csv')
    lock_path = results_csv_path + '.lock'
    evaluated_index = read_results_index(results_csv_path) if skip_if_exists else None

    for row in rows:
        model_path = (row.get('model_path') or '').strip()
        if not model_path:
            continue

        # Skip duplicates if requested (re-check file each time to reflect concurrent writers)
        if skip_if_exists:
            evaluated_index = read_results_index(results_csv_path)
            if evaluated_index is not None and not evaluated_index.empty:
                try:
                    if 'model_path' in evaluated_index.columns and (evaluated_index['model_path'] == model_path).any():
                        print(f"- Skipping already evaluated model: {model_path}")
                        continue
                except Exception:
                    pass

        # Determine pathology for label selection (row value preferred)
        pathology = (row.get('pathology') or default_pathology or '').strip() or None

        # Per-model lock to avoid concurrent evaluation of the same checkpoint
        locks_dir = os.path.join(PROJECT_ROOT, 'results', 'eval_locks')
        os.makedirs(locks_dir, exist_ok=True)
        lock_name = hashlib.sha1(model_path.encode('utf-8')).hexdigest() + '.lock'
        model_lock_path = os.path.join(locks_dir, lock_name)

        try:
            model_lock_fp = open(model_lock_path, 'w')
        except Exception:
            # If we cannot create lock file, proceed best-effort
            model_lock_fp = None

        if model_lock_fp is not None:
            try:
                fcntl.flock(model_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Another process is evaluating the same model
                print(f"- Skipping (in progress elsewhere): {model_path}")
                try:
                    model_lock_fp.close()
                except Exception:
                    pass
                continue

        # Evaluate single model
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        try:
            metrics = evaluate_single_model(
                model_path=model_path,
                model_name=model_name,
                dataset_root=dataset_root,
                pathology_for_label=pathology,
                test_csv_explicit=None,
                use_frontal_only=use_frontal_only_default,
                norm_mean_override=None,
                norm_std_override=None,
                tta_mode='none',
                view_filter=None,
                bn_adapt=False,
                bn_adapt_passes=1,
                finding_labels_col='Finding Labels',
                split_list=None,
                cxr8_label_tag=None,
                images_root=None,
                cxr8_strict=False,
            )
        except Exception as e:
            # Release model lock on error before re-raising
            if model_lock_fp is not None:
                try:
                    fcntl.flock(model_lock_fp.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    model_lock_fp.close()
                except Exception:
                    pass
                try:
                    os.remove(model_lock_path)
                except Exception:
                    pass
            raise

        # Prepare extra info
        extra_cols: Dict[str, object] = {}

        # Common fields available in the original training logs
        for c in ['model_path', 'pathology', 'class_id', 'sampler', 'seed', 'best_val_auc', 'training_time_sec',
                  'original_class_0', 'original_class_1', 'final_class_0', 'final_class_1',
                  'sampling_ratio_achieved', 'sampling_strategy_used', 'sampling_ratio_requested', 'timestamp']:
            if c in row and row[c] not in (None, ''):
                extra_cols[c] = row[c]

        # Extra fields for parameter persistence
        # - Ratio-independent samplers: prefer sampling_hyperparams (JSON)
        # - Ratio-based samplers: already covered by sampling_ratio_requested
        if 'sampling_applied' in row and row['sampling_applied'] not in (None, ''):
            extra_cols['sampling_applied'] = row['sampling_applied']
        if 'sampling_hyperparams' in row and row['sampling_hyperparams'] not in (None, ''):
            parsed = parse_json_maybe(row['sampling_hyperparams'])
            # Save both raw and parsed forms for traceability
            extra_cols['sampling_hyperparams'] = row['sampling_hyperparams']
            if isinstance(parsed, (dict, list)):
                # Flatten simple dict as JSON string to be CSV-friendly
                extra_cols['sampling_hyperparams_parsed'] = json.dumps(parsed, ensure_ascii=False)

        # Serialize writes across concurrent processes using an OS lock
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        with open(lock_path, 'w') as lock_fp:
            try:
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
                save_results_to_csv_file(model_name, metrics, extra_info=extra_cols)
            finally:
                try:
                    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

        # Release per-model lock and remove the lock file
        if model_lock_fp is not None:
            try:
                fcntl.flock(model_lock_fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                model_lock_fp.close()
            except Exception:
                pass
            try:
                os.remove(model_lock_path)
            except Exception:
                pass


def main():
    # Load configuration to resolve dataset root and options
    CONFIG = load_config()
    dataset_root = CONFIG.get('dataset.test_dataset', '/home/yamaichi/CXR8/dataset/test_dataset/')
    use_frontal_only_default = CONFIG.get('dataset.use_frontal_only', False)

    # Define pathology order and corresponding split CSV pairs
    split_dir = os.path.join(PROJECT_ROOT, 'results', 'split')
    tasks = [
        ('Cardiomegaly', os.path.join(split_dir, 'training_log_cardiomegaly_ratio_independent.csv')),
        ('Cardiomegaly', os.path.join(split_dir, 'training_log_cardiomegaly_ratio_based.csv')),
        ('Edema', os.path.join(split_dir, 'training_log_edema_ratio_independent.csv')),
        ('Edema', os.path.join(split_dir, 'training_log_edema_ratio_based.csv')),
        ('Atelectasis', os.path.join(split_dir, 'training_log_atelectasis_ratio_independent.csv')),
        ('Atelectasis', os.path.join(split_dir, 'training_log_atelectasis_ratio_based.csv')),
        ('Consolidation', os.path.join(split_dir, 'training_log_consolidation_ratio_independent.csv')),
        ('Consolidation', os.path.join(split_dir, 'training_log_consolidation_ratio_based.csv')),
        ('Pleural Effusion', os.path.join(split_dir, 'training_log_pleural_effusion_ratio_independent.csv')),
        ('Pleural Effusion', os.path.join(split_dir, 'training_log_pleural_effusion_ratio_based.csv')),
    ]

    print('Evaluating models in fixed pathology order:')
    print('  cardiomegaly → edema → atelectasis → consolidation → pleural_effusion')
    print()

    for pathology, csv_path in tasks:
        print(f"==> Processing {pathology} | {os.path.relpath(csv_path, PROJECT_ROOT)}")
        rows = read_split_csv_loose(csv_path)
        if not rows:
            print(f"   (no rows or file missing)")
            continue
        evaluate_from_rows(
            rows,
            default_pathology=pathology,
            dataset_root=dataset_root,
            use_frontal_only_default=use_frontal_only_default,
            skip_if_exists=True,
        )

    print('\nAll evaluations attempted. Results in results/final_evaluation_metrics.csv')


if __name__ == '__main__':
    main()



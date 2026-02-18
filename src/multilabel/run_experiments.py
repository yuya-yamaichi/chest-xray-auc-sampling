#!/usr/bin/env python3
"""
Multi-label Experiment Runner with Parameter Optimization and CXR8 Evaluation

Runs all sampling approaches with parameter grids for comprehensive comparison.
Automatically evaluates each trained model on CXR8 test set.

Experiment Design:
- 5 Approaches (0, A, B, C, D)
- 5 Parameter settings per approach (or 1 for Baseline)
- 3 Seeds (42, 123, 456)
- Total: 63 experiments

Output Structure:
    results/multilabel/
    ├── models/
    │   └── approach_A/
    │       └── k_neighbors=5/
    │           ├── seed_42/model.pth
    │           ├── seed_123/model.pth
    │           └── seed_456/model.pth
    ├── evaluations/
    │   └── approach_A/
    │       └── k_neighbors=5/
    │           ├── seed_42_cxr8_eval.json
    │           ├── seed_123_cxr8_eval.json
    │           └── seed_456_cxr8_eval.json
    └── experiment_logs/
        └── experiment_summary.json

Usage:
    python run_experiments.py --all
    python run_experiments.py --approach A
    python run_experiments.py --approach 0 --dry_run
"""

import os
import sys
import argparse
import subprocess
import time
import json
import csv
from datetime import datetime
from typing import List, Dict, Any

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'multilabel')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
EVAL_DIR = os.path.join(RESULTS_DIR, 'evaluations')
LOG_DIR = os.path.join(RESULTS_DIR, 'experiment_logs')

# =====================================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================================

SEEDS = [42, 123, 456]

PARAM_GRIDS = {
    '0': [
        {}  # Baseline - no parameters
    ],
    'A': [
        {'use_phase2_defaults': True},
        {'use_phase2_defaults': True, 'k_neighbors': 3},
        {'use_phase2_defaults': True, 'k_neighbors': 5},
        {'use_phase2_defaults': True, 'k_neighbors': 7},
        {'use_phase2_defaults': False},
    ],
    'B': [
        {'alpha': 0.5, 'rare_weight': 1.0},
        {'alpha': 1.0, 'rare_weight': 1.0},
        {'alpha': 1.0, 'rare_weight': 2.0},
        {'alpha': 2.0, 'rare_weight': 2.0},
        {'alpha': 2.0, 'rare_weight': 3.0},
    ],
    'C': [
        {'reweight_mode': 'inverse', 'multi_label_boost': 1.0},
        {'reweight_mode': 'inverse', 'multi_label_boost': 2.0},
        {'reweight_mode': 'sqrt_inv', 'multi_label_boost': 1.5},
        {'reweight_mode': 'sqrt_inv', 'multi_label_boost': 2.0},
        {'reweight_mode': 'effective_num', 'multi_label_boost': 1.5},
    ],
    'D': [
        {'ml_smote_k': 3, 'ml_smote_threshold': 'meanIR'},
        {'ml_smote_k': 5, 'ml_smote_threshold': 'meanIR'},
        {'ml_smote_k': 5, 'ml_smote_threshold': 'labelsetIR'},
        {'ml_smote_k': 7, 'ml_smote_threshold': 'meanIR'},
        {'ml_smote_k': 7, 'ml_smote_threshold': 'labelsetIR'},
    ],
}


def get_param_dirname(params: Dict[str, Any]) -> str:
    """Generate directory name from parameters."""
    if not params:
        return 'default'
    return '_'.join(f'{k}={v}' for k, v in sorted(params.items()))


def get_experiment_id(approach: str, params: Dict[str, Any], seed: int) -> str:
    """Generate unique experiment ID."""
    param_str = get_param_dirname(params)
    return f"approach_{approach}_{param_str}_seed_{seed}"


def build_train_command(approach: str, seed: int, params: Dict[str, Any], 
                        output_dir: str, dry_run: bool = False) -> List[str]:
    """Build command for training."""
    cmd = [
        'python3', 'src/multilabel/train_multilabel.py',
        '--approach', approach,
        '--seed', str(seed),
        '--output_dir', output_dir,
        '--experiment_name', f'{get_param_dirname(params)}_seed{seed}',
    ]
    
    for key, value in params.items():
        if key == 'use_phase2_defaults':
            # Always explicitly pass this flag
            cmd.extend(['--use_phase2_defaults', 'true' if value else 'false'])
        elif key in ['alpha', 'rare_weight', 'multi_label_boost']:
            cmd.extend([f'--{key}', str(value)])
        elif key in ['k_neighbors', 'ml_smote_k']:
            cmd.extend([f'--{key}', str(value)])
        elif key in ['reweight_mode', 'ml_smote_threshold', 'mixup_layer']:
            cmd.extend([f'--{key}', str(value)])
    
    if dry_run:
        cmd.append('--dry_run')
    
    return cmd


def build_eval_command(model_path: str, output_path: str) -> List[str]:
    """Build command for CXR8 evaluation."""
    return [
        'python3', 'src/multilabel/evaluate_multilabel.py',
        '--model_path', model_path,
        '--output', output_path,
    ]


def run_single_experiment(
    approach: str,
    seed: int,
    params: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run training + evaluation for a single experiment."""
    
    experiment_id = get_experiment_id(approach, params, seed)
    param_dir = get_param_dirname(params)
    
    # Create directories
    model_dir = os.path.join(MODELS_DIR, f'approach_{approach}', param_dir, f'seed_{seed}')
    eval_dir = os.path.join(EVAL_DIR, f'approach_{approach}', param_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    result = {
        'experiment_id': experiment_id,
        'approach': approach,
        'seed': seed,
        'params': params,
        'param_dir': param_dir,
        'timestamp': datetime.now().isoformat(),
    }
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # Step 1: Training
    # -------------------------------------------------------------------------
    print("\n📚 STEP 1: Training")
    print("-" * 40)
    
    train_cmd = build_train_command(approach, seed, params, model_dir, dry_run)
    print(f"Command: {' '.join(train_cmd)}")
    
    train_start = time.time()
    try:
        train_result = subprocess.run(
            train_cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        train_time = time.time() - train_start
        
        # Parse training output
        val_auc = None
        model_path = None
        for line in train_result.stdout.split('\n'):
            if 'Final Best AUC:' in line:
                try:
                    val_auc = float(line.split(':')[-1].strip())
                except:
                    pass
            if 'Model saved to:' in line:
                model_path = line.split(':', 1)[-1].strip()
        
        result['train_status'] = 'success' if train_result.returncode == 0 else 'failed'
        result['train_time_sec'] = train_time
        result['val_auc'] = val_auc
        result['model_path'] = model_path
        
        if train_result.returncode != 0:
            result['train_stderr'] = train_result.stderr[-500:]
            print(f"✗ Training failed: {train_result.stderr[-200:]}")
        else:
            print(f"✓ Training completed in {train_time:.1f}s | Val AUC: {val_auc}")
        
    except Exception as e:
        result['train_status'] = 'error'
        result['train_error'] = str(e)
        print(f"✗ Training error: {e}")
        return result
    
    # -------------------------------------------------------------------------
    # Step 2: CXR8 Evaluation
    # -------------------------------------------------------------------------
    if model_path and os.path.exists(model_path) and result['train_status'] == 'success':
        print("\n🔍 STEP 2: CXR8 Evaluation")
        print("-" * 40)
        
        eval_output = os.path.join(eval_dir, f'seed_{seed}_cxr8_eval.json')
        eval_cmd = build_eval_command(model_path, eval_output)
        print(f"Command: {' '.join(eval_cmd)}")
        
        eval_start = time.time()
        try:
            eval_result = subprocess.run(
                eval_cmd,
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            eval_time = time.time() - eval_start
            
            result['eval_status'] = 'success' if eval_result.returncode == 0 else 'failed'
            result['eval_time_sec'] = eval_time
            result['eval_output_path'] = eval_output
            
            # Parse CXR8 metrics from output
            for line in eval_result.stdout.split('\n'):
                if 'Macro-AUC:' in line and 'Val' not in line:
                    try:
                        result['cxr8_macro_auc'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                if 'Micro-AUC:' in line and 'Val' not in line:
                    try:
                        result['cxr8_micro_auc'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                if 'Macro-AUPRC:' in line:
                    try:
                        result['cxr8_macro_auprc'] = float(line.split(':')[-1].strip())
                    except:
                        pass
                if 'Micro-AUPRC:' in line:
                    try:
                        result['cxr8_micro_auprc'] = float(line.split(':')[-1].strip())
                    except:
                        pass
            
            if eval_result.returncode != 0:
                result['eval_stderr'] = eval_result.stderr[-500:]
                print(f"✗ Evaluation failed")
            else:
                print(f"✓ CXR8 Evaluation completed in {eval_time:.1f}s")
                print(f"  Macro-AUC:   {result.get('cxr8_macro_auc', 'N/A')}")
                print(f"  Micro-AUC:   {result.get('cxr8_micro_auc', 'N/A')}")
                print(f"  Macro-AUPRC: {result.get('cxr8_macro_auprc', 'N/A')}")
                print(f"  Micro-AUPRC: {result.get('cxr8_micro_auprc', 'N/A')}")
                
        except Exception as e:
            result['eval_status'] = 'error'
            result['eval_error'] = str(e)
            print(f"✗ Evaluation error: {e}")
    else:
        result['eval_status'] = 'skipped'
        print("\n⏭ Skipping evaluation (no model)")
    
    # Save experiment log
    log_path = os.path.join(LOG_DIR, f'{experiment_id}.json')
    with open(log_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def run_all_experiments(
    approaches: List[str] = None,
    dry_run: bool = False,
    resume: bool = False
) -> List[Dict[str, Any]]:
    """Run all experiments."""
    
    if approaches is None:
        approaches = ['0', 'A', 'B', 'C', 'D']
    
    # Calculate total
    total = sum(len(PARAM_GRIDS[a]) * len(SEEDS) for a in approaches)
    
    # Check for completed experiments
    completed_ids = set()
    if resume and os.path.exists(LOG_DIR):
        for f in os.listdir(LOG_DIR):
            if f.endswith('.json') and f != 'experiment_summary.json':
                try:
                    with open(os.path.join(LOG_DIR, f)) as fp:
                        data = json.load(fp)
                        if data.get('train_status') == 'success':
                            completed_ids.add(data.get('experiment_id'))
                except:
                    pass
    
    print(f"\n{'#'*70}")
    print(f"MULTI-LABEL EXPERIMENT RUNNER (with CXR8 Evaluation)")
    print(f"{'#'*70}")
    print(f"Approaches: {approaches}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {total}")
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already completed")
    print(f"Output: {RESULTS_DIR}")
    print(f"{'#'*70}\n")
    
    results = []
    completed = 0
    start_all = time.time()
    
    for approach in approaches:
        for params in PARAM_GRIDS[approach]:
            for seed in SEEDS:
                experiment_id = get_experiment_id(approach, params, seed)
                completed += 1
                
                if experiment_id in completed_ids:
                    print(f"[{completed}/{total}] ⏭ Skipping: {experiment_id}")
                    continue
                
                print(f"\n[{completed}/{total}]", end='')
                result = run_single_experiment(approach, seed, params, dry_run)
                results.append(result)
    
    total_time = time.time() - start_all
    
    # Generate summary CSV
    summary_csv_path = os.path.join(RESULTS_DIR, 'experiment_results_summary.csv')
    all_results = list(results)
    
    # Load previous results if resuming
    if resume and os.path.exists(LOG_DIR):
        for f in os.listdir(LOG_DIR):
            if f.endswith('.json') and f != 'experiment_summary.json':
                try:
                    with open(os.path.join(LOG_DIR, f)) as fp:
                        data = json.load(fp)
                        if data.get('experiment_id') not in [r.get('experiment_id') for r in all_results]:
                            all_results.append(data)
                except:
                    pass
    
    # Write summary CSV with per-disease metrics
    if all_results:
        # Define pathologies for per-disease metrics
        pathologies = ['cardiomegaly', 'edema', 'consolidation', 'atelectasis', 'pleural_effusion']
        
        # Base fieldnames
        fieldnames = [
            'approach', 'params', 'seed', 'val_auc',
            # Overall metrics
            'cxr8_macro_auc', 'cxr8_micro_auc', 
            'cxr8_macro_auprc', 'cxr8_micro_auprc',
            'cxr8_macro_f1', 'cxr8_micro_f1',
        ]
        
        # Add per-disease metrics (matching Phase 2: AUC, AUPRC, Precision, Recall, Balanced Accuracy, G-Mean)
        for p in pathologies:
            fieldnames.extend([
                f'{p}_auc', f'{p}_auprc', 
                f'{p}_precision', f'{p}_recall',
                f'{p}_balanced_accuracy', f'{p}_g_mean'
            ])
        
        fieldnames.extend(['train_status', 'eval_status', 'train_time_sec', 'eval_time_sec'])
        
        # Load detailed metrics from evaluation JSONs
        for r in all_results:
            eval_path = r.get('eval_output_path')
            if eval_path and os.path.exists(eval_path):
                try:
                    with open(eval_path) as f:
                        eval_data = json.load(f)
                    
                    metrics = eval_data.get('metrics', {})
                    
                    # Add F1 metrics
                    r['cxr8_macro_f1'] = metrics.get('macro_f1')
                    r['cxr8_micro_f1'] = metrics.get('micro_f1')
                    
                    # Add per-disease metrics
                    per_disease = metrics.get('per_disease', {})
                    pathology_map = {
                        'cardiomegaly': 'Cardiomegaly',
                        'edema': 'Edema', 
                        'consolidation': 'Consolidation',
                        'atelectasis': 'Atelectasis',
                        'pleural_effusion': 'Pleural Effusion'
                    }
                    
                    for p_key, p_name in pathology_map.items():
                        if p_name in per_disease:
                            d = per_disease[p_name]
                            r[f'{p_key}_auc'] = d.get('auc')
                            r[f'{p_key}_auprc'] = d.get('auprc')
                            r[f'{p_key}_precision'] = d.get('precision')
                            r[f'{p_key}_recall'] = d.get('recall')
                            r[f'{p_key}_balanced_accuracy'] = d.get('balanced_accuracy')
                            r[f'{p_key}_g_mean'] = d.get('g_mean')
                except Exception as e:
                    print(f"Warning: Could not load eval data from {eval_path}: {e}")
        
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in sorted(all_results, key=lambda x: (x.get('approach', ''), 
                                                         get_param_dirname(x.get('params', {})), 
                                                         x.get('seed', 0))):
                row = {
                    'params': get_param_dirname(r.get('params', {})),
                    **r
                }
                writer.writerow(row)
    
    # Save summary
    summary = {
        'total_experiments': total,
        'completed_this_run': len(results),
        'successful': sum(1 for r in results if r.get('train_status') == 'success'),
        'failed': sum(1 for r in results if r.get('train_status') != 'success'),
        'total_time_sec': total_time,
        'summary_csv': summary_csv_path,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(LOG_DIR, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'#'*70}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'#'*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Successful: {summary['successful']}/{summary['completed_this_run']}")
    print(f"Results CSV: {summary_csv_path}")
    print(f"{'#'*70}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run multi-label experiments with CXR8 evaluation')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--approach', type=str, choices=['0', 'A', 'B', 'C', 'D'])
    parser.add_argument('--dry_run', action='store_true', help='Dry-run mode')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--list', action='store_true', help='List experiments only')
    
    args = parser.parse_args()
    
    if args.list:
        total = 0
        for approach, params_list in PARAM_GRIDS.items():
            print(f"\nApproach {approach}: {len(params_list)} × 3 seeds = {len(params_list)*3}")
            for i, params in enumerate(params_list):
                print(f"  [{i+1}] {get_param_dirname(params)}")
            total += len(params_list) * 3
        print(f"\nTotal: {total} experiments")
        return
    
    if args.all:
        run_all_experiments(dry_run=args.dry_run, resume=args.resume)
    elif args.approach:
        run_all_experiments(approaches=[args.approach], dry_run=args.dry_run, resume=args.resume)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

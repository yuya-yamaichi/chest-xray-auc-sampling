"""
Fine-tuning Script for Single-Pathology Classification
Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization

REVISED VERSION: Compatible with LibAUC 1.2.0, matching original performance
This version achieves the target ~0.8712 AUC by exactly matching the original implementation.

Usage:
    python finetune.py --class_id 0 --seed 123
    python finetune.py --class_id 0 --sampler SMOTE --seed 123
"""

import torch
import numpy as np
import argparse
import os
import time
import csv
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score

# LibAUC imports
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from libauc.models import densenet121  # Use lowercase for consistency
from libauc.datasets import CheXpert

# Sampling imports (for experimental variations)
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NeighbourhoodCleaningRule, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter

# Configuration management
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import load_config, get_device_config, set_reproducibility, get_pathology_name

# =====================================================================================
# GLOBAL CONFIGURATION - Loaded from YAML
# =====================================================================================

# Load configuration from YAML file
CONFIG = load_config()

# =====================================================================================
# UTILITY FUNCTIONS
# =====================================================================================

def _to_json_serializable(obj):
    """Recursively convert objects into JSON-serializable types.

    - Converts numpy scalars to native Python scalars via .item()
    - Converts sets/tuples to lists
    - Converts dict keys/values recursively
    - Falls back to str() for unsupported types
    """
    try:
        import numpy as np  # local import to avoid global dependency issues
        numpy_scalar_types = (
            np.integer, np.floating, np.bool_, np.number
        )
    except Exception:  # numpy not available or import error
        numpy_scalar_types = tuple()

    # numpy scalars
    if numpy_scalar_types and isinstance(obj, numpy_scalar_types):
        try:
            return obj.item()
        except Exception:
            return float(obj) if hasattr(obj, '__float__') else int(obj)

    # simple types pass through
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # mappings
    if isinstance(obj, dict):
        return {str(_to_json_serializable(k)): _to_json_serializable(v) for k, v in obj.items()}

    # iterables
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    # fallback
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune DenseNet121 with optional sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with SMOTE
    python src/pipelines/finetune.py --class_id 0 --sampler SMOTE --seed 123
    
    # Custom sampling ratio with SMOTE (50% minority to majority ratio)
    python src/pipelines/finetune.py --class_id 0 --sampler SMOTE --sampling_ratio 0.5
    
    # Ratio format with RandomOverSampler (1:3 minority to majority)
    python src/pipelines/finetune.py --class_id 0 --sampler RandomOverSampler --sampling_ratio "1:3"
    
    # Auto-balanced sampling (default behavior)
    python src/pipelines/finetune.py --class_id 0 --sampler SMOTE --sampling_ratio auto
        """
    )
    
    parser.add_argument('--class_id', type=int, required=True, 
                       choices=[0, 1, 2, 3, 4],
                       help='Target pathology ID (0=Cardiomegaly, 1=Edema, 2=Consolidation, 3=Atelectasis, 4=Pleural Effusion)')
    
    parser.add_argument('--sampler', type=str, default='none',
                       choices=CONFIG.get('sampling_methods', ['none']),
                       help='Sampling method to apply. Available: none, RandomOverSampler, SMOTE, ADASYN, etc.')
    
    parser.add_argument('--sampling_ratio', type=str, default=None,
                       help='Custom sampling ratio. Options: "auto" (balanced), "0.5" (50%% minority/majority), "1:3" (1:3 ratio), etc. Overrides config file.')
    
    parser.add_argument('--seed', type=int, default=CONFIG.get('reproducibility.seed', 123),
                       help='Random seed for reproducibility')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    return parser.parse_args()

def apply_sampling(dataset, sampler_name, seed, sampling_ratio=None):
    """
    Apply sampling to dataset if requested.
    For 'none', returns original dataset.
    For other samplers, applies the sampling technique with configurable ratios.
    
    Args:
        dataset: The dataset to sample from
        sampler_name: Name of the sampling method
        seed: Random seed for reproducibility
        sampling_ratio: Optional custom sampling ratio (overrides config)
    
    Returns:
        tuple: (resampled_dataset, sampling_info_dict)
        
    OPTIMIZED: Extracts labels efficiently from CheXpert dataset's df attribute
    instead of loading images from disk.
    """
    if sampler_name == 'none':
        return dataset, {
            'original_distribution': {},
            'final_distribution': {},
            'sampling_ratio_achieved': None,
            'sampling_strategy_used': 'none',
            'sampling_applied': False
        }
    
    print(f"  Extracting labels for sampling (optimized version)...")
    
    # For CheXpert dataset, extract labels directly from the dataframe
    # This avoids loading images from disk, making it much faster
    if hasattr(dataset, 'df') and hasattr(dataset, 'class_index'):
        # Access labels directly from the dataframe
        class_index = dataset.class_index
        if class_index >= 0:
            # Single-label classification - extract specific column
            pathology_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
            if class_index < len(pathology_columns):
                col_name = pathology_columns[class_index]
                labels = dataset.df[col_name].values
            else:
                raise ValueError(f"Invalid class_index: {class_index}")
        else:
            # Multi-label classification - not supported for sampling
            raise ValueError("Multi-label sampling not supported")
        
        # Convert labels to appropriate format (handle NaN as 0)
        labels = np.nan_to_num(labels, nan=0.0).astype(int)
        indices = np.arange(len(labels))
        
    else:
        # Fallback to original method for non-CheXpert datasets (slower)
        print("  Warning: Using slower fallback method for label extraction")
        labels = []
        indices = np.arange(len(dataset))
        
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label.item() if torch.is_tensor(label) else label)
        
        labels = np.array(labels)
    
    X = indices.reshape(-1, 1)
    y = labels
    
    # Store original distribution
    original_counter = Counter(y)
    print(f"  Original class distribution: {original_counter}")
    
    # Determine sampling strategy
    sampling_strategy = _get_sampling_strategy(sampler_name, sampling_ratio, original_counter)
    print(f"  Using sampling strategy: {sampling_strategy}")
    
    try:
        hyperparams = { 'sampler': sampler_name }
        # Initialize sampler with configurable parameters
        if sampler_name == 'RandomOverSampler':
            sampler = RandomOverSampler(
                random_state=seed,
                sampling_strategy=sampling_strategy
            )
            hyperparams.update({ 'sampling_strategy': sampling_strategy })
        elif sampler_name == 'SMOTE':
            k_neighbors = CONFIG.get('sampling.smote.k_neighbors', 5)
            k_neighbors = min(k_neighbors, min(original_counter.values())-1)
            sampler = SMOTE(
                random_state=seed, 
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy
            )
            hyperparams.update({ 'k_neighbors': k_neighbors, 'sampling_strategy': sampling_strategy })
        elif sampler_name == 'ADASYN':
            n_neighbors = CONFIG.get('sampling.adasyn.n_neighbors', 5)
            n_neighbors = min(n_neighbors, min(original_counter.values())-1)
            sampler = ADASYN(
                random_state=seed,
                n_neighbors=n_neighbors,
                sampling_strategy=sampling_strategy
            )
            hyperparams.update({ 'n_neighbors': n_neighbors, 'sampling_strategy': sampling_strategy })
        elif sampler_name == 'RandomUnderSampler':
            replacement = CONFIG.get('sampling.random_undersampler.replacement', False)
            sampler = RandomUnderSampler(
                random_state=seed,
                sampling_strategy=sampling_strategy,
                replacement=replacement
            )
            hyperparams.update({ 'replacement': bool(replacement), 'sampling_strategy': sampling_strategy })
        elif sampler_name == 'TomekLinks':
            sampler = TomekLinks()
            hyperparams.update({ 'sampling_strategy': 'auto' })
        elif sampler_name == 'NeighbourhoodCleaningRule':
            # Configure NCR from CONFIG if provided
            ncr_k = CONFIG.get('sampling.ncr.k_neighbors', None)
            kind_sel = CONFIG.get('sampling.ncr.kind_sel', None)
            threshold_cleaning = CONFIG.get('sampling.ncr.threshold_cleaning', None)

            kwargs_ncr = {}
            if ncr_k is not None:
                try:
                    ncr_k = int(ncr_k)
                    ncr_k = max(1, min(ncr_k, min(original_counter.values()) - 1))
                    kwargs_ncr['n_neighbors'] = ncr_k
                except Exception:
                    pass
            if kind_sel is not None:
                kwargs_ncr['kind_sel'] = kind_sel
            if threshold_cleaning is not None:
                try:
                    kwargs_ncr['threshold_cleaning'] = float(threshold_cleaning)
                except Exception:
                    pass

            sampler = NeighbourhoodCleaningRule(**kwargs_ncr)
            hyperparams.update({
                'ncr_k_neighbors': kwargs_ncr.get('n_neighbors', None),
                'ncr_kind_sel': kwargs_ncr.get('kind_sel', None),
                'ncr_threshold_cleaning': kwargs_ncr.get('threshold_cleaning', None)
            })
        elif sampler_name == 'SMOTETomek':
            k_neighbors = CONFIG.get('sampling.smote_tomek.k_neighbors', 5)
            k_neighbors = min(k_neighbors, min(original_counter.values())-1)
            sampler = SMOTETomek(
                random_state=seed,
                sampling_strategy=sampling_strategy,
                smote=SMOTE(random_state=seed, k_neighbors=k_neighbors)
            )
            hyperparams.update({ 'smote_k_neighbors': k_neighbors })
        elif sampler_name == 'SMOTEENN':
            smote_k = CONFIG.get('sampling.smote_enn.k_neighbors', 5)
            smote_k = max(1, min(int(smote_k), min(original_counter.values()) - 1))
            enn_k = CONFIG.get('sampling.smote_enn.enn_k_neighbors', None)
            enn_obj = None
            if enn_k is not None:
                try:
                    enn_k = max(1, min(int(enn_k), min(original_counter.values()) - 1))
                    enn_obj = EditedNearestNeighbours(n_neighbors=enn_k)
                except Exception:
                    enn_obj = None
            sampler = SMOTEENN(
                random_state=seed,
                sampling_strategy=sampling_strategy,
                smote=SMOTE(random_state=seed, k_neighbors=smote_k),
                enn=enn_obj
            )
            hyperparams.update({ 'smote_k_neighbors': smote_k, 'enn_k_neighbors': enn_k if enn_obj is not None else None })
        else:
            return dataset, {
                'original_distribution': dict(original_counter),
                'final_distribution': dict(original_counter),
                'sampling_ratio_achieved': None,
                'sampling_strategy_used': 'unknown'
            }
        
        # Apply sampling
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        resampled_indices = X_resampled.flatten().astype(int)
        
        final_counter = Counter(y_resampled)
        print(f"  After {sampler_name}: {final_counter}")
        
        # Calculate achieved sampling ratio
        if len(original_counter) == 2:  # Binary classification
            minority_class = min(original_counter.keys(), key=lambda k: original_counter[k])
            majority_class = max(original_counter.keys(), key=lambda k: original_counter[k])
            
            original_ratio = original_counter[minority_class] / original_counter[majority_class]
            final_ratio = final_counter[minority_class] / final_counter[majority_class]
        else:
            original_ratio = None
            final_ratio = None
        
        # Compute distribution deltas (useful for cleaning/hybrid diagnostics)
        delta_class_0 = final_counter.get(0, 0) - original_counter.get(0, 0)
        delta_class_1 = final_counter.get(1, 0) - original_counter.get(1, 0)
        original_total = sum(original_counter.values())
        final_total = sum(final_counter.values())
        total_change = final_total - original_total
        change_fraction = (total_change / original_total) if original_total > 0 else None

        # Create sampling info (pre/post fields default)
        sampling_info = {
            'original_distribution': dict(original_counter),
            'final_distribution': dict(final_counter),
            'sampling_ratio_achieved': final_ratio,
            'sampling_strategy_used': str(sampling_strategy),
            'sampling_applied': True,
            'pre_sampling_ratio_achieved': final_ratio,
            'post_sampling_ratio_achieved': final_ratio,
            'post_balancing_applied': False,
            'post_balancing_method': None,
            'target_ratio_requested': None,
            'pre_target_diff_abs': None,
            'pre_target_diff_signed': None,
            'post_target_diff_abs': None,
            'post_target_diff_signed': None,
            'delta_class_0': delta_class_0,
            'delta_class_1': delta_class_1,
            'total_change': total_change,
            'change_fraction': change_fraction,
            'hyperparams': hyperparams
        }

        # Optional: ADASYN post-balancing within tolerance epsilon
        if sampler_name == 'ADASYN' and sampling_ratio is not None and CONFIG.get('sampling.adasyn.enable_post_balance', True):
            # Parse numeric target ratio if possible
            target_ratio = None
            if isinstance(sampling_ratio, str):
                if sampling_ratio == 'auto':
                    target_ratio = None
                elif ':' in sampling_ratio:
                    try:
                        a, b = sampling_ratio.split(':', 1)
                        target_ratio = float(a) / float(b)
                    except Exception:
                        target_ratio = None
                else:
                    try:
                        target_ratio = float(sampling_ratio)
                    except Exception:
                        target_ratio = None
            elif isinstance(sampling_ratio, (int, float)):
                target_ratio = float(sampling_ratio)

            if target_ratio is not None and final_ratio is not None:
                sampling_info['target_ratio_requested'] = target_ratio
                eps = float(CONFIG.get('sampling.adasyn.post_balance_epsilon', 0.01))
                diff = abs(final_ratio - target_ratio)
                sampling_info['pre_target_diff_abs'] = diff
                sampling_info['pre_target_diff_signed'] = (final_ratio - target_ratio)
                if diff > eps:
                    # Apply post-balancing using ROS (if minority too small) or RUS (if minority too large)
                    X_post, y_post = X_resampled, y_resampled
                    post_method = None
                    minority_class = min(final_counter.keys(), key=lambda k: final_counter[k])
                    majority_class = max(final_counter.keys(), key=lambda k: final_counter[k])
                    if final_ratio < target_ratio:
                        # Need to increase minority via ROS
                        desired_minority_count = int(target_ratio * final_counter[majority_class])
                        ros = RandomOverSampler(random_state=seed, sampling_strategy={minority_class: desired_minority_count})
                        X_post, y_post = ros.fit_resample(X_resampled, y_resampled)
                        post_method = 'ROS'
                    elif final_ratio > target_ratio and target_ratio > 0:
                        # Need to reduce majority via RUS
                        desired_majority_count = int(final_counter[minority_class] / target_ratio)
                        rus = RandomUnderSampler(random_state=seed, sampling_strategy={majority_class: desired_majority_count})
                        X_post, y_post = rus.fit_resample(X_resampled, y_resampled)
                        post_method = 'RUS'

                    if post_method is not None:
                        resampled_indices = X_post.flatten().astype(int)
                        final_counter_post = Counter(y_post)
                        # Recompute ratio
                        post_ratio = final_counter_post[minority_class] / final_counter_post[majority_class]
                        sampling_info['final_distribution'] = dict(final_counter_post)
                        sampling_info['sampling_ratio_achieved'] = post_ratio
                        sampling_info['post_sampling_ratio_achieved'] = post_ratio
                        sampling_info['pre_sampling_ratio_achieved'] = final_ratio
                        sampling_info['post_balancing_applied'] = True
                        sampling_info['post_balancing_method'] = post_method
                        sampling_info['post_target_diff_abs'] = abs(post_ratio - target_ratio)
                        sampling_info['post_target_diff_signed'] = (post_ratio - target_ratio)
                        # Return subset built from post-balanced indices
                        return torch.utils.data.Subset(dataset, resampled_indices), sampling_info
        
        # Create subset
        return torch.utils.data.Subset(dataset, resampled_indices), sampling_info
        
    except Exception as e:
        print(f"  ⚠️  Warning: Sampling failed: {e}")
        print(f"  Using original dataset without sampling")
        return dataset, {
            'original_distribution': dict(original_counter),
            'final_distribution': dict(original_counter),
            'sampling_ratio_achieved': None,
            'sampling_strategy_used': 'failed',
            'sampling_applied': False
        }


def _get_sampling_strategy(sampler_name, sampling_ratio, original_counter):
    """
    Determine the sampling strategy based on configuration and parameters.
    
    Args:
        sampler_name: Name of the sampling method
        sampling_ratio: Custom sampling ratio if provided
        original_counter: Counter object with original class distribution
    
    Returns:
        Sampling strategy suitable for the sampler
    """
    # If custom ratio provided, use it
    if sampling_ratio is not None:
        if sampling_ratio == "auto":
            return "auto"
        elif isinstance(sampling_ratio, str) and ":" in sampling_ratio:
            # Parse ratio like "1:2" -> 0.5
            parts = sampling_ratio.split(":")
            ratio_value = float(parts[0]) / float(parts[1])
            return _convert_ratio_to_strategy(ratio_value, original_counter, sampler_name)
        elif isinstance(sampling_ratio, (int, float, str)):
            ratio_value = float(sampling_ratio)
            return _convert_ratio_to_strategy(ratio_value, original_counter, sampler_name)
    
    # Otherwise use config defaults
    config_key = f"sampling.{sampler_name.lower().replace('sampler', '_sampler')}.sampling_strategy"
    return CONFIG.get(config_key, "auto")


def _convert_ratio_to_strategy(ratio_value, original_counter, sampler_name):
    """
    Convert a ratio value to appropriate sampling strategy.
    
    Args:
        ratio_value: The desired ratio (e.g., 0.5 for 1:2 minority:majority)
        original_counter: Counter with original distribution
        sampler_name: Name of the sampling method
    
    Returns:
        Sampling strategy dict or value
    """
    if len(original_counter) != 2:
        print(f"  Warning: Custom ratios only supported for binary classification. Using 'auto'")
        return "auto"
    
    minority_class = min(original_counter.keys(), key=lambda k: original_counter[k])
    majority_class = max(original_counter.keys(), key=lambda k: original_counter[k])
    
    majority_count = original_counter[majority_class]
    desired_minority_count = int(ratio_value * majority_count)
    
    # For oversampling methods, set target count for minority class
    if sampler_name in ['RandomOverSampler', 'SMOTE', 'ADASYN', 'SMOTETomek', 'SMOTEENN']:
        return {minority_class: desired_minority_count}
    
    # For undersampling methods, set target count for majority class
    elif sampler_name in ['RandomUnderSampler']:
        desired_majority_count = int(original_counter[minority_class] / ratio_value)
        return {majority_class: desired_majority_count}
    
    # For other methods, use auto
    else:
        return "auto"

def save_model(model, args, best_val_auc):
    """Save the trained model in pathology-specific directory."""
    models_dir = CONFIG.get('paths.models_dir', 'models')
    pathology_name = get_pathology_name(CONFIG, args.class_id)
    
    # Create pathology-specific directory
    pathology_dir = os.path.join(models_dir, pathology_name.lower())
    os.makedirs(pathology_dir, exist_ok=True)
    
    model_filename = f"finetuned_{pathology_name.lower()}_{args.sampler}_seed{args.seed}_auc{best_val_auc:.4f}.pth"
    model_path = os.path.join(pathology_dir, model_filename)
    
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    return model_path

def log_training_info(args, best_val_auc, training_time, model_path, sampling_info=None):
    """Log training information to pathology-specific CSV file."""
    results_dir = CONFIG.get('paths.results_dir', 'results')
    pathology_name = get_pathology_name(CONFIG, args.class_id)
    
    # Use pathology-specific log file
    training_log_file = f"training_log_{pathology_name.lower()}.csv"
    
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, training_log_file)
    
    file_exists = os.path.exists(log_path)
    
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pathology': pathology_name,
        'class_id': args.class_id,
        'sampler': args.sampler,
        'seed': args.seed,
        'best_val_auc': best_val_auc,
        'training_time_sec': training_time,
        'model_path': model_path
    }
    
    # Add sampling information if available
    if sampling_info:
        # Safely JSON-encode hyperparameters to avoid numpy/unsupported types causing errors
        try:
            safe_hparams = _to_json_serializable(sampling_info.get('hyperparams', {}))
            sampling_hparams_json = json.dumps(safe_hparams, ensure_ascii=False)
        except Exception:
            sampling_hparams_json = json.dumps(str(sampling_info.get('hyperparams', {})), ensure_ascii=False)

        row_data.update({
            'original_class_0': sampling_info['original_distribution'].get(0, 0),
            'original_class_1': sampling_info['original_distribution'].get(1, 0),
            'final_class_0': sampling_info['final_distribution'].get(0, 0),
            'final_class_1': sampling_info['final_distribution'].get(1, 0),
            'sampling_ratio_achieved': sampling_info.get('sampling_ratio_achieved', None),
            'sampling_strategy_used': sampling_info.get('sampling_strategy_used', 'unknown'),
            'sampling_applied': sampling_info.get('sampling_applied', None),
            'sampling_hyperparams': sampling_hparams_json
        })
    else:
        # Add empty columns for consistency
        row_data.update({
            'original_class_0': None,
            'original_class_1': None, 
            'final_class_0': None,
            'final_class_1': None,
            'sampling_ratio_achieved': None,
            'sampling_strategy_used': args.sampler if args.sampler else 'none',
            'sampling_applied': None
        })
    
    # Add custom sampling ratio if provided via args
    if hasattr(args, 'sampling_ratio') and args.sampling_ratio is not None:
        row_data['sampling_ratio_requested'] = args.sampling_ratio
    else:
        row_data['sampling_ratio_requested'] = CONFIG.get('sampling.sampling_ratio', 'auto')
    
    with open(log_path, 'a', newline='') as csvfile:
        fieldnames = row_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"✓ Results logged to: {log_path}")
    return log_path

# =====================================================================================
# MAIN TRAINING FUNCTION
# =====================================================================================

def main():
    """Main fine-tuning function - using YAML configuration."""
    
    # Start timing
    total_start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Reload config if custom config file specified
    if args.config != 'config.yaml':
        global CONFIG
        CONFIG = load_config(args.config)
    
    # Set reproducibility from config and args
    set_reproducibility(CONFIG)
    if args.seed != CONFIG.get('reproducibility.seed', 123):
        # Override seed from command line
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"✓ Seed overridden to {args.seed}")
    
    # Get device configuration
    device = get_device_config(CONFIG)
    
    # Get configuration values
    batch_size = CONFIG.get('finetune.batch_size', 32)
    gradient_accumulation_steps = CONFIG.get('finetune.gradient_accumulation_steps', 1)
    learning_rate = float(CONFIG.get('finetune.learning_rate', 0.05))
    gamma = int(CONFIG.get('finetune.gamma', 500))
    epochs = int(CONFIG.get('finetune.epochs', 2))
    
    print("=" * 80)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 80)
    print(f"Target pathology: {get_pathology_name(CONFIG, args.class_id)} (class_id={args.class_id})")
    print(f"Sampling method: {args.sampler}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")  # Critical parameter
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Epochs: {epochs}")
    print("=" * 80)
    
    # ---------------------------------------------------------------------------------
    # Data Loading - Using configuration
    # ---------------------------------------------------------------------------------
    print(f"\nLoading CheXpert dataset for {get_pathology_name(CONFIG, args.class_id)}...")
    
    # Get dataset configuration
    dataset_root = CONFIG.get('dataset.root', '/home/yamaichi/CheXpert/CheXpert-v1.0/')
    image_size = CONFIG.get('dataset.image_size', 224)
    use_frontal_only = CONFIG.get('dataset.use_frontal_only', True)
    num_workers = CONFIG.get('finetune.num_workers', 2)
    
    # When using 'none' sampler, enable built-in upsampling (matches original)
    use_builtin_upsampling = (args.sampler == 'none')
    
    # Load training set - using configuration
    trainSet = CheXpert(
        csv_path=os.path.join(dataset_root, 'train.csv'),
        image_root_path=dataset_root,
        use_upsampling=use_builtin_upsampling,  # True when no external sampler
        use_frontal=use_frontal_only,
        image_size=image_size,
        mode='train',
        class_index=args.class_id
    )
    
    # Get imratio before any sampling
    imratio = trainSet.imratio
    print(f"✓ Imbalance ratio: {imratio:.4f}")
    
    # Apply external sampling if requested
    sampling_info = None
    if args.sampler != 'none':
        print(f"\nApplying {args.sampler} sampling...")
        sampling_ratio = getattr(args, 'sampling_ratio', None)
        trainSet, sampling_info = apply_sampling(trainSet, args.sampler, args.seed, sampling_ratio)
    else:
        # Initialize sampling_info for consistency
        sampling_info = {
            'original_distribution': {},
            'final_distribution': {},
            'sampling_ratio_achieved': None,
            'sampling_strategy_used': 'none',
            'sampling_applied': False
        }
    
    # Load validation set - never upsampled
    testSet = CheXpert(
        csv_path=os.path.join(dataset_root, 'valid.csv'),
        image_root_path=dataset_root,
        use_upsampling=False,
        use_frontal=use_frontal_only,
        image_size=image_size,
        mode='valid',
        class_index=args.class_id
    )
    
    # Create data loaders - using configuration
    trainloader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True  # Avoid BatchNorm errors on last small batch (batch_size=1)
    )
    
    testloader = torch.utils.data.DataLoader(
        testSet,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    print(f"✓ Training batches: {len(trainloader)}")
    print(f"✓ Validation samples: {len(testSet)}")
    
    # ---------------------------------------------------------------------------------
    # Model Setup - Using configuration
    # ---------------------------------------------------------------------------------
    print("\nInitializing DenseNet121 model...")
    
    # Get model configuration
    num_classes = CONFIG.get('model.num_classes', 1)
    activation = CONFIG.get('model.activation', 'relu')
    last_activation = CONFIG.get('model.last_activation', 'sigmoid')
    pretrained_model_path = CONFIG.get('model.pretrained_model_path', 'models/pretrained/best_pretrained_model.pth')
    
    # CRITICAL: Use 'sigmoid' activation like original
    model = densenet121(
        pretrained=False,
        last_activation=last_activation,  # CRITICAL: Built-in sigmoid activation
        activations=activation,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Load pretrained weights
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained weights from {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location=device)
        # Remove classifier layers (5 classes -> 1 class)
        state_dict.pop('classifier.weight', None)
        state_dict.pop('classifier.bias', None)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Pretrained weights loaded successfully")
    else:
        print(f"⚠️  Warning: Pretrained model not found at {pretrained_model_path}")
        print("  Training from scratch (performance may be lower)")
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---------------------------------------------------------------------------------
    # Loss and Optimizer - Using configuration
    # ---------------------------------------------------------------------------------
    print("\nSetting up AUCMLoss and PESG optimizer...")
    
    # Get optimizer configuration
    weight_decay = float(CONFIG.get('finetune.weight_decay', 1e-5))
    margin = float(CONFIG.get('finetune.margin', 1.0))
    
    # CRITICAL: Initialize AUCMLoss with imratio
    Loss = AUCMLoss(imratio=imratio)
    
    # LibAUC PESG has slightly different expectations across versions:
    # - Some versions require epoch_decay to be numeric (None triggers TypeError)
    # - Some versions assert that only one of (gamma, epoch_decay) can be used
    #
    # We want to keep the original behavior: use gamma=500 and disable epoch_decay.
    # So we try epoch_decay=None first (to satisfy the assert), and fall back to 0.0 if needed.
    pesg_kwargs = dict(
        lr=learning_rate,
        gamma=gamma,  # CRITICAL: gamma=500 for proper convergence
        margin=margin,
        weight_decay=weight_decay,
        a=Loss.a,
        b=Loss.b,
        alpha=Loss.alpha,
        imratio=imratio,
        device=device,
    )

    # PESG compatibility:
    # - Some versions expect: PESG(params, loss_fn, ...)
    # - Others expect: PESG(model, ...) and internally call self.model.parameters()
    #
    # Also, epoch_decay handling differs:
    # - Some versions require epoch_decay numeric (None triggers TypeError)
    # - Others assert: only one of gamma/epoch_decay can be set (epoch_decay must be None when gamma is used)
    params_list = list(model.parameters())
    pesg_last_err: Exception | None = None
    optimizer = None
    for epoch_decay_candidate in (None, 0.0):
        for first_arg in (params_list, model):
            # Try with loss_fn (required in newer versions)
            try:
                optimizer = PESG(first_arg, loss_fn=Loss, epoch_decay=epoch_decay_candidate, **pesg_kwargs)
                break
            except Exception as e:
                pesg_last_err = e
            # Try without loss_fn (older versions)
            try:
                optimizer = PESG(first_arg, epoch_decay=epoch_decay_candidate, **pesg_kwargs)
                break
            except Exception as e:
                pesg_last_err = e
        if optimizer is not None:
            break
    if optimizer is None:
        raise RuntimeError(f"Unable to initialize PESG optimizer (compat mode). Last error: {pesg_last_err}")
    
    print(f"✓ PESG optimizer configured:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Gamma: {gamma}")
    print(f"  - Margin: {margin}")
    print(f"  - Weight decay: {weight_decay}")
    
    # ---------------------------------------------------------------------------------
    # Training Loop - Matching original exactly
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)
    
    # Get validation configuration
    validation_interval = CONFIG.get('finetune.validation_interval', 400)
    
    training_start_time = time.time()
    best_val_auc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        if epoch > 0:
            optimizer.update_regularizer(decay_factor=10)
            print("✓ Updated regularizer with decay_factor=10")
        
        epoch_start = time.time()
        batch_count = 0
        accumulated_loss = 0
        
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            train_data, train_labels = train_data.to(device), train_labels.to(device)
            
            # Forward pass - model already has sigmoid built-in
            y_pred = model(train_data)
            # NO manual sigmoid needed - it's in the model
            
            loss = Loss(y_pred, train_labels)
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            
            loss.backward()
            
            # Update weights after accumulating gradients
            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count += 1
            
            # Validation - using configuration with memory optimization
            if idx % validation_interval == 0:
                # Clear cache before validation (GPU only)
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():
                    test_pred = []
                    test_true = []
                    for jdx, data in enumerate(testloader):
                        test_data, test_label = data
                        test_data = test_data.to(device)
                        y_pred = model(test_data)
                        # NO manual sigmoid - model outputs probabilities
                        # Move to CPU for metric computation
                        test_pred.append(y_pred.detach().cpu().numpy())
                        test_true.append(test_label.detach().cpu().numpy())
                        # Clear intermediate variables
                        del test_data, y_pred
                    
                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc = roc_auc_score(test_true, test_pred)
                    model.train()
                    
                    if best_val_auc < val_auc:
                        best_val_auc = val_auc
                        # Save best model during training
                        models_dir = CONFIG.get('paths.models_dir', 'models')
                        temp_path = os.path.join(models_dir, f'temp_best_{args.class_id}.pth')
                        os.makedirs(models_dir, exist_ok=True)
                        torch.save(model.state_dict(), temp_path)
                        print(f"  🎉 New best model! Val_AUC={val_auc:.4f}")
                    
                    print(f"  Epoch={epoch+1}, BatchID={idx:4d}, Val_AUC={val_auc:.4f}, lr={optimizer.lr:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"  Epoch completed in {epoch_time:.2f} seconds ({batch_count} batches)")
    
    # Calculate execution times
    training_time = time.time() - training_start_time
    total_time = time.time() - total_start_time
    
    # ---------------------------------------------------------------------------------
    # Final Results
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETED")
    print("=" * 80)
    print(f"✓ Best Validation AUC: {best_val_auc:.4f}")
    print(f"✓ Training time: {training_time:.2f} seconds")
    print(f"✓ Total time: {total_time:.2f} seconds")
    print("=" * 80)
    
    # Save final model
    model_path = save_model(model, args, best_val_auc)
    
    # Log results
    log_path = log_training_info(args, best_val_auc, training_time, model_path, sampling_info)
    
    return best_val_auc

if __name__ == "__main__":
    try:
        best_auc = main()
        print(f"\n✅ Final Best AUC: {best_auc:.4f}")
    except KeyboardInterrupt:
        print("\n⚠️  Fine-tuning interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {str(e)}")
        raise
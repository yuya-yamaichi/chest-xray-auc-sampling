#!/usr/bin/env python3
"""
Multi-label Training Script for Chest X-ray Classification

Master's Thesis Research: Comparison of 5 Sampling Approaches
- Approach 0: Baseline (no sampling)
- Approach A: Disease-Adaptive Binary Relevance
- Approach B: Multi-label Manifold Mixup
- Approach C: Distribution-Balanced WeightedSampler
- Approach D: ML-SMOTE

Usage:
    python train_multilabel.py --approach 0 --seed 123
    python train_multilabel.py --approach A --seed 123
    python train_multilabel.py --approach B --seed 123 --mixup_layer features.denseblock3
    python train_multilabel.py --approach C --seed 123
    python train_multilabel.py --approach D --seed 123
"""

import os
import sys
import argparse
import time
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# LibAUC imports
from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG
from libauc.models import densenet121

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilabel.datasets import CheXpertMultiLabel
from multilabel.samplers import (
    get_sampler,
    BaselineSampler,
    ManifoldMixupSampler,
    DistributionBalancedSampler
)
from multilabel.metrics import compute_multilabel_metrics, format_metrics_report


# =====================================================================================
# CONFIGURATION LOADING
# =====================================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Try multiple paths to find config
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'config_multilabel.yaml'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src', 'multilabel', 'config_multilabel.yaml'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            config_path = possible_paths[0]  # Use first as fallback
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}
    
    print(f"✓ Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_nested(config: dict, key: str, default=None):
    """Get nested config value using dot notation."""
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


# =====================================================================================
# ARGUMENT PARSING
# =====================================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train multi-label chest X-ray classifier with sampling approaches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sampling Approaches:
    0: Baseline - No sampling applied
    A: Disease-Adaptive Binary Relevance - Per-disease SMOTE/ADASYN
    B: Multi-label Manifold Mixup - Feature space interpolation
    C: Distribution-Balanced WeightedSampler - Frequency-based sampling
    D: ML-SMOTE - Multi-label specific oversampling

Examples:
    python train_multilabel.py --approach 0 --epochs 2 --seed 123
    python train_multilabel.py --approach A --seed 42 --k_neighbors 3
    python train_multilabel.py --approach B --alpha 0.5 --rare_weight 3.0
    python train_multilabel.py --approach C --reweight_mode inverse --multi_label_boost 2.0
    python train_multilabel.py --approach D --ml_smote_k 5
        """
    )
    
    # Required arguments
    parser.add_argument('--approach', type=str, required=True,
                       choices=['0', 'A', 'B', 'C', 'D'],
                       help='Sampling approach to use')
    
    # General training arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # Approach A parameters
    parser.add_argument('--k_neighbors', type=int, default=None,
                       help='k-neighbors for SMOTE/ADASYN (Approach A)')
    parser.add_argument('--use_phase2_defaults', type=str, default='true',
                       choices=['true', 'false'],
                       help='Use Phase 2 optimal settings for Approach A (default: true)')
    
    # Approach B parameters
    parser.add_argument('--alpha', type=float, default=None,
                       help='Beta distribution alpha for Mixup (Approach B)')
    parser.add_argument('--rare_weight', type=float, default=None,
                       help='Rare label weight multiplier (Approach B)')
    parser.add_argument('--mixup_layer', type=str, default='features.denseblock3',
                       help='Layer name for Manifold Mixup (Approach B)')
    
    # Approach C parameters
    parser.add_argument('--reweight_mode', type=str, default=None,
                       choices=['inverse', 'sqrt_inv', 'effective_num'],
                       help='Reweight mode for WeightedSampler (Approach C)')
    parser.add_argument('--multi_label_boost', type=float, default=None,
                       help='Multi-label sample boost factor (Approach C)')
    
    # Approach D parameters
    parser.add_argument('--ml_smote_k', type=int, default=None,
                       help='k-neighbors for ML-SMOTE (Approach D)')
    parser.add_argument('--ml_smote_threshold', type=str, default=None,
                       help='Threshold method for ML-SMOTE (Approach D)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for models and logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment run')
    
    # Debug/dry run
    parser.add_argument('--dry_run', action='store_true',
                       help='Run one batch only for testing')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save model checkpoint')
    
    return parser.parse_args()


# =====================================================================================
# REPRODUCIBILITY
# =====================================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================================
# TRAINING FUNCTIONS
# =====================================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    sampler=None,
    dry_run: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Handles:
    - Approach B: Manifold Mixup during training
    - Robust batch handling for empty labels
    """
    model.train()
    
    running_loss = 0.0
    n_batches = 0
    skipped_batches = 0
    
    use_mixup = (sampler is not None and 
                 isinstance(sampler, ManifoldMixupSampler) and
                 sampler.supports_online_augmentation())
    
    # Get mixup_layer from config with fallback to default
    mixup_layer = config.get('sampling', {}).get('approach_b', {}).get('mixup_layer', 'features.denseblock3')
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Robust handling: skip batches with all missing labels
        valid_mask = (labels >= 0).float()
        if valid_mask.sum() == 0:
            skipped_batches += 1
            continue
        
        # Handle uncertain labels
        labels_clean = torch.clamp(labels, min=0)
        
        optimizer.zero_grad()
        
        if use_mixup and mixup_layer is not None:
            # Approach B: Manifold Mixup
            outputs, mixed_labels = sampler.mixup_at_layer(
                model, images, labels_clean, layer_name=mixup_layer
            )
            outputs = torch.sigmoid(outputs)
            loss = loss_fn(outputs, mixed_labels)
        else:
            # Standard forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = loss_fn(outputs, labels_clean)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        n_batches += 1
        
        if dry_run and batch_idx >= 2:
            break
    
    avg_loss = running_loss / (n_batches + 1e-10)
    
    return {
        'loss': avg_loss,
        'n_batches': n_batches,
        'skipped_batches': skipped_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Validate model and compute metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())
        
        if dry_run and batch_idx >= 2:
            break
    
    y_pred_proba = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    # Handle uncertain labels (set to 0 for metrics)
    y_true_clean = np.clip(y_true, 0, 1)
    
    metrics = compute_multilabel_metrics(y_true_clean, y_pred_proba)
    
    return metrics


# =====================================================================================
# MAIN TRAINING FUNCTION
# =====================================================================================

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed
    set_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get training parameters
    epochs = args.epochs or get_nested(config, 'training.epochs', 2)
    batch_size = args.batch_size or get_nested(config, 'training.batch_size', 32)
    learning_rate = args.lr or get_nested(config, 'training.learning_rate', 0.1)
    epoch_decay = get_nested(config, 'training.epoch_decay', 2e-3)  # From notebook
    margin = get_nested(config, 'training.margin', 1.0)
    weight_decay = get_nested(config, 'training.weight_decay', 1e-5)
    num_workers = get_nested(config, 'training.num_workers', 4)
    validation_interval = get_nested(config, 'training.validation_interval', 400)
    
    print("=" * 80)
    print("MULTI-LABEL TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Approach: {args.approach}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epoch decay: {epoch_decay}")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Dataset Loading
    # -------------------------------------------------------------------------
    print("\n📁 Loading datasets...")
    
    dataset_root = get_nested(config, 'dataset.chexpert_root', '/home/yamaichi/CheXpert/CheXpert-v1.0/')
    image_size = get_nested(config, 'dataset.image_size', 224)
    use_frontal = get_nested(config, 'dataset.use_frontal_only', True)
    uncertainty_policy = get_nested(config, 'dataset.uncertainty_policy', 'zeros')
    
    train_dataset = CheXpertMultiLabel(
        csv_path=os.path.join(dataset_root, 'train.csv'),
        image_root_path=dataset_root,
        image_size=image_size,
        mode='train',
        use_frontal=use_frontal,
        uncertainty_policy=uncertainty_policy
    )
    
    val_dataset = CheXpertMultiLabel(
        csv_path=os.path.join(dataset_root, 'valid.csv'),
        image_root_path=dataset_root,
        image_size=image_size,
        mode='valid',
        use_frontal=use_frontal,
        uncertainty_policy=uncertainty_policy
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Imbalance ratios: {train_dataset.imratio_list}")
    
    # -------------------------------------------------------------------------
    # Apply Sampling
    # -------------------------------------------------------------------------
    print(f"\n🔄 Applying sampling approach {args.approach}...")
    
    # Build sampler configuration from args
    sampler_config = {}
    
    if args.approach == 'A':
        # Start with defaults, then override from config, then from CLI
        sampler_config = {
            'k_neighbors': 5,
            'use_phase2_defaults': True,
        }
        sampler_config.update(get_nested(config, 'sampling.approach_a', {}))
        if args.k_neighbors is not None:
            sampler_config['k_neighbors'] = args.k_neighbors
        # Handle use_phase2_defaults from CLI
        sampler_config['use_phase2_defaults'] = (args.use_phase2_defaults == 'true')
        sampler_config['seed'] = seed
        print(f"  Approach A config: k_neighbors={sampler_config['k_neighbors']}, use_phase2_defaults={sampler_config['use_phase2_defaults']}")
    
    elif args.approach == 'B':
        # Start with defaults, then override from config, then from CLI
        sampler_config = {
            'alpha': 1.0,
            'rare_weight': 2.0,
            'min_lambda': 0.2,
            'mix_probability': 0.5,
        }
        sampler_config.update(get_nested(config, 'sampling.approach_b', {}))
        if args.alpha is not None:
            sampler_config['alpha'] = args.alpha
        if args.rare_weight is not None:
            sampler_config['rare_weight'] = args.rare_weight
        sampler_config['seed'] = seed
        print(f"  Approach B config: alpha={sampler_config['alpha']}, rare_weight={sampler_config['rare_weight']}")
    
    elif args.approach == 'C':
        # Start with defaults, then override from config, then from CLI
        sampler_config = {
            'reweight_mode': 'sqrt_inv',
            'multi_label_boost': 1.5,
            'replacement': True,
        }
        sampler_config.update(get_nested(config, 'sampling.approach_c', {}))
        if args.reweight_mode is not None:
            sampler_config['reweight_mode'] = args.reweight_mode
        if args.multi_label_boost is not None:
            sampler_config['multi_label_boost'] = args.multi_label_boost
        sampler_config['seed'] = seed
        print(f"  Approach C config: reweight_mode={sampler_config['reweight_mode']}, multi_label_boost={sampler_config['multi_label_boost']}")
    
    elif args.approach == 'D':
        # Start with defaults, then override from config, then from CLI
        sampler_config = {
            'k_neighbors': 5,
            'threshold': 'meanIR',
        }
        sampler_config.update(get_nested(config, 'sampling.approach_d', {}))
        if args.ml_smote_k is not None:
            sampler_config['k_neighbors'] = args.ml_smote_k
        if args.ml_smote_threshold is not None:
            sampler_config['threshold'] = args.ml_smote_threshold
        sampler_config['seed'] = seed
        print(f"  Approach D config: k_neighbors={sampler_config['k_neighbors']}, threshold={sampler_config['threshold']}")
    
    print(f"  Full sampler_config: {sampler_config}")
    
    # Get sampler
    sampler = get_sampler(args.approach, sampler_config)
    
    # Apply sampling
    train_dataset = sampler.fit_resample(train_dataset)
    sampling_info = sampler.get_info()
    
    print(f"✓ Sampling method: {sampling_info.get('method', 'unknown')}")
    print(f"✓ Final training samples: {len(train_dataset)}")
    
    # -------------------------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------------------------
    print("\n📦 Creating data loaders...")
    
    # For Approach C, use WeightedRandomSampler
    if args.approach == 'C' and isinstance(sampler, DistributionBalancedSampler):
        weighted_sampler = sampler.get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=weighted_sampler,
            num_workers=num_workers,
            drop_last=True
        )
        print("✓ Using WeightedRandomSampler for training")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # -------------------------------------------------------------------------
    # Model Setup
    # -------------------------------------------------------------------------
    print("\n🧠 Initializing model...")
    
    model = densenet121(
        pretrained=True,
        last_activation=None,
        activations='relu',
        num_classes=5
    )
    model = model.to(device)
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # -------------------------------------------------------------------------
    # Loss and Optimizer
    # -------------------------------------------------------------------------
    print("\n⚙️ Setting up loss and optimizer...")
    
    loss_fn = AUCM_MultiLabel(num_classes=5)
    
    # PESG optimizer - use epoch_decay as in the reference notebook
    # Reference: 07_Optimizing_Multi_Label_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb
    optimizer = PESG(
        model,
        loss_fn=loss_fn,
        lr=learning_rate,
        epoch_decay=epoch_decay,
        margin=margin,
        weight_decay=weight_decay
    )
    
    print("✓ Using AUCM_MultiLabel loss")
    print("✓ Using PESG optimizer")
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Update learning rate / regularizer for PESG
        if epoch > 0 and hasattr(optimizer, 'update_regularizer'):
            optimizer.update_regularizer(decay_factor=10)
        
        # Train one epoch
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            sampler=sampler if args.approach == 'B' else None,
            dry_run=args.dry_run
        )
        
        print(f"  Train Loss: {train_stats['loss']:.4f}")
        print(f"  Batches: {train_stats['n_batches']} (skipped: {train_stats['skipped_batches']})")
        
        # Validate
        val_metrics = validate(model, val_loader, device, dry_run=args.dry_run)
        
        val_auc = val_metrics['macro_auc']
        print(f"  Val Macro-AUC:   {val_auc:.4f}")
        print(f"  Val Micro-AUC:   {val_metrics['micro_auc']:.4f}" if val_metrics['micro_auc'] else "  Val Micro-AUC:   N/A")
        print(f"  Val Macro-AUPRC: {val_metrics['multilabel_auprc']:.4f}" if val_metrics['multilabel_auprc'] else "  Val Macro-AUPRC: N/A")
        print(f"  Val Micro-AUPRC: {val_metrics['micro_auprc']:.4f}" if val_metrics['micro_auprc'] else "  Val Micro-AUPRC: N/A")
        
        # Save best model
        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            print(f"  🎉 New best model: AUC={best_val_auc:.4f}")
        
        if args.dry_run:
            print("  [DRY RUN: stopping after 1 epoch]")
            break
    
    training_time = time.time() - start_time
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"✓ Best Validation AUC: {best_val_auc:.4f}")
    print(f"✓ Training time: {training_time:.2f} seconds")
    
    if not args.no_save and best_model_state is not None:
        # Create output directory
        output_dir = args.output_dir or get_nested(config, 'paths.models_dir', 'models/multilabel')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate model filename
        experiment_name = args.experiment_name or f"approach_{args.approach}"
        model_filename = f"multilabel_{experiment_name}_seed{seed}_auc{best_val_auc:.4f}.pth"
        model_path = os.path.join(output_dir, model_filename)
        
        # Save model
        torch.save(best_model_state, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save training log
        logs_dir = get_nested(config, 'paths.logs_dir', 'results/multilabel/logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Helper function to convert numpy types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'approach': args.approach,
            'seed': seed,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_auc': float(best_val_auc) if best_val_auc else None,
            'training_time_sec': training_time,
            'model_path': model_path,
            'sampling_info': convert_to_native(sampling_info),
        }
        
        log_path = os.path.join(logs_dir, 'training_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        print(f"✓ Log saved to: {log_path}")
    
    print("=" * 80)
    
    return best_val_auc


if __name__ == '__main__':
    try:
        best_auc = main()
        print(f"\n✅ Final Best AUC: {best_auc:.4f}")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

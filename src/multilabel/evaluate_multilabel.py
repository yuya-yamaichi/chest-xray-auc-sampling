#!/usr/bin/env python3
"""
Multi-label Evaluation Script for CXR8 Test Dataset

Master's Thesis Research: Evaluation of trained multi-label models on unseen CXR8 dataset.

Usage:
    python evaluate_multilabel.py --model_path path/to/model.pth
    python evaluate_multilabel.py --model_path path/to/model.pth --output results.json
"""

import os
import sys
import argparse
import json
import csv
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# LibAUC imports
from libauc.models import densenet121

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multilabel.datasets import CXR8MultiLabel
from multilabel.metrics import (
    compute_multilabel_metrics,
    format_metrics_report,
    find_optimal_thresholds
)


# =====================================================================================
# CONFIGURATION
# =====================================================================================

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config_multilabel.yaml')
    
    if not os.path.exists(config_path):
        return {}
    
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
        description='Evaluate multi-label model on CXR8 test dataset'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    # Dataset options
    parser.add_argument('--cxr8_csv', type=str, default=None,
                       help='Path to CXR8 test CSV')
    parser.add_argument('--cxr8_images', type=str, default=None,
                       help='Path to CXR8 images directory')
    
    # Evaluation options
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--use_optimal_threshold', action='store_true',
                       help='Find and use optimal thresholds per disease')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--print_report', action='store_true', default=True,
                       help='Print formatted report')
    
    return parser.parse_args()


# =====================================================================================
# MODEL LOADING
# =====================================================================================

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    model = densenet121(
        pretrained=False,
        last_activation=None,
        activations='relu',
        num_classes=5
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


# =====================================================================================
# EVALUATION
# =====================================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Run inference on test dataset.
    
    Returns:
        tuple: (y_true, y_pred_proba)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1} batches...")
    
    y_pred_proba = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    return y_true, y_pred_proba


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    """Main evaluation function."""
    args = parse_arguments()
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------------------
    model = load_model(args.model_path, device)
    
    # -------------------------------------------------------------------------
    # Load Test Dataset
    # -------------------------------------------------------------------------
    print("\n📁 Loading CXR8 test dataset...")
    
    cxr8_csv = args.cxr8_csv or get_nested(config, 'dataset.cxr8_test_csv', 
                                           '/home/yamaichi/master_theis/cxr8_test.csv')
    cxr8_images = args.cxr8_images or get_nested(config, 'dataset.cxr8_root',
                                                  '/home/yamaichi/CXR8/dataset/test_dataset/')
    
    test_dataset = CXR8MultiLabel(
        csv_path=cxr8_csv,
        image_root_path=cxr8_images,
        image_size=get_nested(config, 'dataset.image_size', 224)
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Label stats: {test_dataset.get_label_stats()}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=get_nested(config, 'evaluation.num_workers', 4)
    )
    
    # -------------------------------------------------------------------------
    # Run Evaluation
    # -------------------------------------------------------------------------
    print("\n🔍 Running evaluation...")
    
    y_true, y_pred_proba = evaluate_model(model, test_loader, device)
    
    print(f"✓ Predictions shape: {y_pred_proba.shape}")
    
    # -------------------------------------------------------------------------
    # Compute Metrics
    # -------------------------------------------------------------------------
    print("\n📊 Computing metrics...")
    
    threshold = args.threshold
    
    if args.use_optimal_threshold:
        print("  Finding optimal thresholds...")
        optimal_thresholds, threshold_info = find_optimal_thresholds(
            y_true, y_pred_proba, method='f1'
        )
        print(f"  Optimal thresholds: {optimal_thresholds}")
        
        # Compute metrics with per-label thresholds
        y_pred = np.zeros_like(y_pred_proba)
        for i in range(y_pred_proba.shape[1]):
            y_pred[:, i] = (y_pred_proba[:, i] >= optimal_thresholds[i]).astype(int)
        
        # Compute metrics manually
        from multilabel.metrics import (
            compute_auc_metrics, compute_f1_metrics, 
            compute_auprc, compute_per_disease_metrics
        )
        
        pathologies = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        
        auc_metrics = compute_auc_metrics(y_true, y_pred_proba)
        f1_metrics = compute_f1_metrics(y_true, y_pred)
        auprc_metrics = compute_auprc(y_true, y_pred_proba)
        
        per_disease = {}
        for i, name in enumerate(pathologies):
            per_disease[name] = compute_per_disease_metrics(
                y_true[:, i], y_pred[:, i], y_pred_proba[:, i]
            )
            per_disease[name]['optimal_threshold'] = float(optimal_thresholds[i])
        
        metrics = {
            'macro_auc': auc_metrics['macro_auc'],
            'micro_auc': auc_metrics['micro_auc'],
            'per_label_auc': auc_metrics['per_label_auc'],
            'macro_f1': f1_metrics['macro_f1'],
            'micro_f1': f1_metrics['micro_f1'],
            'multilabel_auprc': auprc_metrics['macro_auprc'],
            'micro_auprc': auprc_metrics['micro_auprc'],
            'per_disease': per_disease,
            'n_samples': len(y_true),
            'n_labels': 5,
            'thresholds_used': optimal_thresholds.tolist(),
        }
    else:
        metrics = compute_multilabel_metrics(y_true, y_pred_proba, threshold=threshold)
    
    # -------------------------------------------------------------------------
    # Output Results
    # -------------------------------------------------------------------------
    if args.print_report:
        print("\n" + format_metrics_report(metrics))
    
    # Save to JSON
    if args.output:
        output_data = {
            'model_path': args.model_path,
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'cxr8_csv': cxr8_csv,
            'metrics': metrics,
        }
        
        # Handle numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        output_data = convert_numpy(output_data)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (CXR8 Test Set)")
    print("=" * 60)
    print(f"  Macro-AUC:   {metrics['macro_auc']:.4f}" if metrics['macro_auc'] else "  Macro-AUC:   N/A")
    print(f"  Micro-AUC:   {metrics['micro_auc']:.4f}" if metrics['micro_auc'] else "  Micro-AUC:   N/A")
    print(f"  Macro-AUPRC: {metrics['multilabel_auprc']:.4f}" if metrics['multilabel_auprc'] else "  Macro-AUPRC: N/A")
    print(f"  Micro-AUPRC: {metrics['micro_auprc']:.4f}" if metrics['micro_auprc'] else "  Micro-AUPRC: N/A")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    try:
        metrics = main()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise

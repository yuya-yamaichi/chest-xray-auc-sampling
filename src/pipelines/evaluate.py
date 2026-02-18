#!/usr/bin/env python3
"""
Final Evaluation Script for Master's Thesis Research
Synergistic Effects of Sampling Methods and AUC Optimization

This script performs the final evaluation of trained models on the unseen CXR8 test dataset
to generate definitive performance metrics for thesis reporting. It loads a trained model
and calculates comprehensive evaluation metrics on completely unseen data.

Usage (single model):
    python src/pipelines/evaluate.py --model_path models/smote_seed42.pth --model_name smote_seed42

Usage (from training log CSV):
    python src/pipelines/evaluate.py --training_log results/training_log_edema.csv
    # オプションでフィルタ
    python src/pipelines/evaluate.py --training_log results/training_log_edema.csv \
        --seed 42 --sampler RandomOverSampler --limit 3

Author: Master's Thesis Research
Date: August 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix
)

# Torchvision functional for simple TTA
import torchvision.transforms.functional as TF
# LibAUC imports for model architecture
from libauc.models import densenet121

# Configuration management
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import load_config, get_device_config

# Resolve project root (two levels up from this file: src/pipelines/ → project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# =====================================================================================
# GLOBAL CONFIGURATION
# =====================================================================================

# Load configuration from YAML file
CONFIG = load_config()

# =====================================================================================
# COMMAND LINE ARGUMENT PARSING  
# =====================================================================================

def parse_arguments():
    """
    Parse and validate command line arguments for model evaluation.
    
    Returns:
        argparse.Namespace: Parsed arguments containing model_path and model_name
    """
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on CXR8 test dataset for thesis research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/pipelines/evaluate.py --model_path models/smote_seed42.pth --model_name smote_seed42
    python src/pipelines/evaluate.py --model_path models/baseline_seed123.pth --model_name baseline_seed123
        """
    )

    # Mutually exclusive input modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--model_path',
        type=str,
        help='Path to the trained model checkpoint (.pth file)'
    )
    group.add_argument(
        '--training_log',
        type=str,
        help='Path to a training log CSV (e.g., results/training_log_edema.csv) to evaluate listed models'
    )

    # Optional for single-model mode
    parser.add_argument(
        '--model_name',
        type=str,
        help='Descriptive name for the model (used for CSV logging). Defaults to checkpoint basename when omitted.'
    )

    # Dataset settings
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/yamaichi/CXR8/dataset/test_dataset/',
        help='Root directory of the CXR8 test dataset'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default=None,
        help='Explicit path to CXR8 test CSV (e.g., cxr8_test.csv). Overrides auto-discovery.'
    )
    parser.add_argument(
        '--use_frontal_only',
        action='store_true',
        help='Filter evaluation to frontal views only (e.g., PA/AP) when such a column exists.'
    )
    parser.add_argument(
        '--norm_mean', type=str, default=None,
        help='Override normalization mean as comma-separated floats, e.g., 0.485,0.456,0.406'
    )
    parser.add_argument(
        '--norm_std', type=str, default=None,
        help='Override normalization std as comma-separated floats, e.g., 0.229,0.224,0.225'
    )
    parser.add_argument(
        '--tta', type=str, default='none', choices=['none', 'hflip'],
        help='Enable simple test-time augmentation (currently: horizontal flip)'
    )
    parser.add_argument(
        '--bn_adapt', action='store_true',
        help='Enable BatchNorm adaptation on test data (updates running stats without gradients)'
    )
    parser.add_argument(
        '--bn_adapt_passes', type=int, default=1,
        help='Number of passes over test data for BN adaptation (default: 1)'
    )
    parser.add_argument(
        '--view_filter', type=str, default=None,
        help='Filter by specific view(s) when available, e.g., PA or PA,AP'
    )
    parser.add_argument(
        '--finding_labels_col', type=str, default='Finding Labels',
        help='Column name for pipe-separated labels in CXR8 CSV (default: Finding Labels)'
    )
    parser.add_argument(
        '--split_list', type=str, default=None,
        help='Path to split list (e.g., test_list.txt) to subset evaluation'
    )
    parser.add_argument(
        '--images_root', type=str, default=None,
        help='Directory containing CXR8 images (e.g., /path/to/images). Overrides auto-discovery.'
    )
    parser.add_argument(
        '--cxr8_strict', action='store_true',
        help='Enable strict CXR8 mode (use split list, derive labels from Finding Labels, apply view_filter, TTA/BN adapt)'
    )
    parser.add_argument(
        '--cxr8_label_column', type=str, default=None,
        help='Explicitly set the label column for CXR8 (e.g., "Pleural Effusion")'
    )
    parser.add_argument(
        '--domain_adapt', action='store_true',
        help='Enable full domain adaptation: TTA + BN adapt + stricter view filtering'
    )
    parser.add_argument(
        '--confidence_threshold', type=float, default=None,
        help='Filter predictions above threshold for precision-focused evaluation'
    )
    parser.add_argument(
        '--pathology',
        type=str,
        default=None,
        help='Pathology name to select label column (e.g., Cardiomegaly, Edema). If omitted, auto-detect.'
    )

    # Filters for training_log mode
    parser.add_argument('--seed', type=int, default=None, help='Filter: seed value')
    parser.add_argument('--sampler', type=str, default=None, help='Filter: sampler name')
    parser.add_argument('--ratio', type=str, default=None, help='Filter: requested sampling ratio (as in CSV)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of models to evaluate from training_log')
    parser.add_argument('--skip_if_exists', action='store_true', help='Skip models already evaluated (by model_path)')

    return parser.parse_args()

# =====================================================================================
# CXR8 DATASET CLASS
# =====================================================================================

class CXR8TestDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading CXR8 chest X-ray test dataset.
    
    This class handles loading of the test dataset from:
    /home/yamaichi/CXR8/dataset/test_dataset/
    ├── test.csv (or similar)  # CSV file with image paths and Cardiomegaly labels  
    └── images/               # Directory containing chest X-ray images
    """
    
    def __init__(self, dataset_root, transform=None, label_column_override=None, test_csv_explicit=None,
                 use_frontal_only=False, view_filter: str = None,
                 finding_labels_col: str = 'Finding Labels', split_list: str = None,
                 cxr8_label_tag: str = None, images_root: str = None,
                 strict_mode: bool = False):
        """
        Initialize the CXR8 test dataset.
        
        Args:
            dataset_root (str): Root directory of the CXR8 test dataset
            transform (callable, optional): Image transformations to apply
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.label_column_override = label_column_override
        self.images_root = None
        
        # Define the path to the test CSV file - use explicit if provided; otherwise try candidates
        possible_csv_names = ['cxr8_test.csv', 'test.csv', '@test.csv', 'test_list.txt', 'Data_Entry_2017.csv']
        self.csv_path = None
        
        if test_csv_explicit is not None:
            self.csv_path = test_csv_explicit if os.path.isabs(test_csv_explicit) else os.path.join(dataset_root, test_csv_explicit)
        else:
            for csv_name in possible_csv_names:
                test_path = os.path.join(dataset_root, csv_name)
                if os.path.exists(test_path):
                    self.csv_path = test_path
                    break
        
        if self.csv_path is None:
            # List available files to help debug
            available_files = []
            if os.path.exists(dataset_root):
                available_files = [f for f in os.listdir(dataset_root) if f.endswith(('.csv', '.txt'))]
            raise FileNotFoundError(
                f"No test CSV file found in {dataset_root}\n"
                f"Looked for: {possible_csv_names}\n"
                f"Available CSV/TXT files: {available_files}\n"
                f"Please check the dataset structure."
            )
        
        # Verify that the CSV file exists
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Test CSV file not found: {self.csv_path}")
        
        # Load the CSV file
        print(f"Loading CXR8 test dataset from: {self.csv_path}")
        self.data_df = pd.read_csv(self.csv_path)
        # Resolve images root directory
        if images_root is not None and os.path.exists(images_root):
            self.images_root = images_root
        else:
            candidates = [
                os.path.join(dataset_root, 'images'),
                os.path.join(os.path.dirname(dataset_root.rstrip(os.sep)), 'images')
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.images_root = c
                    break
        if self.images_root:
            print(f"✓ Images root: {self.images_root}")
        
        # Check for different possible column name combinations for CXR8 dataset
        possible_image_cols = ['Path', 'Image Index', 'Image_Index', 'filename', 'image_path']
        # Include common single-disease columns that may appear in prepared test CSVs
        possible_label_cols = [
            'Cardiomegaly', 'cardiomegaly',
            'Edema', 'edema',
            'Atelectasis', 'atelectasis',
            'Consolidation', 'consolidation',
            'Pleural Effusion', 'Pleural_Effusion', 'pleural_effusion', 'pleural effusion',
            'label', 'target'
        ]

        # Optional split subset (e.g., test_list.txt)
        if strict_mode and split_list is not None:
            split_path = split_list if os.path.isabs(split_list) else os.path.join(dataset_root, split_list)
            if os.path.exists(split_path):
                try:
                    with open(split_path, 'r') as f:
                        ids = set(line.strip() for line in f if line.strip())
                    # Try to match by basename against common image id columns
                    def _basename(x: str) -> str:
                        try:
                            return os.path.basename(str(x))
                        except Exception:
                            return str(x)
                    id_col = None
                    for c in ['Image Index', 'Image_Index', 'Path', 'image_path', 'filename']:
                        if c in self.data_df.columns:
                            id_col = c
                            break
                    if id_col is not None:
                        before = len(self.data_df)
                        self.data_df = self.data_df[self.data_df[id_col].apply(lambda v: _basename(v) in ids)]
                        after = len(self.data_df)
                        print(f"✓ Split list applied on '{id_col}': {before} → {after}")
                except Exception as e:
                    print(f"⚠ Failed to apply split list '{split_path}': {e}")

        # Optional frontal view filter
        # Only honor view_filter in strict mode; always honor use_frontal_only
        if use_frontal_only or (strict_mode and view_filter is not None):
            # Common view columns in NIH CXR8 derivatives
            possible_view_cols = ['View Position', 'ViewPosition', 'view', 'View']
            view_col = next((c for c in possible_view_cols if c in self.data_df.columns), None)
            if view_col is not None:
                before = len(self.data_df)
                if strict_mode and view_filter is not None:
                    allowed = [v.strip().upper() for v in str(view_filter).split(',') if v.strip()]
                else:
                    allowed = ['PA','AP']
                self.data_df = self.data_df[self.data_df[view_col].astype(str).str.upper().isin(allowed)]
                after = len(self.data_df)
                print(f"✓ View filter applied on column '{view_col}' ({allowed}): {before} → {after}")
        
        # Find the correct column names
        image_col = None
        label_col = None
        
        for col in possible_image_cols:
            if col in self.data_df.columns:
                image_col = col
                break
                
        # Determine label column
        if self.label_column_override is not None and self.label_column_override in self.data_df.columns:
            label_col = self.label_column_override
        else:
            for col in possible_label_cols:
                if col in self.data_df.columns:
                    label_col = col
                    break
            # Derive from Finding Labels if per-label column is not present (strict mode only)
            if strict_mode and label_col is None and finding_labels_col in self.data_df.columns:
                # Map provided pathology to CXR8 tag if explicit tag not given
                def _map_to_cxr8_tag(name: str) -> str:
                    if cxr8_label_tag:
                        return cxr8_label_tag
                    if name is None:
                        return None
                    n = str(name).strip().lower().replace('_', ' ')
                    mapping = {
                        'pleural effusion': 'Effusion',
                        'edema': 'Edema',
                        'cardiomegaly': 'Cardiomegaly',
                        'atelectasis': 'Atelectasis',
                        'consolidation': 'Consolidation'
                    }
                    return mapping.get(n, name)
                target = _map_to_cxr8_tag(self.label_column_override)
                if target is None:
                    # Try default common case
                    target = 'Cardiomegaly'
                def _has_target(lbl: str) -> int:
                    try:
                        parts = [p.strip() for p in str(lbl).split('|')]
                        return 1 if target in parts else 0
                    except Exception:
                        return 0
                self.data_df['label'] = self.data_df[finding_labels_col].apply(_has_target)
                label_col = 'label'
        
        if image_col is None or label_col is None:
            print(f"Available columns: {list(self.data_df.columns)}")
            raise ValueError(
                f"Could not find required columns.\n"
                f"Image column: {image_col} (looked for: {possible_image_cols})\n"
                f"Label column: {label_col} (looked for: {possible_label_cols})"
            )
        
        self.image_col = image_col
        self.label_col = label_col
        print(f"✓ Using image column: '{image_col}', label column: '{label_col}'")
        
        # Filter out samples with missing labels
        initial_count = len(self.data_df)
        self.data_df = self.data_df.dropna(subset=[label_col])
        final_count = len(self.data_df)
        
        if initial_count != final_count:
            print(f"Warning: Removed {initial_count - final_count} samples with missing {label_col} labels")
        
        # Convert labels to binary (0/1)
        self.data_df[label_col] = self.data_df[label_col].astype(int)
        
        # Quick check for images directory
        images_dir = os.path.join(dataset_root, 'images')
        if os.path.exists(images_dir):
            print(f"✓ Images directory found")
        
        print(f"✓ Loaded {len(self.data_df)} test samples")
        print(f"✓ Label distribution: {self.data_df[label_col].value_counts().to_dict()}")
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_df)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, label) where image_tensor is a torch.Tensor
                   and label is a float (0.0 or 1.0)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label from the dataframe
        row = self.data_df.iloc[idx]
        image_path = row[self.image_col]
        label = float(row[self.label_col])
        
        # Construct full image path
        if not os.path.isabs(image_path):
            candidates = []
            # If the column is a path like images/xxx, try dataset_root and its parent
            rel_path = str(image_path)
            candidates.append(os.path.join(self.dataset_root, rel_path))
            parent_root = os.path.dirname(self.dataset_root.rstrip(os.sep))
            candidates.append(os.path.join(parent_root, rel_path))
            # Try images_root with basename
            base = os.path.basename(rel_path)
            if self.images_root:
                candidates.append(os.path.join(self.images_root, base))
            # Also try dataset_root/images with basename for safety
            candidates.append(os.path.join(self.dataset_root, 'images', base))

            full_image_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    full_image_path = cand
                    break
            if full_image_path is None:
                # Fallback to first candidate
                full_image_path = candidates[0]
        else:
            full_image_path = image_path
        
        # Skip file existence check for speed - let PIL handle file errors
        # This saves thousands of os.path.exists() calls
        
        # Load and preprocess the image
        try:
            image = Image.open(full_image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {full_image_path}: {str(e)}")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

# =====================================================================================
# IMAGE PREPROCESSING
# =====================================================================================

def get_evaluation_transforms(norm_mean_override=None, norm_std_override=None):
    """
    Get image transformations for evaluation that match training preprocessing.
    
    Returns:
        torchvision.transforms.Compose: Composed image transformations
    """
    # Get image size from configuration
    image_size = CONFIG.get('dataset.image_size', 224)
    
    # Compose normalization parameters
    if norm_mean_override is not None and norm_std_override is not None:
        norm_mean = [float(x) for x in str(norm_mean_override).split(',')]
        norm_std = [float(x) for x in str(norm_std_override).split(',')]
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    return transform

# =====================================================================================
# MODEL LOADING
# =====================================================================================

def load_trained_model(model_path):
    """
    Load a trained DenseNet121 model from checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        
    Returns:
        torch.nn.Module: Loaded model ready for evaluation
    """
    print(f"Loading trained model from: {model_path}")
    
    # Get device configuration
    device = get_device_config(CONFIG)
    
    # Verify model file exists (resolve relative to project root if needed)
    resolved_model_path = model_path
    if not os.path.isabs(resolved_model_path):
        candidate = os.path.join(PROJECT_ROOT, resolved_model_path)
        if os.path.exists(candidate):
            resolved_model_path = candidate
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Get model configuration from YAML
    num_classes = CONFIG.get('model.num_classes', 1)
    activation = CONFIG.get('model.activation', 'relu')
    last_activation = CONFIG.get('model.last_activation', 'sigmoid')
    
    # Initialize DenseNet121 model architecture
    model = densenet121(
        pretrained=False,
        last_activation=last_activation,  # Built-in sigmoid activation
        activations=activation,
        num_classes=num_classes
    )
    
    # Load the trained weights
    try:
        state_dict = torch.load(resolved_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {str(e)}")
    
    # Move model to appropriate device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def resolve_label_column_from_pathology(pathology_name: str) -> str:
    """
    Map pathology name (as appears in training logs) to a label column name
    likely present in the prepared CXR8 test CSV.
    """
    if pathology_name is None:
        return None
    name = str(pathology_name).strip().lower().replace('_', ' ')
    mapping = {
        'cardiomegaly': 'Cardiomegaly',
        'edema': 'Edema',
        'atelectasis': 'Atelectasis',
        'consolidation': 'Consolidation',
        'pleural effusion': 'Pleural Effusion',
    }
    # Normalize keys in mapping
    for key, val in mapping.items():
        if name == key:
            return val
    # Fallback: title-case
    return pathology_name

# =====================================================================================
# MODEL EVALUATION
# =====================================================================================

def perform_model_evaluation(model, dataloader, tta_mode: str = 'none'):
    """
    Perform model inference on the test dataset and collect predictions.
    
    Args:
        model (torch.nn.Module): Trained model in evaluation mode
        dataloader (torch.utils.data.DataLoader): DataLoader for test dataset
        
    Returns:
        tuple: (y_true, y_pred_proba) numpy arrays for metric calculation
    """
    print("Starting model evaluation on CXR8 test dataset...")
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    # Get device configuration
    device = get_device_config(CONFIG)
    
    # Perform inference on all test samples
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass through model
            outputs = model(images)

            # Simple TTA: horizontal flip averaging
            if tta_mode == 'hflip':
                try:
                    images_flipped = torch.flip(images, dims=[3])
                    outputs_flipped = model(images_flipped)
                    outputs = (outputs + outputs_flipped) / 2.0
                except Exception:
                    pass
            predictions = outputs.squeeze().cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Handle single sample batch case
            if predictions.ndim == 0:
                predictions = np.array([predictions])
                labels_np = np.array([labels_np])
            
            # Collect predictions and labels
            all_predictions.extend(predictions)
            all_true_labels.extend(labels_np)
            
            # Progress reporting - less frequent for speed
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Convert to numpy arrays
    y_true = np.array(all_true_labels)
    y_pred_proba = np.array(all_predictions)
    
    print(f"✓ Evaluation completed on {len(y_true)} samples | True pos: {np.sum(y_true == 1)}/{len(y_true)}")
    
    return y_true, y_pred_proba

# =====================================================================================
# METRICS CALCULATION
# =====================================================================================

def calculate_all_metrics(y_true, y_pred_proba):
    """
    Calculate all 7 required performance metrics for thesis reporting.
    
    Args:
        y_true (np.ndarray): True binary labels (0 or 1)
        y_pred_proba (np.ndarray): Predicted probabilities (0 to 1)
        
    Returns:
        dict: Dictionary containing all 7 calculated metrics
    """
    print("Calculating comprehensive performance metrics...")
    
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Calculate confusion matrix once for efficiency
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Initialize metrics dictionary and calculate all efficiently
    metrics = {}
    
    try:
        # Calculate all metrics efficiently
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred_binary)
        
        # G-Mean using pre-calculated confusion matrix
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall_calculated = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['g_mean'] = np.sqrt(recall_calculated * specificity)
        
        print(f"✓ All 7 metrics calculated | Predicted pos: {np.sum(y_pred_binary)}/{len(y_pred_binary)}")
        
    except Exception as e:
        print(f"✗ Error calculating metrics: {str(e)}")
        raise
    
    return metrics

# =====================================================================================
# RESULTS DISPLAY AND SAVING
# =====================================================================================

def display_evaluation_results(model_name, metrics):
    """
    Display evaluation results in a clear, formatted manner.
    
    Args:
        model_name (str): Name of the evaluated model
        metrics (dict): Dictionary of calculated metrics
    """
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS - MASTER'S THESIS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: CXR8 Test Set")
    print(f"Target: Binary disease detection (per selected label)")
    print("-" * 80)
    
    # Display metrics in formatted table
    print(f"{'Metric':<20} {'Value':<10} {'Description'}")
    print("-" * 60)
    print(f"{'AUC':<20} {metrics['auc']:<10.4f} Area Under ROC Curve")
    print(f"{'AUPRC':<20} {metrics['auprc']:<10.4f} Area Under Precision-Recall Curve")
    print(f"{'F1-Score':<20} {metrics['f1_score']:<10.4f} Harmonic Mean of Precision/Recall")
    print(f"{'Precision':<20} {metrics['precision']:<10.4f} Positive Predictive Value")
    print(f"{'Recall':<20} {metrics['recall']:<10.4f} Sensitivity/True Positive Rate")
    print(f"{'Balanced Accuracy':<20} {metrics['balanced_accuracy']:<10.4f} Average of TPR and TNR")
    print(f"{'G-Mean':<20} {metrics['g_mean']:<10.4f} Geometric Mean of Recall/Specificity")
    
    print("=" * 80)

def save_results_to_csv_file(model_name, metrics, extra_info=None):
    """
    Save evaluation results to CSV file for thesis documentation.
    
    Args:
        model_name (str): Name of the evaluated model
        metrics (dict): Dictionary of calculated metrics
    """
    # Prepare results row for CSV
    results_row = {
        'model_name': model_name,
        'auc': round(metrics['auc'], 4),
        'auprc': round(metrics['auprc'], 4),
        'f1_score': round(metrics['f1_score'], 4),
        'precision': round(metrics['precision'], 4),
        'recall': round(metrics['recall'], 4),
        'balanced_accuracy': round(metrics['balanced_accuracy'], 4),
        'g_mean': round(metrics['g_mean'], 4)
    }

    # Attach any extra metadata from training logs
    if isinstance(extra_info, dict):
        results_row.update(extra_info)
    
    # Define CSV file path
    results_dir = 'results'
    results_filename = 'final_evaluation_metrics.csv'
    results_path = os.path.join(results_dir, results_filename)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Handle CSV file creation and appending
    if os.path.exists(results_path):
        # Append to existing file
        existing_df = pd.read_csv(results_path)
        new_row_df = pd.DataFrame([results_row])
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        combined_df.to_csv(results_path, index=False)
        print(f"✓ Results appended to existing file: {results_path}")
    else:
        # Create new file with header
        new_df = pd.DataFrame([results_row])
        new_df.to_csv(results_path, index=False)
        print(f"✓ Results saved to new file: {results_path}")
    
    print(f"✓ Evaluation results logged for model: {model_name}")


def evaluate_single_model(model_path: str, model_name: str, dataset_root: str, pathology_for_label: str,
                          test_csv_explicit: str = None, use_frontal_only: bool = False,
                          norm_mean_override: str = None, norm_std_override: str = None,
                          tta_mode: str = 'none', view_filter: str = None,
                          bn_adapt: bool = False, bn_adapt_passes: int = 1,
                          finding_labels_col: str = 'Finding Labels', split_list: str = None,
                          cxr8_label_tag: str = None, images_root: str = None,
                          cxr8_strict: bool = False):
    """
    Helper to evaluate a single model and persist the results.
    """
    # Step 2: Load the trained model
    model = load_trained_model(model_path)

    # Step 3: Load the CXR8 test dataset
    label_override = resolve_label_column_from_pathology(pathology_for_label)

    transforms_fn = get_evaluation_transforms(norm_mean_override, norm_std_override)
    print("Loading CXR8 test dataset...")
    # Gate strict features so that legacy mode behaves exactly as before
    eff_view_filter = view_filter if cxr8_strict else None
    eff_tta = tta_mode if cxr8_strict else 'none'
    eff_bn_adapt = bn_adapt if cxr8_strict else False
    eff_bn_passes = bn_adapt_passes if cxr8_strict else 1
    eff_split = split_list if cxr8_strict else None

    test_dataset = CXR8TestDataset(
        dataset_root=dataset_root,
        transform=transforms_fn,
        label_column_override=label_override,
        test_csv_explicit=test_csv_explicit,
        use_frontal_only=use_frontal_only,
        view_filter=eff_view_filter,
        finding_labels_col=finding_labels_col,
        split_list=eff_split,
        cxr8_label_tag=cxr8_label_tag,
        images_root=images_root,
        strict_mode=cxr8_strict,
    )

    # Create data loader
    batch_size = CONFIG.get('evaluation.batch_size', 128)
    num_workers = CONFIG.get('evaluation.num_workers', 4)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    print(f"✓ Test batches: {len(test_dataloader)} (batch_size={batch_size}, workers={num_workers})")
    print()

    # Optional: BatchNorm adaptation (update running stats with test distribution)
    if eff_bn_adapt:
        try:
            print(f"Adapting BatchNorm statistics with {eff_bn_passes} pass(es)...")
            model.train()
            device = get_device_config(CONFIG)
            with torch.no_grad():
                for _ in range(max(1, int(eff_bn_passes))):
                    for images, _ in test_dataloader:
                        images = images.to(device)
                        _ = model(images)
            model.eval()
            print("✓ BN adaptation complete")
        except Exception as e:
            print(f"⚠ BN adaptation skipped due to error: {e}")

    # Step 4: Execute evaluation
    y_true, y_pred_proba = perform_model_evaluation(model, test_dataloader, eff_tta)

    # Step 5: Calculate all 7 performance metrics
    metrics = calculate_all_metrics(y_true, y_pred_proba)

    # Step 6: Display
    display_evaluation_results(model_name, metrics)

    return metrics


def read_results_index(results_path: str) -> pd.DataFrame:
    if os.path.exists(results_path):
        try:
            return pd.read_csv(results_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _read_csv_lenient(file_path: str) -> pd.DataFrame:
    """
    Lenient CSV reader that tolerates rows with extra fields compared to header.
    It truncates any extra fields beyond the header count, preserving the leading
    columns as defined by the header. This is useful when an existing CSV file
    has evolved to include new columns mid-file.

    Args:
        file_path: Path to the CSV file

    Returns:
        pandas.DataFrame with columns from header and rows truncated to header length
    """
    try:
        # Try normal read first with the Python engine and skip bad lines
        return pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    except Exception:
        pass

    # Manual fallback: truncate rows to header length
    try:
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        if not lines:
            return pd.DataFrame()
        headers = lines[0].split(',')
        num_cols = len(headers)
        rows = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(',', num_cols - 1)
            # Skip if even after splitting we do not have enough columns
            if len(parts) < num_cols:
                continue
            rows.append(parts[:num_cols])
        return pd.DataFrame(rows, columns=headers)
    except Exception:
        # Last resort: empty frame
        return pd.DataFrame()

# =====================================================================================
# MAIN EXECUTION FUNCTION
# =====================================================================================

def main():
    """
    Main execution function that orchestrates the complete evaluation pipeline.
    """
    print("🔬 FINAL MODEL EVALUATION - MASTER'S THESIS RESEARCH")
    print("=" * 70)
    print("Evaluating trained model(s) on unseen CXR8 test dataset")
    print("Target: Binary classification metrics (AUC, AUPRC, F1, etc.)")
    print("=" * 70)
    
    # Parse command line arguments
    args = parse_arguments()

    # Display configuration
    device = get_device_config(CONFIG)
    print(f"Device: {device}")
    if args.model_path:
        print(f"Model Path: {args.model_path}")
        print(f"Model Name: {args.model_name or os.path.splitext(os.path.basename(args.model_path))[0]}")
    if args.training_log:
        print(f"Training Log: {args.training_log}")
        if args.seed is not None:
            print(f"Filter seed: {args.seed}")
        if args.sampler is not None:
            print(f"Filter sampler: {args.sampler}")
        if args.ratio is not None:
            print(f"Filter ratio: {args.ratio}")
        if args.limit is not None:
            print(f"Limit: {args.limit}")
        if args.skip_if_exists:
            print("Skip if already evaluated: ON")
    print()

    try:
        # SINGLE MODEL MODE
        if args.model_path is not None:
            # Validate inputs
            # Resolve relative path against project root for validation
            model_path_for_check = args.model_path
            if not os.path.isabs(model_path_for_check):
                model_path_for_check = os.path.join(PROJECT_ROOT, model_path_for_check)
            if not os.path.exists(model_path_for_check):
                raise FileNotFoundError(f"Model file not found: {args.model_path}")

            model_name = args.model_name or os.path.splitext(os.path.basename(args.model_path))[0]
            metrics = evaluate_single_model(
                model_path=args.model_path,
                model_name=model_name,
                dataset_root=args.dataset_root,
                pathology_for_label=args.pathology,
                test_csv_explicit=args.test_csv,
                use_frontal_only=args.use_frontal_only,
                norm_mean_override=args.norm_mean,
                norm_std_override=args.norm_std,
                tta_mode=args.tta,
                view_filter=args.view_filter,
                bn_adapt=args.bn_adapt,
                bn_adapt_passes=args.bn_adapt_passes,
                finding_labels_col=args.finding_labels_col,
                split_list=args.split_list,
                cxr8_label_tag=args.cxr8_label_column,
                images_root=args.images_root,
                cxr8_strict=args.cxr8_strict,
            )
            save_results_to_csv_file(model_name, metrics, extra_info={'model_path': args.model_path, 'pathology': args.pathology})

            print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            print("📊 Results saved to: results/final_evaluation_metrics.csv")
            print("📋 All 7 metrics calculated and logged for thesis documentation")
            return

        # TRAINING LOG MODE
        if args.training_log is not None:
            if not os.path.exists(args.training_log):
                raise FileNotFoundError(f"Training log CSV not found: {args.training_log}")

            # Robust CSV loading to tolerate schema drift across appended rows
            try:
                df = pd.read_csv(args.training_log)
            except Exception as e:
                print(f"Warning: Failed to parse training log with default parser ({str(e)}). Using lenient reader...")
                df = _read_csv_lenient(args.training_log)
                if df.empty:
                    raise RuntimeError(f"Unable to read training log CSV even with lenient parser: {args.training_log}")

            # Basic sanity columns
            required_cols = ['model_path']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Training log missing required column: {col}")

            # Optional filters
            if args.seed is not None and 'seed' in df.columns:
                df = df[df['seed'] == args.seed]
            if args.sampler is not None and 'sampler' in df.columns:
                df = df[df['sampler'] == args.sampler]
            if args.ratio is not None and 'sampling_ratio_requested' in df.columns:
                df = df[df['sampling_ratio_requested'].astype(str) == str(args.ratio)]

            # Remove rows with missing paths
            df = df[df['model_path'].notna()]

            # If pathology column exists, we will use per-row pathology; otherwise fall back to CLI pathology
            has_pathology = 'pathology' in df.columns

            # Skip already evaluated models if requested
            evaluated_index = None
            results_csv_path = os.path.join('results', 'final_evaluation_metrics.csv')
            if args.skip_if_exists:
                evaluated_index = read_results_index(results_csv_path)

            # Limit number of rows if requested
            if args.limit is not None:
                df = df.head(args.limit)

            # Iterate and evaluate
            total = len(df)
            print(f"Found {total} model(s) to evaluate from training log")
            for idx, row in df.iterrows():
                model_path = row['model_path']
                model_name = os.path.splitext(os.path.basename(model_path))[0]

                # Skip if results already contain this model_path
                if evaluated_index is not None and not evaluated_index.empty:
                    if 'model_path' in evaluated_index.columns and (evaluated_index['model_path'] == model_path).any():
                        print(f"- Skipping already evaluated model: {model_name}")
                        continue

                # Determine pathology for label selection
                pathology = args.pathology
                if has_pathology:
                    try:
                        pathology = row['pathology']
                    except Exception:
                        pass

                print(f"\n[{model_name}] Evaluating... (pathology={pathology})")
                metrics = evaluate_single_model(
                    model_path=model_path,
                    model_name=model_name,
                    dataset_root=args.dataset_root,
                    pathology_for_label=pathology,
                    test_csv_explicit=args.test_csv,
                    use_frontal_only=args.use_frontal_only,
                    norm_mean_override=args.norm_mean,
                    norm_std_override=args.norm_std,
                    tta_mode=args.tta,
                    view_filter=args.view_filter,
                    bn_adapt=args.bn_adapt,
                    bn_adapt_passes=args.bn_adapt_passes,
                    finding_labels_col=args.finding_labels_col,
                    split_list=args.split_list,
                    cxr8_label_tag=args.cxr8_label_column,
                    images_root=args.images_root,
                    cxr8_strict=args.cxr8_strict,
                )

                # Prepare extra info from row (include hyperparams and applied flag when available)
                extra_cols = {}
                for c in ['model_path', 'pathology', 'class_id', 'sampler', 'seed', 'best_val_auc', 'training_time_sec',
                          'original_class_0', 'original_class_1', 'final_class_0', 'final_class_1',
                          'sampling_ratio_achieved', 'sampling_strategy_used', 'sampling_applied', 'sampling_hyperparams',
                          'sampling_ratio_requested', 'timestamp']:
                    if c in row.index:
                        extra_cols[c] = row[c]

                save_results_to_csv_file(model_name, metrics, extra_info=extra_cols)

            print("\n🎉 BATCH EVALUATION COMPLETED!")
            print("📊 Results saved to: results/final_evaluation_metrics.csv")
            return

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
Multi-label Evaluation Metrics

Comprehensive evaluation metrics for multi-label chest X-ray classification.
Implements: Macro/Micro AUC, Macro/Micro F1, Multi-label AUPRC, and per-disease metrics.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
    balanced_accuracy_score,
    confusion_matrix
)
import warnings


# Pathology names for reporting
PATHOLOGIES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    pathology_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute all required multi-label evaluation metrics.
    
    Args:
        y_true: True labels (N, K), binary {0, 1}
        y_pred_proba: Predicted probabilities (N, K), range [0, 1]
        threshold: Threshold for converting probabilities to binary predictions
        pathology_names: Optional list of pathology names
    
    Returns:
        dict: Comprehensive metrics including:
            - macro_auc, micro_auc
            - macro_f1, micro_f1
            - multilabel_auprc
            - per_disease: dict of per-disease metrics
    """
    if pathology_names is None:
        pathology_names = PATHOLOGIES
    
    n_samples, n_labels = y_true.shape
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute AUC metrics
    auc_metrics = compute_auc_metrics(y_true, y_pred_proba)
    
    # Compute F1 metrics
    f1_metrics = compute_f1_metrics(y_true, y_pred)
    
    # Compute AUPRC
    auprc = compute_auprc(y_true, y_pred_proba)
    
    # Compute per-disease metrics
    per_disease = {}
    for i, name in enumerate(pathology_names):
        per_disease[name] = compute_per_disease_metrics(
            y_true[:, i],
            y_pred[:, i],
            y_pred_proba[:, i]
        )
    
    return {
        'macro_auc': auc_metrics['macro_auc'],
        'micro_auc': auc_metrics['micro_auc'],
        'per_label_auc': auc_metrics['per_label_auc'],
        'macro_f1': f1_metrics['macro_f1'],
        'micro_f1': f1_metrics['micro_f1'],
        'multilabel_auprc': auprc['macro_auprc'],
        'micro_auprc': auprc['micro_auprc'],
        'per_disease': per_disease,
        'threshold': threshold,
        'n_samples': n_samples,
        'n_labels': n_labels,
    }


def compute_auc_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Macro-AUC and Micro-AUC.
    
    Macro-AUC: Average AUC across all labels
    Micro-AUC: AUC computed on flattened predictions
    
    Args:
        y_true: True labels (N, K)
        y_pred_proba: Predicted probabilities (N, K)
    
    Returns:
        dict: {'macro_auc', 'micro_auc', 'per_label_auc'}
    """
    n_samples, n_labels = y_true.shape
    
    # Per-label AUC
    per_label_auc = []
    valid_labels = []
    
    for i in range(n_labels):
        # Check if we have both classes
        unique = np.unique(y_true[:, i])
        if len(unique) < 2:
            per_label_auc.append(np.nan)
            continue
        
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            per_label_auc.append(auc)
            valid_labels.append(i)
        except ValueError as e:
            per_label_auc.append(np.nan)
    
    # Macro-AUC (average of per-label AUCs, excluding NaN)
    valid_aucs = [a for a in per_label_auc if not np.isnan(a)]
    macro_auc = np.mean(valid_aucs) if valid_aucs else np.nan
    
    # Micro-AUC (flatten and compute)
    try:
        # Create mask for valid samples (where both classes exist per label)
        micro_auc = roc_auc_score(
            y_true.ravel(),
            y_pred_proba.ravel()
        )
    except ValueError:
        micro_auc = np.nan
    
    return {
        'macro_auc': float(macro_auc) if not np.isnan(macro_auc) else None,
        'micro_auc': float(micro_auc) if not np.isnan(micro_auc) else None,
        'per_label_auc': [float(a) if not np.isnan(a) else None for a in per_label_auc],
    }


def compute_f1_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Macro-F1 and Micro-F1.
    
    Macro-F1: Average F1 across all labels
    Micro-F1: F1 computed on aggregated TP/FP/FN
    
    Args:
        y_true: True labels (N, K)
        y_pred: Predicted labels (N, K)
    
    Returns:
        dict: {'macro_f1', 'micro_f1', 'per_label_f1'}
    """
    n_labels = y_true.shape[1]
    
    # Per-label F1
    per_label_f1 = []
    for i in range(n_labels):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            per_label_f1.append(f1)
        except Exception:
            per_label_f1.append(0.0)
    
    # Macro-F1
    macro_f1 = np.mean(per_label_f1)
    
    # Micro-F1 (aggregate counts across all labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        micro_f1 = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    
    return {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'per_label_f1': [float(f) for f in per_label_f1],
    }


def compute_auprc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Multi-label Area Under Precision-Recall Curve.
    
    Args:
        y_true: True labels (N, K)
        y_pred_proba: Predicted probabilities (N, K)
    
    Returns:
        dict: {'macro_auprc', 'micro_auprc', 'per_label_auprc'}
    """
    n_labels = y_true.shape[1]
    
    # Per-label AUPRC
    per_label_auprc = []
    for i in range(n_labels):
        unique = np.unique(y_true[:, i])
        if len(unique) < 2:
            per_label_auprc.append(np.nan)
            continue
        
        try:
            ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
            per_label_auprc.append(ap)
        except ValueError:
            per_label_auprc.append(np.nan)
    
    # Macro-AUPRC
    valid_auprcs = [a for a in per_label_auprc if not np.isnan(a)]
    macro_auprc = np.mean(valid_auprcs) if valid_auprcs else np.nan
    
    # Micro-AUPRC
    try:
        micro_auprc = average_precision_score(
            y_true.ravel(),
            y_pred_proba.ravel()
        )
    except ValueError:
        micro_auprc = np.nan
    
    return {
        'macro_auprc': float(macro_auprc) if not np.isnan(macro_auprc) else None,
        'micro_auprc': float(micro_auprc) if not np.isnan(micro_auprc) else None,
        'per_label_auprc': [float(a) if not np.isnan(a) else None for a in per_label_auprc],
    }


def compute_per_disease_metrics(
    y_true_col: np.ndarray,
    y_pred_col: np.ndarray,
    y_pred_proba_col: np.ndarray
) -> Dict[str, Any]:
    """
    Compute per-disease metrics: Precision, Recall, Balanced Accuracy, G-Mean.
    
    Args:
        y_true_col: True labels for single disease (N,)
        y_pred_col: Predicted labels for single disease (N,)
        y_pred_proba_col: Predicted probabilities for single disease (N,)
    
    Returns:
        dict: Per-disease metrics
    """
    # Handle edge cases
    unique = np.unique(y_true_col)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Precision
        try:
            precision = precision_score(y_true_col, y_pred_col, zero_division=0)
        except Exception:
            precision = 0.0
        
        # Recall (Sensitivity)
        try:
            recall = recall_score(y_true_col, y_pred_col, zero_division=0)
        except Exception:
            recall = 0.0
        
        # Specificity (for G-Mean calculation)
        cm = confusion_matrix(y_true_col, y_pred_col, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        # Balanced Accuracy
        try:
            balanced_acc = balanced_accuracy_score(y_true_col, y_pred_col)
        except Exception:
            balanced_acc = 0.0
        
        # G-Mean (geometric mean of sensitivity and specificity)
        g_mean = np.sqrt(recall * specificity) if recall > 0 and specificity > 0 else 0.0
        
        # AUC for this disease
        if len(unique) >= 2:
            try:
                auc = roc_auc_score(y_true_col, y_pred_proba_col)
            except ValueError:
                auc = None
        else:
            auc = None
        
        # AUPRC for this disease
        if len(unique) >= 2:
            try:
                auprc = average_precision_score(y_true_col, y_pred_proba_col)
            except ValueError:
                auprc = None
        else:
            auprc = None
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_acc),
        'g_mean': float(g_mean),
        'auc': float(auc) if auc is not None else None,
        'auprc': float(auprc) if auprc is not None else None,
        'pos_count': int((y_true_col == 1).sum()),
        'neg_count': int((y_true_col == 0).sum()),
    }


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = 'f1'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Find optimal classification thresholds per label.
    
    Args:
        y_true: True labels (N, K)
        y_pred_proba: Predicted probabilities (N, K)
        method: Optimization criterion ('f1', 'youden', 'balanced')
    
    Returns:
        tuple: (optimal_thresholds, threshold_info)
    """
    n_labels = y_true.shape[1]
    optimal_thresholds = np.zeros(n_labels)
    threshold_info = {}
    
    for i in range(n_labels):
        unique = np.unique(y_true[:, i])
        if len(unique) < 2:
            optimal_thresholds[i] = 0.5
            continue
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i], y_pred_proba[:, i]
        )
        
        if method == 'f1':
            # Find threshold that maximizes F1
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores[:-1])  # Last one is always 0
            optimal_thresholds[i] = thresholds[best_idx]
            threshold_info[i] = {'threshold': thresholds[best_idx], 'f1': f1_scores[best_idx]}
        
        elif method == 'youden':
            # Youden's J statistic (requires ROC curve)
            from sklearn.metrics import roc_curve
            fpr, tpr, roc_thresholds = roc_curve(y_true[:, i], y_pred_proba[:, i])
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            optimal_thresholds[i] = roc_thresholds[best_idx]
            threshold_info[i] = {'threshold': roc_thresholds[best_idx], 'youden': youden[best_idx]}
        
        elif method == 'balanced':
            # Find threshold closest to balanced precision/recall
            pr_diff = np.abs(precision[:-1] - recall[:-1])
            best_idx = np.argmin(pr_diff)
            optimal_thresholds[i] = thresholds[best_idx]
            threshold_info[i] = {'threshold': thresholds[best_idx]}
        
        else:
            optimal_thresholds[i] = 0.5
    
    return optimal_thresholds, threshold_info


def format_metrics_report(metrics: Dict[str, Any], pathology_names: Optional[List[str]] = None) -> str:
    """
    Format metrics as a human-readable report.
    
    Args:
        metrics: Output from compute_multilabel_metrics()
        pathology_names: Optional list of pathology names
    
    Returns:
        str: Formatted report
    """
    if pathology_names is None:
        pathology_names = PATHOLOGIES
    
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-LABEL CLASSIFICATION METRICS REPORT")
    lines.append("=" * 70)
    
    # Overall metrics
    lines.append("\n📊 OVERALL METRICS")
    lines.append("-" * 40)
    lines.append(f"  Macro-AUC:         {metrics['macro_auc']:.4f}" if metrics['macro_auc'] else "  Macro-AUC:         N/A")
    lines.append(f"  Micro-AUC:         {metrics['micro_auc']:.4f}" if metrics['micro_auc'] else "  Micro-AUC:         N/A")
    lines.append(f"  Macro-F1:          {metrics['macro_f1']:.4f}")
    lines.append(f"  Micro-F1:          {metrics['micro_f1']:.4f}")
    lines.append(f"  Multi-label AUPRC: {metrics['multilabel_auprc']:.4f}" if metrics['multilabel_auprc'] else "  Multi-label AUPRC: N/A")
    
    # Per-disease metrics
    lines.append("\n📋 PER-DISEASE METRICS")
    lines.append("-" * 70)
    header = f"{'Disease':<20} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'Bal.Acc':>8} {'G-Mean':>8}"
    lines.append(header)
    lines.append("-" * 70)
    
    for name in pathology_names:
        if name in metrics['per_disease']:
            d = metrics['per_disease'][name]
            auc_str = f"{d['auc']:.4f}" if d['auc'] else "N/A"
            lines.append(
                f"{name:<20} {auc_str:>8} {d['precision']:>8.4f} "
                f"{d['recall']:>8.4f} {d['balanced_accuracy']:>8.4f} {d['g_mean']:>8.4f}"
            )
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

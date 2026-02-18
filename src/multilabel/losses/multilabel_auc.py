"""
Multi-label AUC Loss Wrapper

Wrapper for LibAUC's AUCM_MultiLabel loss with additional features for
handling uncertain labels and robust batch processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


class MultiLabelAUCLoss(nn.Module):
    """
    Wrapper for LibAUC's AUCM_MultiLabel loss.
    
    Features:
    - Handles uncertain labels (-1) by masking
    - Robust to batches with missing classes
    - Optional class weighting
    
    Args:
        num_classes: Number of labels (default: 5)
        margin: Margin for AUC loss
        imratio_list: Optional list of imbalance ratios per class
        use_label_mask: Whether to mask uncertain (-1) labels
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        margin: float = 1.0,
        imratio_list: Optional[List[float]] = None,
        use_label_mask: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.margin = margin
        self.use_label_mask = use_label_mask
        
        # Try to import LibAUC loss
        try:
            from libauc.losses import AUCM_MultiLabel
            self.libauc_loss = AUCM_MultiLabel(
                num_classes=num_classes,
                margin=margin,
                imratio=imratio_list
            )
            self.use_libauc = True
        except ImportError:
            print("Warning: LibAUC not available, using fallback BCE loss")
            self.libauc_loss = None
            self.use_libauc = False
            self.bce_loss = nn.BCELoss(reduction='none')
        
        # Initialize imbalance ratios
        if imratio_list is not None:
            self.imratio_list = imratio_list
        else:
            self.imratio_list = [0.5] * num_classes
    
    def update_imratio(self, imratio_list: List[float]):
        """Update imbalance ratios for the loss."""
        self.imratio_list = imratio_list
        if self.use_libauc and hasattr(self.libauc_loss, 'imratio'):
            self.libauc_loss.imratio = imratio_list
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-label AUC loss.
        
        Args:
            y_pred: Predictions (B, K), after sigmoid
            y_true: True labels (B, K), values in {-1, 0, 1}
            mask: Optional mask for uncertain labels (B, K)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size, n_classes = y_pred.shape
        
        # Create mask for uncertain labels
        if mask is None and self.use_label_mask:
            # Mask out -1 labels
            mask = (y_true >= 0).float()
        elif mask is None:
            mask = torch.ones_like(y_true)
        
        # Replace -1 with 0 for loss computation
        y_true_clean = torch.clamp(y_true, min=0)
        
        if self.use_libauc:
            # LibAUC loss (doesn't support masking directly)
            loss = self.libauc_loss(y_pred, y_true_clean)
        else:
            # Fallback BCE loss with masking
            bce = self.bce_loss(y_pred, y_true_clean)
            
            # Apply mask and compute mean
            masked_loss = bce * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def get_optimizer_params(self) -> dict:
        """
        Get parameters needed for PESG optimizer if using LibAUC.
        
        Returns:
            dict: Parameters (a, b, alpha) or empty dict
        """
        if self.use_libauc and hasattr(self.libauc_loss, 'a'):
            return {
                'a': self.libauc_loss.a,
                'b': self.libauc_loss.b,
                'alpha': self.libauc_loss.alpha,
            }
        return {}


class WeightedMultiLabelBCE(nn.Module):
    """
    Weighted Binary Cross-Entropy for multi-label classification.
    
    Alternative to AUC loss for comparison experiments.
    
    Args:
        num_classes: Number of labels
        pos_weights: Optional weights for positive class per label
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pos_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pos_weights is not None:
            self.register_buffer('pos_weights', torch.tensor(pos_weights))
        else:
            self.pos_weights = None
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            y_pred: Predictions (B, K), after sigmoid
            y_true: True labels (B, K)
            mask: Optional mask (B, K)
        
        Returns:
            torch.Tensor: Scalar loss
        """
        # Handle uncertain labels
        if mask is None:
            mask = (y_true >= 0).float()
        
        y_true_clean = torch.clamp(y_true, min=0)
        
        # Compute BCE
        eps = 1e-7
        bce = -(
            y_true_clean * torch.log(y_pred + eps) +
            (1 - y_true_clean) * torch.log(1 - y_pred + eps)
        )
        
        # Apply positive class weights
        if self.pos_weights is not None:
            weights = torch.where(
                y_true_clean > 0.5,
                self.pos_weights.unsqueeze(0),
                torch.ones_like(bce)
            )
            bce = bce * weights
        
        # Apply mask and compute mean
        masked_loss = bce * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        
        return loss
    
    @classmethod
    def from_imratio(cls, imratio_list: List[float]) -> 'WeightedMultiLabelBCE':
        """
        Create instance with pos_weights derived from imbalance ratios.
        
        Args:
            imratio_list: List of positive class ratios per label
        
        Returns:
            WeightedMultiLabelBCE: Configured loss
        """
        # Weight inversely proportional to class frequency
        pos_weights = [1.0 / (ir + 1e-6) for ir in imratio_list]
        # Normalize
        max_w = max(pos_weights)
        pos_weights = [w / max_w for w in pos_weights]
        
        return cls(num_classes=len(imratio_list), pos_weights=pos_weights)

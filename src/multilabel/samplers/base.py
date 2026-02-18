"""
Base Sampler Abstract Class

Defines the interface for all sampling approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for multi-label sampling approaches.
    
    All sampling strategies must implement:
    - fit_resample(): Apply the sampling strategy to a dataset
    - get_info(): Return sampling statistics and metadata
    """
    
    def __init__(self, **kwargs):
        """Initialize sampler with optional parameters."""
        self.config = kwargs
        self._info = {}
    
    @abstractmethod
    def fit_resample(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Apply the sampling strategy to the dataset.
        
        Args:
            dataset: The input dataset (must have get_label_matrix() method)
            **kwargs: Additional arguments (e.g., feature_extractor for some methods)
        
        Returns:
            Dataset: The transformed dataset (may be same object for in-training methods)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the sampling operation.
        
        Returns:
            dict: Sampling statistics including:
                - original_distribution: Label counts before sampling
                - final_distribution: Label counts after sampling
                - method: Sampling method name
                - parameters: Method-specific parameters used
        """
        pass
    
    def get_weighted_sampler(self, dataset: Dataset) -> Optional[WeightedRandomSampler]:
        """
        Get a WeightedRandomSampler for the dataset if applicable.
        Only relevant for Approach C (DistributionBalancedSampler).
        
        Args:
            dataset: Dataset to create sampler for
        
        Returns:
            WeightedRandomSampler or None if not applicable
        """
        return None
    
    def supports_online_augmentation(self) -> bool:
        """
        Check if this sampler applies augmentation during training.
        True for Approach B (ManifoldMixup), False for others.
        """
        return False
    
    def augment_batch(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply online augmentation to a batch (for samplers that support it).
        Only relevant for Approach B (ManifoldMixup).
        
        Args:
            features: Batch of feature vectors or images
            labels: Batch of label vectors
        
        Returns:
            tuple: (augmented_features, augmented_labels)
        """
        return features, labels
    
    def _compute_label_stats(self, label_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistics from a label matrix.
        
        Args:
            label_matrix: Shape (N, K) with binary labels
        
        Returns:
            dict: Label statistics
        """
        n_samples, n_labels = label_matrix.shape
        
        # Per-label counts
        pos_counts = (label_matrix == 1).sum(axis=0)
        neg_counts = (label_matrix == 0).sum(axis=0)
        
        # Imbalance ratios
        imratios = pos_counts / (pos_counts + neg_counts + 1e-10)
        
        # Multi-label statistics
        labels_per_sample = (label_matrix == 1).sum(axis=1)
        multi_label_samples = (labels_per_sample > 1).sum()
        
        return {
            'n_samples': n_samples,
            'n_labels': n_labels,
            'pos_counts': pos_counts.tolist(),
            'neg_counts': neg_counts.tolist(),
            'imratios': imratios.tolist(),
            'multi_label_samples': int(multi_label_samples),
            'avg_labels_per_sample': float(labels_per_sample.mean()),
        }

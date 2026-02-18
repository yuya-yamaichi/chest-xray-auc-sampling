"""
Approach C: Distribution-Balanced WeightedSampler

Implements weighted sampling based on label frequency and co-occurrence patterns.
Increases probability of sampling rare-label and multi-morbidity samples.
"""

from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import torch

from .base import BaseSampler


class DistributionBalancedSampler(BaseSampler):
    """
    Distribution-Balanced Weighted Random Sampler for multi-label data.
    
    Computes sample weights based on:
    - Label frequency (rare labels get higher weight)
    - Co-occurrence patterns (multi-morbidity samples get boosted)
    
    Uses PyTorch's WeightedRandomSampler for efficient batch sampling.
    
    Args:
        reweight_mode: Weight computation method
            - 'inverse': 1 / frequency
            - 'sqrt_inv': 1 / sqrt(frequency)
            - 'effective_num': (1 - beta) / (1 - beta^n)
        multi_label_boost: Weight multiplier for multi-disease samples
        smooth_eps: Smoothing factor to avoid division by zero
        replacement: Whether to sample with replacement (default: True)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        reweight_mode: str = 'sqrt_inv',
        multi_label_boost: float = 1.5,
        smooth_eps: float = 1e-6,
        replacement: bool = True,
        seed: int = 123,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.reweight_mode = reweight_mode
        self.multi_label_boost = multi_label_boost
        self.smooth_eps = smooth_eps
        self.replacement = replacement
        self.seed = seed
        
        self._sample_weights = None
        self._weighted_sampler = None
        
        self._info = {
            'method': 'distribution_balanced_weighted_sampler',
            'reweight_mode': reweight_mode,
            'multi_label_boost': multi_label_boost,
            'replacement': replacement,
        }
    
    def fit_resample(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Compute sample weights but don't modify the dataset.
        Weighted sampling is applied via get_weighted_sampler().
        
        Args:
            dataset: Dataset with get_label_matrix() method
        
        Returns:
            Dataset: The same dataset (unchanged)
        """
        if not hasattr(dataset, 'get_label_matrix'):
            raise ValueError("Dataset must have get_label_matrix() method")
        
        label_matrix = dataset.get_label_matrix()
        
        # Compute sample weights
        self._sample_weights = self._compute_sample_weights(label_matrix)
        
        # Create WeightedRandomSampler
        self._weighted_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(self._sample_weights).double(),
            num_samples=len(dataset),
            replacement=self.replacement,
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Store statistics
        self._info['original_distribution'] = self._compute_label_stats(label_matrix)
        self._info['final_distribution'] = self._info['original_distribution']
        self._info['weight_stats'] = {
            'min': float(self._sample_weights.min()),
            'max': float(self._sample_weights.max()),
            'mean': float(self._sample_weights.mean()),
            'std': float(self._sample_weights.std()),
        }
        
        return dataset
    
    def _compute_sample_weights(self, label_matrix: np.ndarray) -> np.ndarray:
        """
        Compute per-sample weights based on label distribution.
        
        Returns:
            np.ndarray: Shape (N,) with normalized sample weights
        """
        n_samples, n_labels = label_matrix.shape
        
        # Compute label frequencies (handle uncertain labels)
        valid_labels = label_matrix >= 0
        pos_counts = (label_matrix == 1).sum(axis=0)
        total_valid = valid_labels.sum(axis=0)
        
        label_freq = pos_counts / (total_valid + self.smooth_eps)
        
        # Compute per-label weights based on mode
        if self.reweight_mode == 'inverse':
            label_weights = 1.0 / (label_freq + self.smooth_eps)
        elif self.reweight_mode == 'sqrt_inv':
            label_weights = 1.0 / np.sqrt(label_freq + self.smooth_eps)
        elif self.reweight_mode == 'effective_num':
            beta = 0.999
            effective_num = 1.0 - np.power(beta, pos_counts)
            label_weights = (1.0 - beta) / (effective_num + self.smooth_eps)
        else:
            raise ValueError(f"Unknown reweight_mode: {self.reweight_mode}")
        
        # Normalize label weights
        label_weights = label_weights / (label_weights.sum() + self.smooth_eps)
        
        # Compute sample weights
        sample_weights = np.zeros(n_samples)
        
        for i in range(n_samples):
            pos_mask = label_matrix[i] == 1
            neg_mask = label_matrix[i] == 0
            
            if pos_mask.any():
                # Weight based on average of positive label weights
                # This gives equal importance to each rare label in the sample
                sample_weights[i] = label_weights[pos_mask].mean()
                
                # Boost multi-label samples
                n_positive = pos_mask.sum()
                if n_positive > 1:
                    # Exponential boost for multi-morbidity
                    boost = self.multi_label_boost ** (n_positive - 1)
                    sample_weights[i] *= boost
            else:
                # No positive labels: use minimum weight
                sample_weights[i] = label_weights.min() * 0.5
        
        # Normalize weights to sum to n_samples (expected by WeightedRandomSampler)
        sample_weights = sample_weights / (sample_weights.sum() + self.smooth_eps) * n_samples
        
        return sample_weights
    
    def get_weighted_sampler(self, dataset: Dataset) -> Optional[WeightedRandomSampler]:
        """
        Get the WeightedRandomSampler for use with DataLoader.
        
        Must call fit_resample() first.
        
        Returns:
            WeightedRandomSampler configured for this dataset
        """
        if self._weighted_sampler is None:
            # Compute weights if not already done
            self.fit_resample(dataset)
        
        return self._weighted_sampler
    
    def get_sample_weights(self) -> Optional[np.ndarray]:
        """Get the computed sample weights."""
        return self._sample_weights
    
    def get_info(self) -> Dict[str, Any]:
        """Return sampling info."""
        return self._info.copy()

"""
Approach 0: Baseline Sampler (No Sampling)

Standard multi-label learning without any sampling modification.
"""

from typing import Dict, Any
from torch.utils.data import Dataset
import numpy as np

from .base import BaseSampler


class BaselineSampler(BaseSampler):
    """
    Baseline approach: No sampling applied.
    
    Returns the original dataset unchanged.
    Serves as the control group for comparing sampling strategies.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._info = {
            'method': 'baseline',
            'sampling_applied': False,
        }
    
    def fit_resample(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Return the dataset unchanged.
        
        Args:
            dataset: The input dataset
        
        Returns:
            Dataset: The same dataset (unchanged)
        """
        # Get label statistics for info
        if hasattr(dataset, 'get_label_matrix'):
            label_matrix = dataset.get_label_matrix()
            stats = self._compute_label_stats(label_matrix)
            self._info['original_distribution'] = stats
            self._info['final_distribution'] = stats  # Same as original
        
        return dataset
    
    def get_info(self) -> Dict[str, Any]:
        """Return sampling info."""
        return self._info.copy()

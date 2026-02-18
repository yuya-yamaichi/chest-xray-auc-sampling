"""Sampling approaches for multi-label classification."""

from .base import BaseSampler
from .baseline import BaselineSampler
from .binary_relevance import DiseaseAdaptiveBRSampler
from .manifold_mixup import ManifoldMixupSampler
from .weighted_sampler import DistributionBalancedSampler
from .ml_smote import MLSMOTESampler


def get_sampler(approach: str, config: dict = None) -> BaseSampler:
    """
    Factory function to get the appropriate sampler based on approach name.
    
    Args:
        approach: One of '0', 'A', 'B', 'C', 'D' or full name
        config: Configuration dictionary for sampler parameters
    
    Returns:
        Configured sampler instance
    """
    config = config or {}
    
    approach_map = {
        '0': BaselineSampler,
        'baseline': BaselineSampler,
        'A': DiseaseAdaptiveBRSampler,
        'binary_relevance': DiseaseAdaptiveBRSampler,
        'B': ManifoldMixupSampler,
        'manifold_mixup': ManifoldMixupSampler,
        'C': DistributionBalancedSampler,
        'weighted_sampler': DistributionBalancedSampler,
        'D': MLSMOTESampler,
        'ml_smote': MLSMOTESampler,
    }
    
    sampler_class = approach_map.get(approach.upper() if len(approach) == 1 else approach.lower())
    
    if sampler_class is None:
        raise ValueError(f"Unknown approach: {approach}. Valid options: {list(approach_map.keys())}")
    
    # Pass config directly to sampler (train_multilabel.py already builds approach-specific config)
    return sampler_class(**config)


__all__ = [
    'BaseSampler',
    'BaselineSampler', 
    'DiseaseAdaptiveBRSampler',
    'ManifoldMixupSampler',
    'DistributionBalancedSampler',
    'MLSMOTESampler',
    'get_sampler',
]

"""
Approach A: Disease-Adaptive Binary Relevance Sampler

Treats each pathology as independent binary classification problem.
Applies optimal sampling method (SMOTE, ADASYN, etc.) per disease.

Optimal settings derived from Phase 2 single-label experiments:
- Cardiomegaly:     ADASYN, ratio=0.7 (ΔAUC=+0.0061)
- Edema:            RandomOverSampler, ratio=0.9 (ΔAUC=+0.0251, largest improvement)
- Consolidation:    RandomOverSampler, ratio=0.5 (ΔAUC=+0.0075)
- Atelectasis:      RandomOverSampler, ratio=0.7 (AUPRC improvement)
- Pleural Effusion: none (minimal improvement, ratio feasibility issues)
"""

from typing import Dict, Any, Optional, List, Callable, Union
from torch.utils.data import Dataset
import numpy as np
from collections import Counter

from .base import BaseSampler

# Imbalanced-learn imports
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# =====================================================================================
# OPTIMAL SETTINGS FROM PHASE 2 SINGLE-LABEL EXPERIMENTS
# Based on: docs/Phase2_論文用_結果まとめ.md
# =====================================================================================

# 病変ごとの最適サンプリング手法（Phase 2結果より）
OPTIMAL_METHODS_PHASE2 = {
    0: 'ADASYN',            # Cardiomegaly: ADASYN（比率0.7）が最良、ΔAUC=+0.0061
    1: 'RandomOverSampler', # Edema: RandomOverSampler（比率0.9）が最良、ΔAUC=+0.0251
    2: 'RandomOverSampler', # Consolidation: RandomOverSampler（比率0.5）が最良、ΔAUC=+0.0075
    3: 'RandomOverSampler', # Atelectasis: RandomOverSampler（比率0.7）が最良、ΔAUC=+0.0004
    4: 'SMOTE',             # Pleural Effusion: SMOTE（比率0.7）が最良、ΔAUC=+0.0004
}

# 病変ごとの最適サンプリング比率（Phase 2結果より）
# 比率 = 少数派クラス数 / 多数派クラス数
OPTIMAL_RATIOS_PHASE2 = {
    0: 0.7,   # Cardiomegaly
    1: 0.9,   # Edema
    2: 0.5,   # Consolidation
    3: 0.7,   # Atelectasis
    4: 0.7,   # Pleural Effusion (SMOTE)
}


class DiseaseAdaptiveBRSampler(BaseSampler):
    """
    Disease-Adaptive Binary Relevance Sampler.
    
    For each of the 5 pathologies:
    1. Extract binary labels for that disease
    2. Apply the specified sampling method with optimal ratio
    3. Combine results into a unified resampled dataset
    
    Default settings are derived from Phase 2 single-label experiments.
    
    Args:
        methods: Dict mapping disease index to sampling method name
                 Default: Use Phase 2 optimal settings
                 e.g., {0: 'ADASYN', 1: 'RandomOverSampler', ...}
        ratios: Dict mapping disease index to sampling ratio
                Default: Use Phase 2 optimal settings
                e.g., {0: 0.7, 1: 0.9, 2: 0.5, 3: 0.7, 4: None}
        k_neighbors: Number of neighbors for SMOTE/ADASYN (default: 5)
        seed: Random seed for reproducibility
        sampling_strategy: Global fallback ratio ('auto' or float). 
                          Overridden by per-disease ratios if specified.
        use_phase2_defaults: If True (default), use optimal settings from Phase 2
    """
    
    PATHOLOGIES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    SUPPORTED_METHODS = ['SMOTE', 'ADASYN', 'RandomOverSampler', 'none']
    
    def __init__(
        self,
        methods: Optional[Dict[int, str]] = None,
        ratios: Optional[Dict[int, Optional[float]]] = None,
        k_neighbors: int = 5,
        seed: int = 123,
        sampling_strategy: Union[str, float] = 'auto',
        use_phase2_defaults: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imbalanced-learn is required for DiseaseAdaptiveBRSampler")
        
        # Use Phase 2 optimal settings as defaults
        if use_phase2_defaults:
            self.methods = methods if methods is not None else OPTIMAL_METHODS_PHASE2.copy()
            self.ratios = ratios if ratios is not None else OPTIMAL_RATIOS_PHASE2.copy()
        else:
            # Fallback: SMOTE for all diseases with auto ratio
            self.methods = methods or {i: 'SMOTE' for i in range(5)}
            self.ratios = ratios or {i: None for i in range(5)}
        
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.use_phase2_defaults = use_phase2_defaults
        
        self._info = {
            'method': 'disease_adaptive_binary_relevance',
            'per_disease_methods': self.methods,
            'per_disease_ratios': self.ratios,
            'k_neighbors': k_neighbors,
            'sampling_strategy': sampling_strategy,
            'use_phase2_defaults': use_phase2_defaults,
        }
    
    def fit_resample(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Apply per-disease sampling and create a unified resampled dataset.
        
        The algorithm:
        1. For each disease, identify minority samples
        2. Apply the corresponding sampling method
        3. Track which samples are selected/duplicated for each disease
        4. Create a union of all selected indices
        
        Args:
            dataset: Dataset with get_label_matrix() method
        
        Returns:
            Dataset: Resampled dataset (subset with potential duplicates)
        """
        if not hasattr(dataset, 'get_label_matrix'):
            raise ValueError("Dataset must have get_label_matrix() method")
        
        label_matrix = dataset.get_label_matrix()
        n_samples, n_labels = label_matrix.shape
        
        # Store original stats
        self._info['original_distribution'] = self._compute_label_stats(label_matrix)
        
        # Create sample indices (we'll select/duplicate indices)
        all_indices = np.arange(n_samples)
        
        # Track selected indices with their multiplicities
        index_counts = Counter(all_indices)  # Start with all indices once
        
        per_disease_info = {}
        
        for disease_idx in range(n_labels):
            method_name = self.methods.get(disease_idx, 'SMOTE')
            disease_name = self.PATHOLOGIES[disease_idx]
            
            if method_name == 'none':
                per_disease_info[disease_name] = {'method': 'none', 'skipped': True}
                continue
            
            # Get binary labels for this disease
            y = label_matrix[:, disease_idx].astype(int)
            
            # Handle uncertain labels (-1)
            valid_mask = y >= 0
            if not valid_mask.all():
                # Only consider valid samples for this disease
                valid_indices = np.where(valid_mask)[0]
                y_valid = y[valid_mask]
                X_valid = valid_indices.reshape(-1, 1)
            else:
                X_valid = all_indices.reshape(-1, 1)
                y_valid = y
            
            # Check class distribution
            counter = Counter(y_valid)
            
            if len(counter) < 2:
                # Only one class, skip sampling
                per_disease_info[disease_name] = {
                    'method': method_name, 
                    'skipped': True, 
                    'reason': 'single_class'
                }
                continue
            
            minority_class = min(counter.keys(), key=lambda k: counter[k])
            minority_count = counter[minority_class]
            
            # Adjust k_neighbors based on minority class size
            k = min(self.k_neighbors, minority_count - 1)
            if k < 1:
                per_disease_info[disease_name] = {
                    'method': method_name,
                    'skipped': True,
                    'reason': 'insufficient_minority_samples'
                }
                continue
            
            try:
                # Get the sampling ratio for this disease
                disease_ratio = self.ratios.get(disease_idx)
                
                # Determine sampling strategy
                if disease_ratio is not None:
                    # Use per-disease ratio
                    majority_class = max(counter.keys(), key=lambda k: counter[k])
                    target_minority_count = int(disease_ratio * counter[majority_class])
                    # Ensure we have at least the current minority count
                    target_minority_count = max(target_minority_count, counter[minority_class])
                    sampling_strategy = {minority_class: target_minority_count}
                else:
                    # Fall back to global strategy
                    sampling_strategy = self.sampling_strategy
                
                # Initialize sampler with disease-specific ratio
                if method_name == 'SMOTE':
                    sampler = SMOTE(
                        random_state=self.seed,
                        k_neighbors=k,
                        sampling_strategy=sampling_strategy
                    )
                elif method_name == 'ADASYN':
                    sampler = ADASYN(
                        random_state=self.seed,
                        n_neighbors=k,
                        sampling_strategy=sampling_strategy
                    )
                elif method_name == 'RandomOverSampler':
                    sampler = RandomOverSampler(
                        random_state=self.seed,
                        sampling_strategy=sampling_strategy
                    )
                else:
                    raise ValueError(f"Unknown method: {method_name}")
                
                # Apply sampling
                X_resampled, y_resampled = sampler.fit_resample(X_valid, y_valid)
                
                # Track which indices were duplicated
                resampled_indices = X_resampled.flatten().astype(int)
                new_index_counts = Counter(resampled_indices)
                
                # Merge with existing counts (take max to avoid over-duplication)
                for idx, count in new_index_counts.items():
                    index_counts[idx] = max(index_counts[idx], count)
                
                final_counter = Counter(y_resampled)
                
                # Calculate achieved ratio
                min_class_final = min(final_counter.values())
                max_class_final = max(final_counter.values())
                achieved_ratio = min_class_final / max_class_final if max_class_final > 0 else None
                
                per_disease_info[disease_name] = {
                    'method': method_name,
                    'requested_ratio': disease_ratio,
                    'achieved_ratio': achieved_ratio,
                    'original_distribution': dict(counter),
                    'final_distribution': dict(final_counter),
                    'samples_added': len(X_resampled) - len(X_valid),
                }
                
            except Exception as e:
                per_disease_info[disease_name] = {
                    'method': method_name,
                    'skipped': True,
                    'error': str(e)
                }
        
        # Create final index array with duplicates
        final_indices = []
        for idx, count in sorted(index_counts.items()):
            final_indices.extend([idx] * count)
        final_indices = np.array(final_indices)
        
        # Store final stats
        self._info['per_disease_info'] = per_disease_info
        self._info['final_sample_count'] = len(final_indices)
        self._info['original_sample_count'] = n_samples
        
        # Get final label matrix
        final_labels = label_matrix[final_indices]
        self._info['final_distribution'] = self._compute_label_stats(final_labels)
        
        # Create subset dataset
        if hasattr(dataset, 'get_subset_by_indices'):
            return dataset.get_subset_by_indices(final_indices)
        else:
            # Fallback to torch Subset
            from torch.utils.data import Subset
            return Subset(dataset, final_indices.tolist())
    
    def get_info(self) -> Dict[str, Any]:
        """Return sampling info."""
        return self._info.copy()

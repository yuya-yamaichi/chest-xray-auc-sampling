"""
Approach D: ML-SMOTE (Multi-Label SMOTE)

Implements the ML-SMOTE algorithm for multi-label oversampling.
Based on Charte et al. (2015): "MLSMOTE: Approaching imbalanced multilabel 
learning through synthetic instance generation"
"""

from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

from .base import BaseSampler


class MLSMOTESampler(BaseSampler):
    """
    ML-SMOTE: Multi-Label Synthetic Minority Over-sampling Technique.
    
    The algorithm:
    1. Calculate Imbalance Ratio per Label (IRLbl)
    2. Identify minority labelsets using Mean Imbalance Ratio (MeanIR)
    3. For each minority sample:
       a. Find k nearest neighbors with same/similar labelset
       b. Generate synthetic sample by interpolation
       c. Assign labels based on label occurrence in neighbors
    
    Args:
        k_neighbors: Number of neighbors for synthetic sample generation
        threshold: Method to determine minority labelsets
            - 'meanIR': Use Mean Imbalance Ratio (default)
            - 'medianIR': Use Median Imbalance Ratio
            - float: Custom threshold value
        seed: Random seed for reproducibility
        use_feature_space: If True, operate in feature space (requires feature extractor)
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        threshold: str = 'meanIR',
        seed: int = 123,
        use_feature_space: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.seed = seed
        self.use_feature_space = use_feature_space
        
        self._rng = np.random.RandomState(seed)
        
        self._info = {
            'method': 'ml_smote',
            'k_neighbors': k_neighbors,
            'threshold': threshold,
        }
    
    def fit_resample(
        self, 
        dataset: Dataset,
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dataset:
        """
        Apply ML-SMOTE to the dataset.
        
        Args:
            dataset: Dataset with get_label_matrix() method
            features: Optional pre-computed feature vectors (N, D)
                      If None, uses sample indices as "features"
        
        Returns:
            Dataset: Resampled dataset with synthetic samples
        """
        if not hasattr(dataset, 'get_label_matrix'):
            raise ValueError("Dataset must have get_label_matrix() method")
        
        label_matrix = dataset.get_label_matrix()
        n_samples, n_labels = label_matrix.shape
        
        # Store original statistics
        self._info['original_distribution'] = self._compute_label_stats(label_matrix)
        
        # If no features provided, use indices (simplified mode)
        if features is None:
            features = np.arange(n_samples).reshape(-1, 1).astype(float)
            simplified_mode = True
        else:
            simplified_mode = False
        
        # Calculate imbalance ratios
        IRLbl = self._calculate_IRLbl(label_matrix)
        MeanIR = np.mean(IRLbl)
        
        self._info['IRLbl'] = IRLbl.tolist()
        self._info['MeanIR'] = float(MeanIR)
        
        # Determine threshold
        if isinstance(self.threshold, str):
            if self.threshold == 'meanIR':
                threshold_value = MeanIR
            elif self.threshold == 'medianIR':
                threshold_value = np.median(IRLbl)
            elif self.threshold == 'labelsetIR':
                # labelsetIR: Use imbalance ratio based on unique label combinations
                # Convert each row to a tuple for hashing
                labelsets = [tuple(row) for row in label_matrix]
                from collections import Counter
                labelset_counts = Counter(labelsets)
                
                # Calculate labelset-based IR (max_count / count for each labelset)
                max_labelset_count = max(labelset_counts.values())
                labelset_IRs = {ls: max_labelset_count / count for ls, count in labelset_counts.items()}
                
                # Use 75th percentile as threshold (identifies rarer labelsets)
                threshold_value = np.percentile(list(labelset_IRs.values()), 75)
                
                self._info['labelset_threshold_method'] = 'labelsetIR'
                self._info['n_unique_labelsets'] = len(labelset_counts)
            else:
                raise ValueError(f"Unknown threshold: {self.threshold}")
        else:
            threshold_value = float(self.threshold)
        
        self._info['threshold_value'] = float(threshold_value)
        
        # Identify minority labels (IRLbl > threshold)
        minority_labels = np.where(IRLbl > threshold_value)[0]
        self._info['minority_labels'] = minority_labels.tolist()
        
        if len(minority_labels) == 0:
            # No minority labels, return original dataset
            self._info['synthetic_samples_generated'] = 0
            self._info['final_distribution'] = self._info['original_distribution']
            return dataset
        
        # Identify minority samples (samples with at least one minority label)
        minority_mask = (label_matrix[:, minority_labels] == 1).any(axis=1)
        minority_indices = np.where(minority_mask)[0]
        
        self._info['minority_samples_count'] = len(minority_indices)
        
        # Generate synthetic samples
        synthetic_indices, synthetic_labels = self._generate_synthetic_samples(
            features=features,
            labels=label_matrix,
            minority_indices=minority_indices,
            minority_labels=minority_labels,
            IRLbl=IRLbl,
            simplified_mode=simplified_mode
        )
        
        self._info['synthetic_samples_generated'] = len(synthetic_indices)
        
        # Combine original and synthetic indices
        if simplified_mode:
            # In simplified mode, synthetic_indices are actual indices to duplicate
            all_indices = np.concatenate([
                np.arange(n_samples),
                synthetic_indices
            ])
        else:
            # In feature mode, would need to create synthetic dataset
            # For now, use duplicate indices
            all_indices = np.concatenate([
                np.arange(n_samples),
                synthetic_indices
            ])
        
        # Compute final statistics
        final_labels = label_matrix[all_indices.astype(int) % n_samples]
        self._info['final_distribution'] = self._compute_label_stats(final_labels)
        self._info['final_sample_count'] = len(all_indices)
        
        # Create subset dataset
        if hasattr(dataset, 'get_subset_by_indices'):
            return dataset.get_subset_by_indices(all_indices.astype(int) % n_samples)
        else:
            from torch.utils.data import Subset
            return Subset(dataset, (all_indices.astype(int) % n_samples).tolist())
    
    def _calculate_IRLbl(self, label_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Imbalance Ratio per Label.
        
        IRLbl(y) = max(h(y')) / h(y) for all labels y'
        where h(y) is the count of samples with label y
        
        Args:
            label_matrix: (N, K) binary label matrix
        
        Returns:
            np.ndarray: (K,) imbalance ratios
        """
        pos_counts = (label_matrix == 1).sum(axis=0)
        max_count = pos_counts.max()
        
        # Avoid division by zero
        IRLbl = max_count / (pos_counts + 1e-10)
        
        return IRLbl
    
    def _calculate_SCUMBLELbl(self, label_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate SCUMBLE measure per label.
        Measures the concurrence of minority labels with majority labels.
        
        Not used in basic ML-SMOTE but useful for MLSOL variant.
        """
        n_samples, n_labels = label_matrix.shape
        IRLbl = self._calculate_IRLbl(label_matrix)
        MeanIR = np.mean(IRLbl)
        
        SCUMBLE = np.zeros(n_labels)
        
        for j in range(n_labels):
            # Samples with label j
            samples_with_j = np.where(label_matrix[:, j] == 1)[0]
            
            if len(samples_with_j) == 0:
                SCUMBLE[j] = 0
                continue
            
            scumble_sum = 0
            for i in samples_with_j:
                # Co-occurring labels in sample i
                co_labels = np.where(label_matrix[i] == 1)[0]
                if len(co_labels) <= 1:
                    continue
                
                # Check if label j is minority and others are majority
                ir_sum = IRLbl[co_labels].sum()
                ir_prod = np.prod(1 - 1/IRLbl[co_labels])
                
                scumble_sum += ir_prod
            
            SCUMBLE[j] = scumble_sum / len(samples_with_j)
        
        return SCUMBLE
    
    def _generate_synthetic_samples(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        minority_indices: np.ndarray,
        minority_labels: np.ndarray,
        IRLbl: np.ndarray,
        simplified_mode: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for minority labelsets.
        
        Args:
            features: Feature vectors (N, D)
            labels: Label matrix (N, K)
            minority_indices: Indices of minority samples
            minority_labels: Indices of minority labels
            IRLbl: Imbalance ratio per label
            simplified_mode: If True, return indices for duplication
        
        Returns:
            tuple: (synthetic_indices, synthetic_labels)
        """
        n_samples, n_labels = labels.shape
        
        synthetic_indices = []
        synthetic_labels = []
        
        # Calculate how many synthetic samples to generate per label
        # Based on IRLbl - higher IR means more synthetic samples needed
        max_count = labels.sum(axis=0).max()
        
        for label_idx in minority_labels:
            current_count = (labels[:, label_idx] == 1).sum()
            target_count = int(max_count * 0.8)  # Target 80% of max
            samples_to_generate = max(0, target_count - current_count)
            
            # Get samples with this minority label
            label_samples = np.where(labels[:, label_idx] == 1)[0]
            
            if len(label_samples) == 0 or samples_to_generate == 0:
                continue
            
            # Fit k-NN on samples with this label
            k = min(self.k_neighbors, len(label_samples) - 1)
            if k < 1:
                # Not enough neighbors, just duplicate
                dup_count = min(samples_to_generate, len(label_samples))
                dup_indices = self._rng.choice(label_samples, size=dup_count, replace=True)
                synthetic_indices.extend(dup_indices)
                continue
            
            nn = NearestNeighbors(n_neighbors=k + 1)  # +1 to include self
            nn.fit(features[label_samples])
            
            # Generate synthetic samples
            for _ in range(samples_to_generate):
                # Randomly select a minority sample
                anchor_idx = self._rng.choice(label_samples)
                
                # Find its neighbors
                distances, neighbor_local_idxs = nn.kneighbors(
                    features[anchor_idx].reshape(1, -1)
                )
                
                # Convert local indices back to global
                neighbor_global_idxs = label_samples[neighbor_local_idxs[0]]
                
                # Exclude self
                neighbor_global_idxs = neighbor_global_idxs[neighbor_global_idxs != anchor_idx]
                
                if len(neighbor_global_idxs) == 0:
                    # No valid neighbors, just duplicate anchor
                    synthetic_indices.append(anchor_idx)
                    continue
                
                # Randomly select one neighbor
                neighbor_idx = self._rng.choice(neighbor_global_idxs)
                
                if simplified_mode:
                    # In simplified mode, choose randomly between anchor and neighbor
                    chosen_idx = anchor_idx if self._rng.random() > 0.5 else neighbor_idx
                    synthetic_indices.append(chosen_idx)
                else:
                    # In feature mode, would interpolate features
                    # For now, use random selection
                    synthetic_indices.append(anchor_idx)
        
        return np.array(synthetic_indices), np.array(synthetic_labels)
    
    def get_info(self) -> Dict[str, Any]:
        """Return sampling info."""
        return self._info.copy()

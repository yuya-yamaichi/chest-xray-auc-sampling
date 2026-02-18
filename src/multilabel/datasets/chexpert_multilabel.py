"""
CheXpert Multi-label Dataset

Custom PyTorch Dataset for multi-label classification of 5 chest pathologies.
Wraps the CheXpert dataset with additional features for sampling algorithms.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, List


class CheXpertMultiLabel(Dataset):
    """
    Multi-label Dataset for CheXpert chest X-ray images.
    
    Supports 5 pathologies: Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion
    
    Args:
        csv_path: Path to train.csv or valid.csv
        image_root_path: Root directory containing images
        image_size: Target image size (default: 224)
        mode: 'train' or 'valid'
        use_frontal: Whether to use only frontal views (default: True)
        uncertainty_policy: How to handle uncertain labels ('zeros', 'ones', 'ignore')
            - 'zeros': Map uncertain (-1) to 0
            - 'ones': Map uncertain (-1) to 1  
            - 'ignore': Keep as -1 for loss masking
        transform: Optional custom transforms (overrides default)
    """
    
    PATHOLOGIES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    
    def __init__(
        self,
        csv_path: str,
        image_root_path: str,
        image_size: int = 224,
        mode: str = 'train',
        use_frontal: bool = True,
        uncertainty_policy: str = 'zeros',
        transform: Optional[transforms.Compose] = None
    ):
        self.image_root_path = image_root_path
        self.image_size = image_size
        self.mode = mode
        self.uncertainty_policy = uncertainty_policy
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter frontal views if requested
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        # Fill NaN values with 0
        for col in self.PATHOLOGIES:
            self.df[col] = self.df[col].fillna(0)
        
        # Handle uncertainty based on policy
        if uncertainty_policy == 'zeros':
            for col in self.PATHOLOGIES:
                self.df[col] = self.df[col].replace(-1, 0)
        elif uncertainty_policy == 'ones':
            for col in self.PATHOLOGIES:
                self.df[col] = self.df[col].replace(-1, 1)
        # 'ignore' keeps -1 for masking during loss computation
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        # Set transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
        
        # Pre-compute label matrix for efficient sampling
        self._label_matrix = self.df[self.PATHOLOGIES].values.astype(np.float32)
        
        # Compute imbalance ratios
        self._compute_imbalance_stats()
        
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms matching LibAUC CheXpert preprocessing."""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _compute_imbalance_stats(self):
        """Compute imbalance statistics for each pathology."""
        self.imratio_list = []
        self.pos_counts = []
        self.neg_counts = []
        
        for i, col in enumerate(self.PATHOLOGIES):
            # Ignore -1 labels in counting
            valid_labels = self._label_matrix[:, i]
            if self.uncertainty_policy != 'ignore':
                pos_count = (valid_labels == 1).sum()
                neg_count = (valid_labels == 0).sum()
            else:
                pos_count = (valid_labels == 1).sum()
                neg_count = (valid_labels == 0).sum()
            
            total = pos_count + neg_count
            imratio = pos_count / total if total > 0 else 0
            
            self.imratio_list.append(imratio)
            self.pos_counts.append(pos_count)
            self.neg_counts.append(neg_count)
    
    def get_label_matrix(self) -> np.ndarray:
        """
        Return the full label matrix for sampling algorithms.
        
        Returns:
            np.ndarray: Shape (N, 5) with labels for all pathologies
        """
        return self._label_matrix.copy()
    
    def get_label_counts(self) -> dict:
        """
        Get label statistics for each pathology.
        
        Returns:
            dict: Statistics including counts and ratios
        """
        return {
            'pathologies': self.PATHOLOGIES,
            'imratio_list': self.imratio_list,
            'pos_counts': self.pos_counts,
            'neg_counts': self.neg_counts,
        }
    
    def get_sample_weights(self, method: str = 'inverse') -> np.ndarray:
        """
        Compute per-sample weights for WeightedRandomSampler.
        
        Args:
            method: 'inverse', 'sqrt_inv', or 'effective_num'
        
        Returns:
            np.ndarray: Shape (N,) with sample weights
        """
        weights = np.zeros(len(self))
        
        # Compute label frequency weights
        label_freq = self._label_matrix.sum(axis=0) / len(self)
        
        if method == 'inverse':
            label_weights = 1.0 / (label_freq + 1e-6)
        elif method == 'sqrt_inv':
            label_weights = 1.0 / np.sqrt(label_freq + 1e-6)
        elif method == 'effective_num':
            beta = 0.999
            effective_num = 1.0 - np.power(beta, self._label_matrix.sum(axis=0))
            label_weights = (1.0 - beta) / (effective_num + 1e-6)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize
        label_weights = label_weights / label_weights.sum()
        
        # Sample weight = max of label weights for positive labels
        for i in range(len(self)):
            pos_labels = self._label_matrix[i] == 1
            if pos_labels.any():
                weights[i] = label_weights[pos_labels].max()
            else:
                weights[i] = label_weights.min()  # No positive labels
        
        # Normalize weights
        weights = weights / weights.sum() * len(self)
        
        return weights
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            tuple: (image_tensor, labels_tensor)
                - image_tensor: Shape (3, H, W)
                - labels_tensor: Shape (5,) with values in {0, 1} or {-1, 0, 1}
        """
        row = self.df.iloc[idx]
        
        # Load image
        # CSV Path column may contain 'CheXpert-v1.0/train/...' or just 'train/...'
        # Handle both cases to avoid path duplication
        path_from_csv = row['Path']
        
        # If the CSV path starts with 'CheXpert-v1.0/', and root already contains it, strip it
        if 'CheXpert-v1.0' in self.image_root_path and path_from_csv.startswith('CheXpert-v1.0/'):
            path_from_csv = path_from_csv.replace('CheXpert-v1.0/', '', 1)
        
        img_path = os.path.join(self.image_root_path, path_from_csv)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor(self._label_matrix[idx], dtype=torch.float32)
        
        return image, labels
    
    def get_subset_by_indices(self, indices: np.ndarray) -> 'CheXpertMultiLabelSubset':
        """
        Create a subset of the dataset using provided indices.
        Used after sampling to create a new dataset with resampled indices.
        
        Args:
            indices: Array of sample indices (may contain duplicates)
        
        Returns:
            CheXpertMultiLabelSubset: Subset dataset
        """
        return CheXpertMultiLabelSubset(self, indices)


class CheXpertMultiLabelSubset(Dataset):
    """
    Subset of CheXpertMultiLabel dataset.
    Supports duplicate indices from oversampling.
    """
    
    def __init__(self, parent_dataset: CheXpertMultiLabel, indices: np.ndarray):
        self.parent = parent_dataset
        self.indices = indices.astype(int)
        
        # Update label matrix for the subset
        self._label_matrix = self.parent._label_matrix[self.indices]
        
    def get_label_matrix(self) -> np.ndarray:
        return self._label_matrix.copy()
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        parent_idx = self.indices[idx]
        return self.parent[parent_idx]


class SyntheticMultiLabelDataset(Dataset):
    """
    Dataset containing both original and synthetic samples.
    Used for samplers that generate synthetic feature vectors.
    """
    
    def __init__(
        self,
        original_dataset: CheXpertMultiLabel,
        synthetic_features: np.ndarray,
        synthetic_labels: np.ndarray,
        original_indices: np.ndarray
    ):
        """
        Args:
            original_dataset: The original CheXpertMultiLabel dataset
            synthetic_features: Synthetic feature vectors (N_syn, D)
            synthetic_labels: Labels for synthetic samples (N_syn, 5)
            original_indices: Indices from original dataset to include
        """
        self.original_dataset = original_dataset
        self.synthetic_features = torch.tensor(synthetic_features, dtype=torch.float32)
        self.synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.float32)
        self.original_indices = original_indices.astype(int)
        
        self.n_original = len(original_indices)
        self.n_synthetic = len(synthetic_features)
        
        # Combined label matrix
        original_labels = original_dataset.get_label_matrix()[original_indices]
        self._label_matrix = np.vstack([original_labels, synthetic_labels])
    
    def get_label_matrix(self) -> np.ndarray:
        return self._label_matrix.copy()
    
    def __len__(self) -> int:
        return self.n_original + self.n_synthetic
    
    def __getitem__(self, idx: int):
        """
        Returns images for original samples, features for synthetic samples.
        
        Note: When using synthetic samples, the model should be split into
        feature extractor and classifier, as synthetic samples are already features.
        """
        if idx < self.n_original:
            original_idx = self.original_indices[idx]
            return self.original_dataset[original_idx]
        else:
            syn_idx = idx - self.n_original
            return self.synthetic_features[syn_idx], self.synthetic_labels[syn_idx]
    
    def is_synthetic(self, idx: int) -> bool:
        """Check if the sample at idx is synthetic."""
        return idx >= self.n_original

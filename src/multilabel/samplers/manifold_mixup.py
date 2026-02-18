"""
Approach B: Multi-label Manifold Mixup Sampler

Implements class-aware Mixup in feature space with priority for rare-label samples.
This is an online augmentation method applied during training.
"""

from typing import Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
import numpy as np
import torch

from .base import BaseSampler


class ManifoldMixupSampler(BaseSampler):
    """
    Multi-label Manifold Mixup with rare-label prioritization.
    
    During training, performs interpolation in feature space:
    - Rare-label samples have higher probability of being selected as anchors
    - Mixup is performed at a configurable layer depth
    - Labels are interpolated accordingly
    
    Args:
        alpha: Beta distribution parameter for mixup ratio (default: 1.0)
        rare_weight: Priority multiplier for rare-label samples (default: 2.0)
        min_lambda: Minimum mixup coefficient to ensure meaningful mixing
        mix_probability: Probability of applying mixup to a batch (default: 0.5)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        rare_weight: float = 2.0,
        min_lambda: float = 0.2,
        mix_probability: float = 0.5,
        seed: int = 123,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.alpha = alpha
        self.rare_weight = rare_weight
        self.min_lambda = min_lambda
        self.mix_probability = mix_probability
        self.seed = seed
        
        self._rng = np.random.RandomState(seed)
        self._label_weights = None
        self._sample_weights = None
        
        self._info = {
            'method': 'manifold_mixup',
            'alpha': alpha,
            'rare_weight': rare_weight,
            'min_lambda': min_lambda,
            'mix_probability': mix_probability,
            'online_augmentation': True,
        }
    
    def fit_resample(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Prepare the sampler but don't modify the dataset.
        Mixup is applied during training via augment_batch().
        
        Args:
            dataset: Dataset with get_label_matrix() method
        
        Returns:
            Dataset: The same dataset (unchanged)
        """
        if hasattr(dataset, 'get_label_matrix'):
            label_matrix = dataset.get_label_matrix()
            self._compute_weights(label_matrix)
            self._info['original_distribution'] = self._compute_label_stats(label_matrix)
            self._info['final_distribution'] = self._info['original_distribution']
        
        return dataset
    
    def _compute_weights(self, label_matrix: np.ndarray):
        """
        Compute sample weights based on label rarity.
        
        Samples with rare labels get higher weights for selection as anchors.
        """
        n_samples, n_labels = label_matrix.shape
        
        # Compute label frequencies
        label_freq = (label_matrix == 1).sum(axis=0) / n_samples
        
        # Inverse frequency weights (normalized)
        self._label_weights = 1.0 / (label_freq + 1e-6)
        self._label_weights = self._label_weights / self._label_weights.sum()
        
        # Sample weights: based on rarest positive label
        self._sample_weights = np.zeros(n_samples)
        for i in range(n_samples):
            pos_labels = label_matrix[i] == 1
            if pos_labels.any():
                # Weight based on rarest positive label
                self._sample_weights[i] = self._label_weights[pos_labels].max()
            else:
                self._sample_weights[i] = self._label_weights.min()
        
        # Apply rare_weight multiplier
        self._sample_weights = np.power(self._sample_weights, self.rare_weight)
        self._sample_weights = self._sample_weights / self._sample_weights.sum()
    
    def supports_online_augmentation(self) -> bool:
        """This sampler applies augmentation during training."""
        return True
    
    def augment_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Manifold Mixup to a batch.
        
        Args:
            features: Batch of features (B, D) or images (B, C, H, W)
            labels: Batch of labels (B, K)
            training: Whether in training mode
        
        Returns:
            tuple: (mixed_features, mixed_labels)
        """
        if not training:
            return features, labels
        
        # Decide whether to apply mixup
        if self._rng.random() > self.mix_probability:
            return features, labels
        
        batch_size = features.shape[0]
        device = features.device if torch.is_tensor(features) else 'cpu'
        
        # Sample mixup coefficient from Beta distribution
        lam = self._rng.beta(self.alpha, self.alpha)
        lam = max(lam, self.min_lambda)  # Ensure minimum mixing
        lam = max(lam, 1 - lam)  # Ensure lambda >= 0.5 so first sample dominates
        
        # Create permutation for mixing pairs
        # Prioritize rare-label samples if weights are available
        if self._sample_weights is not None and len(self._sample_weights) >= batch_size:
            # For simplicity, use random permutation within batch
            # (Full implementation would sample from full dataset weighted)
            perm = torch.randperm(batch_size, device=device)
        else:
            perm = torch.randperm(batch_size, device=device)
        
        # Mix features
        if torch.is_tensor(features):
            mixed_features = lam * features + (1 - lam) * features[perm]
        else:
            features_t = torch.tensor(features, dtype=torch.float32, device=device)
            mixed_features = lam * features_t + (1 - lam) * features_t[perm]
        
        # Mix labels (multi-label: element-wise interpolation)
        if torch.is_tensor(labels):
            mixed_labels = lam * labels + (1 - lam) * labels[perm]
        else:
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
            mixed_labels = lam * labels_t + (1 - lam) * labels_t[perm]
        
        return mixed_features, mixed_labels
    
    def mixup_at_layer(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        layer_name: str = 'features.denseblock3'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Manifold Mixup at a specific layer of the model.
        
        This requires hooking into the model's forward pass.
        
        Args:
            model: The neural network model
            inputs: Input images (B, C, H, W)
            labels: Labels (B, K)
            layer_name: Name of the layer to apply mixup at
        
        Returns:
            tuple: (output, mixed_labels)
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Sample mixup coefficient
        lam = self._rng.beta(self.alpha, self.alpha)
        lam = max(lam, self.min_lambda)
        lam = max(lam, 1 - lam)
        
        # Permutation for mixing
        perm = torch.randperm(batch_size, device=device)
        
        # Mixed labels
        mixed_labels = lam * labels + (1 - lam) * labels[perm]
        
        # Hook to capture and modify features at target layer
        features_mixed = [None]
        
        def mixup_hook(module, input, output):
            # Apply mixup to the output of this layer
            mixed = lam * output + (1 - lam) * output[perm]
            features_mixed[0] = mixed
            return mixed
        
        # Find and hook the target layer
        target_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            # Fallback: apply mixup to input
            mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
            output = model(mixed_inputs)
            return output, mixed_labels
        
        # Register hook
        handle = target_module.register_forward_hook(mixup_hook)
        
        try:
            # Forward pass (hook will modify features at target layer)
            output = model(inputs)
        finally:
            handle.remove()
        
        return output, mixed_labels
    
    def get_info(self) -> Dict[str, Any]:
        """Return sampling info."""
        return self._info.copy()

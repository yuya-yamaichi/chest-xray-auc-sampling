"""
CXR8 Multi-label Dataset for Evaluation

Custom PyTorch Dataset for multi-label classification evaluation on NIH CXR8 dataset.
Maps CXR8 findings to the 5 target pathologies used in training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict


class CXR8MultiLabel(Dataset):
    """
    Multi-label Dataset for CXR8 (NIH ChestX-ray8) evaluation.
    
    Maps CXR8 findings to 5 target pathologies:
    - Cardiomegaly -> Cardiomegaly
    - Edema -> Edema
    - Consolidation -> Consolidation
    - Atelectasis -> Atelectasis
    - Pleural Effusion -> Effusion
    
    Args:
        csv_path: Path to test CSV file
        image_root_path: Root directory containing images
        image_size: Target image size (default: 224)
        transform: Optional custom transforms
    """
    
    # Target pathologies (matching training)
    TARGET_PATHOLOGIES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    
    # CXR8 label mapping to target pathologies
    CXR8_TO_TARGET = {
        'Cardiomegaly': 'Cardiomegaly',
        'Edema': 'Edema',
        'Consolidation': 'Consolidation',
        'Atelectasis': 'Atelectasis',
        'Effusion': 'Pleural Effusion',
    }
    
    def __init__(
        self,
        csv_path: str,
        image_root_path: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_root_path = image_root_path
        self.image_size = image_size
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Identify image column
        if 'Image Index' in self.df.columns:
            self.image_col = 'Image Index'
        elif 'image_id' in self.df.columns:
            self.image_col = 'image_id'
        else:
            # Try first column
            self.image_col = self.df.columns[0]
        
        # Identify finding labels column
        if 'Finding Labels' in self.df.columns:
            self.finding_col = 'Finding Labels'
        elif 'labels' in self.df.columns:
            self.finding_col = 'labels'
        else:
            self.finding_col = None
        
        # Parse labels
        self._parse_labels()
        
        # Set transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
    
    def _get_default_transforms(self) -> transforms.Compose:
        """Get default transforms for evaluation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _parse_labels(self):
        """Parse finding labels and create multi-label matrix."""
        n_samples = len(self.df)
        n_labels = len(self.TARGET_PATHOLOGIES)
        
        self._label_matrix = np.zeros((n_samples, n_labels), dtype=np.float32)
        
        if self.finding_col is None:
            # Check if individual pathology columns exist
            for i, pathology in enumerate(self.TARGET_PATHOLOGIES):
                if pathology in self.df.columns:
                    self._label_matrix[:, i] = self.df[pathology].values
                # Also check CXR8 naming (e.g., 'Effusion' instead of 'Pleural Effusion')
                for cxr8_name, target_name in self.CXR8_TO_TARGET.items():
                    if target_name == pathology and cxr8_name in self.df.columns:
                        self._label_matrix[:, i] = self.df[cxr8_name].values
        else:
            # Parse from 'Finding Labels' column
            for idx, row in self.df.iterrows():
                finding_str = str(row[self.finding_col])
                findings = [f.strip() for f in finding_str.split('|')]
                
                for finding in findings:
                    if finding in self.CXR8_TO_TARGET:
                        target_pathology = self.CXR8_TO_TARGET[finding]
                        target_idx = self.TARGET_PATHOLOGIES.index(target_pathology)
                        self._label_matrix[idx, target_idx] = 1.0
        
        # Compute statistics
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute label statistics."""
        self.pos_counts = self._label_matrix.sum(axis=0).astype(int)
        self.total_samples = len(self)
        self.label_prevalence = self.pos_counts / self.total_samples
    
    def get_label_matrix(self) -> np.ndarray:
        """Return the full label matrix."""
        return self._label_matrix.copy()
    
    def get_label_stats(self) -> Dict:
        """Get label statistics."""
        return {
            'pathologies': self.TARGET_PATHOLOGIES,
            'pos_counts': self.pos_counts.tolist(),
            'total_samples': self.total_samples,
            'prevalence': self.label_prevalence.tolist(),
        }
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            tuple: (image_tensor, labels_tensor)
                - image_tensor: Shape (3, H, W)
                - labels_tensor: Shape (5,)
        """
        row = self.df.iloc[idx]
        
        # Get image filename
        img_name = row[self.image_col]
        
        # Handle different directory structures
        if os.path.exists(os.path.join(self.image_root_path, img_name)):
            img_path = os.path.join(self.image_root_path, img_name)
        elif os.path.exists(os.path.join(self.image_root_path, 'images', img_name)):
            img_path = os.path.join(self.image_root_path, 'images', img_name)
        else:
            # Try to find in subdirectories
            for subdir in os.listdir(self.image_root_path):
                potential_path = os.path.join(self.image_root_path, subdir, img_name)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            else:
                raise FileNotFoundError(f"Image not found: {img_name} in {self.image_root_path}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor(self._label_matrix[idx], dtype=torch.float32)
        
        return image, labels

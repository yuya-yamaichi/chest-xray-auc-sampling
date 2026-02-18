"""
Configuration Management Utilities
Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization

This module provides utilities for loading and managing configuration from YAML files.
"""

import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration manager for loading and accessing YAML config files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"✓ Configuration loaded from: {self.config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key (str): Configuration key using dot notation (e.g., 'dataset.root')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section (str): Section name
            
        Returns:
            Dictionary containing the section configuration
        """
        return self.config.get(section, {})
    
    def print_config(self, section: str = None):
        """
        Print configuration in a formatted manner.
        
        Args:
            section (str, optional): Print only specific section
        """
        if section:
            config_to_print = {section: self.config.get(section, {})}
        else:
            config_to_print = self.config
        
        print("=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        self._print_dict(config_to_print)
        print("=" * 80)
    
    def _print_dict(self, d: Dict[str, Any], indent: int = 0):
        """Recursively print dictionary with proper indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Config: Configuration manager instance
    """
    return Config(config_path)


def get_device_config(config: Config) -> str:
    """
    Get device configuration (cuda/cpu).
    
    Args:
        config (Config): Configuration manager
        
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    import torch
    
    use_cuda = config.get('device.use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def set_reproducibility(config: Config):
    """
    Set reproducibility settings based on configuration.
    
    Args:
        config (Config): Configuration manager
    """
    import torch
    import numpy as np
    
    seed = config.get('reproducibility.seed', 123)
    deterministic = config.get('reproducibility.deterministic', True)
    benchmark = config.get('reproducibility.benchmark', False)
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set CUDA settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if not benchmark:
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Reproducibility settings applied (seed={seed})")


def get_pathology_name(config: Config, class_id: int) -> str:
    """
    Get pathology name from class ID.
    
    Args:
        config (Config): Configuration manager
        class_id (int): Class ID
        
    Returns:
        str: Pathology name
    """
    pathologies = config.get_section('pathologies')
    # Try both int key and string key for compatibility
    return pathologies.get(class_id, pathologies.get(str(class_id), f"Unknown_{class_id}"))
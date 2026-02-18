"""
Pre-training Script for DenseNet121 on CheXpert Dataset
Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization

This script pre-trains a DenseNet121 model on the CheXpert dataset using multi-label 
classification for 5 pathologies. The trained model serves as the foundation for all 
subsequent experiments with sampling techniques and AUC optimization.

REVISED VERSION: Compatible with LibAUC 1.2.0
"""

import torch
import numpy as np
import os
import time
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score

# LibAUC imports
from libauc.losses import CrossEntropyLoss
from libauc.optimizers import Adam
from libauc.models import densenet121  # Use lowercase for consistency
from libauc.datasets import CheXpert

# Configuration management
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import load_config, get_device_config, set_reproducibility

# =====================================================================================
# GLOBAL CONFIGURATION - Loaded from YAML
# =====================================================================================

# Load configuration from YAML file
CONFIG = load_config()

# =====================================================================================
# UTILITY FUNCTIONS
# =====================================================================================

def print_config():
    """Print current configuration settings."""
    device = get_device_config(CONFIG)
    seed = CONFIG.get('reproducibility.seed', 123)
    epochs = CONFIG.get('pretrain.epochs', 1)
    batch_size = CONFIG.get('pretrain.batch_size', 32)
    learning_rate = CONFIG.get('pretrain.learning_rate', 1e-4)
    weight_decay = CONFIG.get('pretrain.weight_decay', 1e-5)
    dataset_root = CONFIG.get('dataset.root', '/home/yamaichi/CheXpert/CheXpert-v1.0/')
    num_classes = CONFIG.get('model.pretrain_num_classes', 5)
    model_save_path = CONFIG.get('model.pretrained_model_path', 'models/pretrained/best_pretrained_model.pth')
    
    print("=" * 80)
    print("PRE-TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Dataset Root: {dataset_root}")
    print(f"Number of Classes: {num_classes}")
    print(f"Pretrained: True")
    print(f"Model Save Path: {model_save_path}")
    print("=" * 80)

def log_pretraining_info(best_val_auc, training_time):
    """Log pre-training information to CSV."""
    results_dir = CONFIG.get('paths.results_dir', 'results')
    pretraining_log_file = CONFIG.get('paths.pretraining_log_file', 'pretraining_log.csv')
    
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, pretraining_log_file)
    
    file_exists = os.path.exists(log_path)
    
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': CONFIG.get('reproducibility.seed', 123),
        'best_val_auc': best_val_auc,
        'training_time_sec': training_time,
        'model_path': CONFIG.get('model.pretrained_model_path', 'models/pretrained/best_pretrained_model.pth')
    }
    
    with open(log_path, 'a', newline='') as csvfile:
        fieldnames = row_data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    print(f"✓ Results logged to: {log_path}")
    return log_path

# =====================================================================================
# MAIN TRAINING FUNCTION
# =====================================================================================

def main():
    """Main training function - using YAML configuration."""
    
    # Start timing
    total_start_time = time.time()
    
    # Set reproducibility
    set_reproducibility(CONFIG)
    
    # Get configuration values
    device = get_device_config(CONFIG)
    dataset_root = CONFIG.get('dataset.root', '/home/yamaichi/CheXpert/CheXpert-v1.0/')
    model_save_path = CONFIG.get('model.pretrained_model_path', 'models/pretrained/best_pretrained_model.pth')
    
    # Print configuration
    print_config()
    
    # Verify dataset path exists
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found at {dataset_root}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # ---------------------------------------------------------------------------------
    # Data Loading - Using configuration
    # ---------------------------------------------------------------------------------
    print("\nLoading CheXpert datasets...")
    
    # Get dataset configuration
    image_size = CONFIG.get('dataset.image_size', 224)
    use_frontal_only = CONFIG.get('dataset.use_frontal_only', True)
    batch_size = CONFIG.get('pretrain.batch_size', 32)
    num_workers = CONFIG.get('pretrain.num_workers', 2)
    
    # Load datasets using configuration
    trainSet = CheXpert(
        csv_path=os.path.join(dataset_root, 'train.csv'),
        image_root_path=dataset_root,
        use_upsampling=False,  # Original uses False for pretraining
        use_frontal=use_frontal_only,
        image_size=image_size,
        mode='train',
        class_index=-1  # Multi-label mode
    )
    
    testSet = CheXpert(
        csv_path=os.path.join(dataset_root, 'valid.csv'),
        image_root_path=dataset_root,
        use_upsampling=False,
        use_frontal=use_frontal_only,
        image_size=image_size,
        mode='valid',
        class_index=-1  # Multi-label mode
    )
    
    # Create data loaders using configuration
    trainloader = torch.utils.data.DataLoader(
        trainSet, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True
    )
    
    testloader = torch.utils.data.DataLoader(
        testSet, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    
    print(f"✓ Training samples: {len(trainSet)}")
    print(f"✓ Validation samples: {len(testSet)}")
    print(f"✓ Training batches: {len(trainloader)}")
    print(f"✓ Validation batches: {len(testloader)}")
    
    # ---------------------------------------------------------------------------------
    # Model Setup - Using configuration
    # ---------------------------------------------------------------------------------
    print("\nInitializing DenseNet121 model...")
    
    # Get model configuration
    num_classes = CONFIG.get('model.pretrain_num_classes', 5)
    activation = CONFIG.get('model.activation', 'relu')
    
    # Use densenet121 (lowercase) for consistency
    model = densenet121(
        pretrained=True,
        last_activation=None,  # None for multi-label
        activations=activation,
        num_classes=num_classes
    )
    model = model.to(device)
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ---------------------------------------------------------------------------------
    # Loss Function and Optimizer - Using configuration
    # ---------------------------------------------------------------------------------
    learning_rate = float(CONFIG.get('pretrain.learning_rate', 1e-4))
    weight_decay = float(CONFIG.get('pretrain.weight_decay', 1e-5))
    
    CELoss = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print(f"✓ Using CrossEntropyLoss and Adam optimizer")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    
    # ---------------------------------------------------------------------------------
    # Training Loop - Exact match to original
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STARTING PRE-TRAINING")
    print("=" * 80)
    
    # Get training configuration
    epochs = CONFIG.get('pretrain.epochs', 1)
    validation_interval = CONFIG.get('pretrain.validation_interval', 400)
    
    training_start_time = time.time()
    best_val_auc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            train_data, train_labels = train_data.to(device), train_labels.to(device)
            
            y_pred = model(train_data)
            loss = CELoss(y_pred, train_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation - using configuration
            if idx % validation_interval == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = []
                    test_true = []
                    for jdx, data in enumerate(testloader):
                        test_data, test_labels = data
                        test_data = test_data.to(device)
                        y_pred = model(test_data)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())
                    
                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc_mean = roc_auc_score(test_true, test_pred)
                    model.train()
                    
                    if best_val_auc < val_auc_mean:
                        best_val_auc = val_auc_mean
                        torch.save(model.state_dict(), model_save_path)
                        print(f"  🎉 New best model saved: Val_AUC={val_auc_mean:.4f}")
                    
                    print(f"  Epoch={epoch}, BatchID={idx:4d}, Val_AUC={val_auc_mean:.4f}, Best_Val_AUC={best_val_auc:.4f}")
    
    # Calculate execution times
    training_time = time.time() - training_start_time
    total_time = time.time() - total_start_time
    
    # ---------------------------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRE-TRAINING COMPLETED")
    print("=" * 80)
    print(f"✓ Best Validation AUC: {best_val_auc:.4f}")
    print(f"✓ Training time: {training_time:.2f} seconds")
    print(f"✓ Total time: {total_time:.2f} seconds")
    print(f"✓ Model saved to: {model_save_path}")
    print("=" * 80)
    
    # Log results
    log_pretraining_info(best_val_auc, training_time)
    
    return best_val_auc

if __name__ == "__main__":
    try:
        best_auc = main()
        print(f"\nFinal result: Best AUC = {best_auc:.4f}")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise
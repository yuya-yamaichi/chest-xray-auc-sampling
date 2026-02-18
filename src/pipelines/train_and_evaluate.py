#!/usr/bin/env python3
"""
Integrated Training and Evaluation Script for Master's Thesis Research
Synergistic Effects of Sampling Methods and AUC Optimization

This script combines the complete pipeline: pretrain → finetune → evaluate
to provide a seamless training and evaluation workflow with proper model naming.

Usage:
    python src/pipelines/train_and_evaluate.py --class_id 0 --sampler SMOTE --seed 123
    python src/pipelines/train_and_evaluate.py --class_id 0 --sampler none --seed 123

Author: Master's Thesis Research
Date: August 2025
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import subprocess
from datetime import datetime

# Configuration management
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import load_config, get_device_config, set_reproducibility, get_pathology_name

# =====================================================================================
# GLOBAL CONFIGURATION
# =====================================================================================

# Load configuration from YAML file
CONFIG = load_config()

# =====================================================================================
# COMMAND LINE ARGUMENT PARSING
# =====================================================================================

def parse_arguments():
    """Parse command line arguments for integrated training and evaluation."""
    parser = argparse.ArgumentParser(
        description='Integrated training and evaluation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with SMOTE sampling and evaluate
    python src/pipelines/train_and_evaluate.py --class_id 0 --sampler SMOTE --seed 123
    
    # Train baseline (no sampling) and evaluate
    python src/pipelines/train_and_evaluate.py --class_id 0 --sampler none --seed 123
    
    # Train with custom sampling ratio
    python src/pipelines/train_and_evaluate.py --class_id 0 --sampler SMOTE --sampling_ratio 0.5 --seed 123
        """
    )
    
    parser.add_argument('--class_id', type=int, required=True, 
                       choices=[0, 1, 2, 3, 4],
                       help='Target pathology ID (0=Cardiomegaly, 1=Edema, 2=Consolidation, 3=Atelectasis, 4=Pleural Effusion)')
    
    parser.add_argument('--sampler', type=str, default='none',
                       choices=CONFIG.get('sampling_methods', ['none']),
                       help='Sampling method to apply. Available: none, RandomOverSampler, SMOTE, ADASYN, etc.')
    
    parser.add_argument('--sampling_ratio', type=str, default=None,
                       help='Custom sampling ratio. Options: "auto" (balanced), "0.5" (50%% minority/majority), "1:3" (1:3 ratio), etc.')
    
    parser.add_argument('--seed', type=int, default=CONFIG.get('reproducibility.seed', 123),
                       help='Random seed for reproducibility')
    
    parser.add_argument('--skip_pretrain', action='store_true',
                       help='Skip pre-training phase (use existing pretrained model)')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    return parser.parse_args()

# =====================================================================================
# MODEL NAMING UTILITIES
# =====================================================================================

def generate_model_name(class_id, sampler, seed, auc=None):
    """
    Generate descriptive model name based on configuration.
    
    Args:
        class_id (int): Target pathology ID
        sampler (str): Sampling method name
        seed (int): Random seed
        auc (float, optional): AUC score for naming
    
    Returns:
        str: Formatted model name
    """
    pathology_name = get_pathology_name(CONFIG, class_id)
    pathology_clean = pathology_name.lower().replace(' ', '_')
    
    if auc is not None:
        return f"{pathology_clean}_{sampler}_seed{seed}_auc{auc:.4f}"
    else:
        return f"{pathology_clean}_{sampler}_seed{seed}"

def get_model_path(class_id, sampler, seed, auc=None):
    """
    Generate full model path based on naming convention.
    
    Args:
        class_id (int): Target pathology ID
        sampler (str): Sampling method name
        seed (int): Random seed
        auc (float, optional): AUC score for path generation
        
    Returns:
        str: Full path to model file
    """
    models_dir = CONFIG.get('paths.models_dir', 'models')
    pathology_name = get_pathology_name(CONFIG, class_id)
    pathology_dir = os.path.join(models_dir, pathology_name.lower())
    
    model_name = generate_model_name(class_id, sampler, seed, auc)
    model_filename = f"finetuned_{model_name}.pth"
    
    return os.path.join(pathology_dir, model_filename)

# =====================================================================================
# PIPELINE EXECUTION FUNCTIONS
# =====================================================================================

def run_pretraining(args):
    """
    Execute pre-training phase if needed.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        bool: True if pre-training completed successfully
    """
    print("\n" + "=" * 80)
    print("PHASE 1: PRE-TRAINING")
    print("=" * 80)
    
    # Check if pretrained model already exists
    pretrained_path = CONFIG.get('model.pretrained_model_path', 'models/pretrained/best_pretrained_model.pth')
    
    if args.skip_pretrain or os.path.exists(pretrained_path):
        if os.path.exists(pretrained_path):
            print(f"✓ Using existing pretrained model: {pretrained_path}")
            return True
        else:
            print(f"⚠️  Warning: --skip_pretrain specified but model not found at {pretrained_path}")
            print("  Proceeding with pre-training...")
    
    print("Starting multi-label pre-training on 5 pathologies...")
    
    try:
        # Import and run pretrain module (support script execution)
        from pipelines import pretrain
        
        # Set seed for reproducibility
        set_reproducibility(CONFIG)
        if args.seed != CONFIG.get('reproducibility.seed', 123):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
        
        # Run pre-training
        best_auc = pretrain.main()
        
        print(f"✓ Pre-training completed successfully with AUC: {best_auc:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Pre-training failed: {str(e)}")
        raise

def run_finetuning(args):
    """
    Execute fine-tuning phase with specified sampling method.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        tuple: (best_val_auc, model_path) from fine-tuning
    """
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNING")
    print("=" * 80)
    
    pathology_name = get_pathology_name(CONFIG, args.class_id)
    print(f"Fine-tuning for {pathology_name} (class_id={args.class_id})")
    print(f"Sampling method: {args.sampler}")
    print(f"Random seed: {args.seed}")
    
    try:
        # Import and run finetune module (support script execution)
        from pipelines import finetune
        
        # Set up arguments for finetune module
        import types
        finetune_args = types.SimpleNamespace(
            class_id=args.class_id,
            sampler=args.sampler,
            sampling_ratio=args.sampling_ratio,
            seed=args.seed,
            config=args.config
        )
        
        # Temporarily replace sys.argv to simulate command line call
        old_argv = sys.argv
        sys.argv = [
            'finetune.py',
            '--class_id', str(args.class_id),
            '--sampler', args.sampler,
            '--seed', str(args.seed)
        ]
        
        if args.sampling_ratio:
            sys.argv.extend(['--sampling_ratio', args.sampling_ratio])
        
        try:
            # Set seed for reproducibility
            set_reproducibility(CONFIG)
            if args.seed != CONFIG.get('reproducibility.seed', 123):
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
            
            # Run fine-tuning by calling main with our args
            best_auc = finetune.main()
            
            # Generate model path based on results
            model_path = get_model_path(args.class_id, args.sampler, args.seed, best_auc)
            
            print(f"✓ Fine-tuning completed successfully with AUC: {best_auc:.4f}")
            print(f"✓ Model should be saved to: {model_path}")
            
            return best_auc, model_path
            
        finally:
            # Restore original sys.argv
            sys.argv = old_argv
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {str(e)}")
        raise

def run_evaluation(args, model_path, best_auc):
    """
    Execute evaluation phase on CXR8 test dataset.
    
    Args:
        args: Parsed command line arguments
        model_path (str): Path to the trained model
        best_auc (float): Best validation AUC from training
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("PHASE 3: FINAL EVALUATION")
    print("=" * 80)
    
    # Generate descriptive model name for evaluation
    model_name = generate_model_name(args.class_id, args.sampler, args.seed, best_auc)
    
    print(f"Evaluating model: {model_name}")
    print(f"Model path: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        # Try to find the model with a pattern match
        models_dir = CONFIG.get('paths.models_dir', 'models')
        pathology_name = get_pathology_name(CONFIG, args.class_id)
        pathology_dir = os.path.join(models_dir, pathology_name.lower())
        
        print(f"⚠️  Model not found at {model_path}")
        print(f"  Searching in {pathology_dir}...")
        
        if os.path.exists(pathology_dir):
            # Look for models with matching pattern
            import glob
            pattern = f"*{args.sampler}*seed{args.seed}*.pth"
            matching_models = glob.glob(os.path.join(pathology_dir, pattern))
            
            if matching_models:
                # Use the most recent model
                model_path = max(matching_models, key=os.path.getmtime)
                print(f"  Found matching model: {model_path}")
            else:
                raise FileNotFoundError(f"No model found matching pattern: {pattern} in {pathology_dir}")
        else:
            raise FileNotFoundError(f"Model directory not found: {pathology_dir}")
    
    try:
        # Import and run evaluation module (support script execution)
        from pipelines import evaluate
        
        # Temporarily replace sys.argv to simulate command line call
        old_argv = sys.argv
        sys.argv = [
            'evaluate.py',
            '--model_path', model_path,
            '--model_name', model_name
        ]
        
        try:
            # Run evaluation
            evaluate.main()
            
            print(f"✓ Evaluation completed successfully")
            print(f"✓ Results saved with model name: {model_name}")
            
            return model_name
            
        finally:
            # Restore original sys.argv
            sys.argv = old_argv
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        raise

# =====================================================================================
# MAIN EXECUTION FUNCTION
# =====================================================================================

def main():
    """
    Main execution function that orchestrates the complete training and evaluation pipeline.
    """
    # Parse command line arguments first
    args = parse_arguments()
    
    print("🔬 INTEGRATED TRAINING AND EVALUATION PIPELINE")
    print("=" * 70)
    print("Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization")
    print("Pipeline: Pretrain → Finetune → Evaluate")
    print("=" * 70)
    
    # Start total timing
    total_start_time = time.time()
    
    # Display configuration
    device = get_device_config(CONFIG)
    pathology_name = get_pathology_name(CONFIG, args.class_id)
    
    print(f"Target Pathology: {pathology_name} (class_id={args.class_id})")
    print(f"Sampling Method: {args.sampler}")
    print(f"Random Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Skip Pre-training: {args.skip_pretrain}")
    print(f"Config File: {args.config}")
    
    if args.sampling_ratio:
        print(f"Sampling Ratio: {args.sampling_ratio}")
    
    print("=" * 70)
    
    try:
        # Phase 1: Pre-training (if needed)
        if not run_pretraining(args):
            raise RuntimeError("Pre-training phase failed")
        
        # Phase 2: Fine-tuning
        best_auc, model_path = run_finetuning(args)
        
        # Phase 3: Evaluation
        model_name = run_evaluation(args, model_path, best_auc)
        
        # Calculate total execution time
        total_time = time.time() - total_start_time
        
        # Final Summary
        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Pathology: {pathology_name}")
        print(f"✅ Sampling Method: {args.sampler}")
        print(f"✅ Best Validation AUC: {best_auc:.4f}")
        print(f"✅ Model Name: {model_name}")
        print(f"✅ Total Execution Time: {total_time:.2f} seconds")
        print(f"✅ Results saved to CSV with proper model naming")
        print("=" * 80)
        print("📊 Check results/final_evaluation_metrics.csv for evaluation results")
        print("📝 Check results/training_log_*.csv for training logs")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
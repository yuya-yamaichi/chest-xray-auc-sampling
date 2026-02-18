#!/usr/bin/env python3
"""
Statistical Analysis Tool for Master's Thesis Research
Analyze comprehensive results from multiple runs with different seeds

Usage:
    python src/analysis/statistical_analysis.py --results_file results/comprehensive_results.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def load_results(results_file):
    """Load comprehensive results from CSV file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} experimental results from {results_file}")
    return df

def analyze_by_method(df):
    """Analyze results grouped by sampling method."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS BY SAMPLING METHOD")
    print("="*80)
    
    # Metrics to analyze
    metrics = ['precision', 'recall', 'f1_score', 'balanced_accuracy', 'g_mean', 'auc', 'auprc']
    
    # Group by sampling method
    grouped = df.groupby(['pathology', 'sampler'])
    
    results_summary = []
    
    for (pathology, sampler), group in grouped:
        n_runs = len(group)
        print(f"\n{pathology} - {sampler} ({n_runs} runs):")
        print("-" * 60)
        
        row_data = {'pathology': pathology, 'sampler': sampler, 'n_runs': n_runs}
        
        for metric in metrics:
            if metric in group.columns:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                min_val = group[metric].min()
                max_val = group[metric].max()
                
                print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
                
                row_data[f'{metric}_mean'] = mean_val
                row_data[f'{metric}_std'] = std_val
                row_data[f'{metric}_min'] = min_val
                row_data[f'{metric}_max'] = max_val
        
        results_summary.append(row_data)
    
    return pd.DataFrame(results_summary)

def find_best_methods(summary_df):
    """Identify best performing methods for each pathology."""
    print("\n" + "="*80)
    print("BEST PERFORMING METHODS (by AUC)")
    print("="*80)
    
    for pathology in summary_df['pathology'].unique():
        pathology_data = summary_df[summary_df['pathology'] == pathology]
        best_method = pathology_data.loc[pathology_data['auc_mean'].idxmax()]
        
        print(f"\n{pathology}:")
        print(f"  Best method: {best_method['sampler']}")
        print(f"  AUC: {best_method['auc_mean']:.4f} ± {best_method['auc_std']:.4f}")
        print(f"  F1:  {best_method['f1_score_mean']:.4f} ± {best_method['f1_score_std']:.4f}")

def save_summary(summary_df, output_dir):
    """Save statistical summary to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'statistical_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nStatistical summary saved to: {summary_file}")
    return summary_file

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of thesis results')
    parser.add_argument('--results_file', type=str, default='results/comprehensive_results.csv',
                       help='Path to comprehensive results CSV file')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results_file)
    
    # Perform statistical analysis
    summary_df = analyze_by_method(df)
    
    # Find best methods
    find_best_methods(summary_df)
    
    # Save summary
    save_summary(summary_df, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Sampling Ratio Analysis Script
Analyzes the results from sampling ratio experiments for master's thesis research.

Usage:
    python scripts/analyze_sampling_ratios.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config import load_config

def load_experiment_results():
    """Load training log results from CSV file.
    拡張: sampling_applied 列の欠損を扱い、必要に応じてブール化する。
    """
    results_file = "results/training_log_cardiomegaly.csv"
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Training log not found: {results_file}")
    
    df = pd.read_csv(results_file)
    # 正規化: sampling_applied が存在すればブール化
    if 'sampling_applied' in df.columns:
        df['sampling_applied'] = df['sampling_applied'].map({True: True, False: False, 'True': True, 'False': False}).fillna(False)
    else:
        df['sampling_applied'] = False
    
    print(f"✅ Loaded {len(df)} experiment results from {results_file}")
    return df

def filter_ratio_experiments(df):
    """Filter results to only include the ratio experiments.
    拡張: sampling_applied==True のフィルタをデフォルト適用。
    """
    target_ratios = ["1:2", "1:3", "1:5"]
    
    df['sampling_ratio_str'] = df['sampling_ratio_requested'].astype(str)
    ratio_experiments = df[df['sampling_ratio_str'].isin(target_ratios)].copy()
    
    # デフォルトでサンプリング適用済みのみ
    applied = ratio_experiments[ratio_experiments['sampling_applied'] == True].copy()
    
    print(f"📊 Found {len(ratio_experiments)} ratio experiments (raw), {len(applied)} with sampling_applied==True")
    
    if len(applied) == 0:
        print("⚠️  No applied ratio experiments found. Consider checking feasibility/runner settings.")
    
    return applied

def analyze_sampling_performance(df):
    """Analyze sampling performance across different methods and ratios.
    拡張: 比率誤差 (requested vs achieved) を出力。
    """
    print("\n🔍 SAMPLING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Group by sampler and ratio
    grouped = df.groupby(['sampler', 'sampling_ratio_requested'])
    
    # Calculate statistics
    performance_stats = grouped.agg({
        'best_val_auc': ['mean', 'std', 'min', 'max', 'count'],
        'training_time_sec': ['mean', 'std'],
        'sampling_ratio_achieved': ['mean', 'std']
    }).round(4)
    
    # 追加: 誤差要約
    ratio_map = {"1:2": 0.5, "1:3": 1/3, "1:5": 0.2}
    df = df.copy()
    df['requested_numeric'] = df['sampling_ratio_requested'].map(ratio_map)
    df['ratio_abs_error'] = (df['sampling_ratio_achieved'] - df['requested_numeric']).abs()
    err_summary = df.groupby(['sampler', 'sampling_ratio_requested'])['ratio_abs_error'].agg(['mean', 'std', 'max']).round(6)
    
    print("📈 AUC Performance by Sampler and Ratio:")
    print(performance_stats['best_val_auc'])
    
    print("\n⏱️  Training Time by Sampler and Ratio:")
    print(performance_stats['training_time_sec'])
    
    print("\n🎯 Achieved vs Requested Ratios:")
    print(performance_stats['sampling_ratio_achieved'])
    
    print("\nΔ Ratio Absolute Error (mean/std/max):")
    print(err_summary)
    
    return performance_stats

def create_visualizations(df, output_dir="results/sampling_ratio_experiments"):
    """Create visualizations for the sampling ratio analysis.
    拡張: 比率誤差のヒストグラムを追加。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. AUC Performance Comparison
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='sampling_ratio_requested', y='best_val_auc', hue='sampler')
    plt.title('AUC Performance by Sampling Method and Ratio\n(Cardiomegaly Classification)', fontsize=14, fontweight='bold')
    plt.xlabel('Sampling Ratio (Minority:Majority)', fontsize=12)
    plt.ylabel('Validation AUC', fontsize=12)
    plt.legend(title='Sampling Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/auc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Time Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='sampling_ratio_requested', y='training_time_sec', hue='sampler', ci='sd')
    plt.title('Training Time by Sampling Method and Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Sampling Ratio (Minority:Majority)', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.legend(title='Sampling Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Ratio Achievement Analysis
    plt.figure(figsize=(10, 8))
    
    # Convert ratio strings to numeric values for comparison
    ratio_mapping = {"1:2": 0.5, "1:3": 0.333, "1:5": 0.2}
    df_ratio = df.copy()
    df_ratio['requested_numeric'] = df_ratio['sampling_ratio_requested'].map(ratio_mapping)
    df_ratio['ratio_error'] = abs(df_ratio['sampling_ratio_achieved'] - df_ratio['requested_numeric'])
    
    sns.scatterplot(data=df_ratio, x='requested_numeric', y='sampling_ratio_achieved', 
                   hue='sampler', s=100, alpha=0.7)
    
    # Add perfect ratio line
    x_line = np.array([0.2, 0.333, 0.5])
    plt.plot(x_line, x_line, 'k--', alpha=0.5, label='Perfect Ratio')
    
    plt.title('Requested vs Achieved Sampling Ratios', fontsize=14, fontweight='bold')
    plt.xlabel('Requested Ratio (Minority:Majority)', fontsize=12)
    plt.ylabel('Achieved Ratio', fontsize=12)
    plt.legend(title='Sampling Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ratio_achievement.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3b. Ratio Error Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_ratio, x='ratio_error', hue='sampler', bins=30, kde=True, alpha=0.5)
    plt.title('Distribution of Absolute Ratio Errors', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Error |achieved - requested|', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ratio_error_hist.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Performance Heatmap
    pivot_data = df.pivot_table(values='best_val_auc', 
                               index='sampler', 
                               columns='sampling_ratio_requested', 
                               aggfunc='mean')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.4f', cbar_kws={'label': 'Mean AUC'})
    plt.title('Mean AUC Performance Heatmap\nby Sampling Method and Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Sampling Ratio (Minority:Majority)', fontsize=12)
    plt.ylabel('Sampling Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {output_dir}/")

def generate_summary_report(df, stats, output_dir="results/sampling_ratio_experiments"):
    """Generate a comprehensive summary report.
    拡張: sampling_applied フィルタ注記と ADASYN ポスト・バランシングの注記を追記。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = f"{output_dir}/sampling_ratio_analysis_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Sampling Ratio Comparison Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Dataset:** Cardiomegaly Classification (CheXpert)\n")
        f.write(f"**Total Experiments (filtered):** {len(df)}\n\n")
        f.write("> Note: Results are filtered to rows with `sampling_applied==True` to avoid bias from failed or skipped sampling.\n\n")
        f.write("> ADASYN post-balancing may adjust ratios within epsilon when deviation > ε; see config.\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write("- **Target Ratios:** 1:2 (0.5), 1:3 (0.333), 1:5 (0.2)\n")
        f.write("- **Sampling Methods:** " + ", ".join(df['sampler'].unique()) + "\n")
        f.write("- **Seeds:** " + ", ".join(map(str, sorted(df['seed'].unique()))) + "\n")
        f.write("- **Pathology:** Cardiomegaly (class_id=0)\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Best performing combinations
        best_overall = df.loc[df['best_val_auc'].idxmax()]
        f.write(f"### Best Overall Performance\n")
        f.write(f"- **Method:** {best_overall['sampler']}\n")
        f.write(f"- **Ratio:** {best_overall['sampling_ratio_requested']}\n")
        f.write(f"- **AUC:** {best_overall['best_val_auc']:.4f}\n")
        f.write(f"- **Seed:** {best_overall['seed']}\n\n")
        
        # Best by ratio
        f.write("### Best Performance by Ratio\n\n")
        for ratio in ["1:2", "1:3", "1:5"]:
            ratio_data = df[df['sampling_ratio_requested'] == ratio]
            if len(ratio_data) > 0:
                best_ratio = ratio_data.loc[ratio_data['best_val_auc'].idxmax()]
                f.write(f"**Ratio {ratio}:**\n")
                f.write(f"- Method: {best_ratio['sampler']}\n")
                f.write(f"- AUC: {best_ratio['best_val_auc']:.4f}\n")
                f.write(f"- Achieved Ratio: {best_ratio['sampling_ratio_achieved']:.4f}\n\n")
        
        # Performance statistics table
        f.write("## Detailed Statistics\n\n")
        f.write("### AUC Performance (Mean ± Std)\n\n")
        f.write("| Sampling Method | 1:2 | 1:3 | 1:5 |\n")
        f.write("|-----------------|-----|-----|-----|\n")
        
        for sampler in df['sampler'].unique():
            row = f"| {sampler} |"
            for ratio in ["1:2", "1:3", "1:5"]:
                subset = df[(df['sampler'] == sampler) & (df['sampling_ratio_requested'] == ratio)]
                if len(subset) > 0:
                    mean_auc = subset['best_val_auc'].mean()
                    std_auc = subset['best_val_auc'].std()
                    row += f" {mean_auc:.4f} ± {std_auc:.4f} |"
                else:
                    row += " N/A |"
            f.write(row + "\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Calculate overall performance by method
        method_performance = df.groupby('sampler')['best_val_auc'].agg(['mean', 'std']).round(4)
        best_method = method_performance['mean'].idxmax()
        
        f.write(f"1. **Best Overall Method:** {best_method} with mean AUC of {method_performance.loc[best_method, 'mean']:.4f}\n")
        
        # Ratio recommendations
        ratio_performance = df.groupby('sampling_ratio_requested')['best_val_auc'].mean().round(4)
        best_ratio = ratio_performance.idxmax()
        f.write(f"2. **Best Overall Ratio:** {best_ratio} with mean AUC of {ratio_performance[best_ratio]:.4f}\n")
        
        f.write("\n3. **For consistent performance:** Consider methods with low standard deviation\n")
        f.write("4. **For computational efficiency:** Check training time vs performance trade-offs\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `auc_comparison.png` - Box plots comparing AUC across methods and ratios\n")
        f.write("- `training_time_comparison.png` - Bar charts showing training time comparison\n")
        f.write("- `ratio_achievement.png` - Scatter plot of requested vs achieved ratios\n")
        f.write("- `performance_heatmap.png` - Heatmap of mean AUC performance\n")
        f.write("- `ratio_error_hist.png` - Histogram of absolute ratio errors\n")
        f.write("- `sampling_ratio_analysis_report.md` - This comprehensive report\n")
    
    print(f"📋 Summary report saved to: {report_file}")

def main():
    """Main analysis function."""
    print("🔬 Sampling Ratio Analysis Script")
    print("=" * 50)
    
    try:
        # Load results
        df = load_experiment_results()
        
        # Filter to ratio experiments
        ratio_df = filter_ratio_experiments(df)
        
        if len(ratio_df) == 0:
            print("❌ No ratio experiments found to analyze.")
            return
        
        # Analyze performance
        stats = analyze_sampling_performance(ratio_df)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        create_visualizations(ratio_df)
        
        # Generate report
        print("\n📋 Generating summary report...")
        generate_summary_report(ratio_df, stats)
        
        print("\n🎉 Analysis complete!")
        print("📁 Check results/sampling_ratio_experiments/ for all outputs")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick test script to verify sampling ratio functionality
Tests each method with one ratio to ensure everything works before full batch
"""

import subprocess
import os
import sys
import time

def run_single_test(sampler, ratio, seed=42):
    """Run a single test experiment."""
    print(f"🧪 Testing: {sampler} with ratio {ratio}, seed {seed}")
    
    cmd = [
        "python", "src/pipelines/finetune.py",
        "--class_id", "0",
        "--sampler", sampler,
        "--sampling_ratio", ratio,
        "--seed", str(seed)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 min timeout
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS in {end_time - start_time:.1f}s")
            return True
        else:
            print(f"❌ FAILED with return code {result.returncode}")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT after 20 minutes")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    """Run quick tests for each sampling method."""
    print("🔬 Testing Sampling Ratio Setup")
    print("=" * 40)
    
    # Test one method with each ratio to verify functionality
    test_cases = [
        ("RandomOverSampler", "1:2"),
        ("SMOTE", "1:3"), 
        ("ADASYN", "1:5"),
    ]
    
    results = []
    for sampler, ratio in test_cases:
        print(f"\n🧪 Testing {sampler} with {ratio}")
        print("-" * 30)
        success = run_single_test(sampler, ratio)
        results.append((sampler, ratio, success))
    
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    for sampler, ratio, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {sampler} with {ratio}")
    
    # Check if training log was created/updated
    log_file = "results/training_log_cardiomegaly.csv"
    if os.path.exists(log_file):
        print(f"\n📋 Training log exists: {log_file}")
        
        # Show the last few entries
        try:
            import pandas as pd
            df = pd.read_csv(log_file)
            print(f"Total entries: {len(df)}")
            if len(df) > 0:
                print("\nLast entry:")
                print(df.iloc[-1][['sampler', 'sampling_ratio_requested', 'sampling_ratio_achieved', 'best_val_auc']].to_string())
        except Exception as e:
            print(f"Could not read log: {e}")
    else:
        print(f"\n⚠️  No training log found at {log_file}")
    
    successful_tests = sum(1 for _, _, success in results if success)
    print(f"\n🎯 {successful_tests}/{len(results)} tests passed")
    
    if successful_tests == len(results):
        print("🚀 All tests passed! Ready for full batch experiments.")
    else:
        print("⚠️  Some tests failed. Check errors above before running full batch.")

if __name__ == "__main__":
    main()
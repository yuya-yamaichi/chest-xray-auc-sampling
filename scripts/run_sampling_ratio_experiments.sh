#!/bin/bash

# Sampling Ratio Comparison Experiments
# Tests various sampling methods at fixed ratios: 1:2, 1:3, 1:5
# For systematic comparison in master's thesis research

set -e  # Exit on any error

echo "🔬 Starting Sampling Ratio Comparison Experiments"
echo "================================================================="
echo "Target Ratios: 1:2 (0.5), 1:3 (0.333), 1:5 (0.2)"
echo "Sampling Methods: RandomOverSampler, SMOTE, ADASYN, SMOTETomek, RandomUnderSampler"
echo "Target Pathology: Cardiomegaly (class_id=0)"
echo "Seeds: 42, 123, 456 (for statistical significance)"
echo "================================================================="

# Configuration
CLASS_ID=0  # Cardiomegaly
RATIOS=("1:2" "1:3" "1:5")
# Keep 3 seeds as default (aligned with comprehensive pipeline)
SEEDS=(42 123 456)
SAMPLERS=("RandomOverSampler" "SMOTE" "ADASYN" "SMOTETomek" "RandomUnderSampler")

# Create results directory
mkdir -p results/sampling_ratio_experiments
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="results/sampling_ratio_experiments/experiment_log_${TIMESTAMP}.txt"

echo "📋 Experiment Log: $LOG_FILE"
echo "=================================================================" | tee "$LOG_FILE"
echo "Sampling Ratio Comparison Experiments - Started: $(date)" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# Function to run single experiment
run_experiment() {
    local sampler=$1
    local ratio=$2
    local seed=$3
    
    echo "🚀 Running: $sampler with ratio $ratio, seed $seed" | tee -a "$LOG_FILE"
    echo "   Command: python src/pipelines/finetune.py --class_id $CLASS_ID --sampler $sampler --sampling_ratio \"$ratio\" --seed $seed" | tee -a "$LOG_FILE"
    
    # Run the experiment
    if python src/pipelines/finetune.py --class_id "$CLASS_ID" --sampler "$sampler" --sampling_ratio "$ratio" --seed "$seed"; then
        echo "   ✅ SUCCESS: $sampler, ratio $ratio, seed $seed" | tee -a "$LOG_FILE"
    else
        echo "   ❌ FAILED: $sampler, ratio $ratio, seed $seed" | tee -a "$LOG_FILE"
        return 1
    fi
    echo "" | tee -a "$LOG_FILE"
}

# Main experiment loop
total_experiments=$((${#SAMPLERS[@]} * ${#RATIOS[@]} * ${#SEEDS[@]}))
current_experiment=0

echo "📊 Total experiments to run: $total_experiments" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for sampler in "${SAMPLERS[@]}"; do
    echo "🔧 Testing sampler: $sampler" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    for ratio in "${RATIOS[@]}"; do
        echo "  📐 Testing ratio: $ratio" | tee -a "$LOG_FILE"
        
        for seed in "${SEEDS[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "  [${current_experiment}/${total_experiments}] Testing seed: $seed" | tee -a "$LOG_FILE"
            
            # Run experiment with timeout (30 minutes max per experiment)
            timeout 1800 run_experiment "$sampler" "$ratio" "$seed" || {
                echo "   ⚠️  TIMEOUT or ERROR: $sampler, ratio $ratio, seed $seed" | tee -a "$LOG_FILE"
            }
        done
    done
    echo "" | tee -a "$LOG_FILE"
done

# Generate summary report
echo "📈 EXPERIMENT SUMMARY" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"

# Check for training log files
echo "" | tee -a "$LOG_FILE"
echo "📋 Generated Training Logs:" | tee -a "$LOG_FILE"
if ls results/training_log_cardiomegaly.csv >/dev/null 2>&1; then
    echo "   ✅ results/training_log_cardiomegaly.csv" | tee -a "$LOG_FILE"
    
    # Show summary statistics
    echo "" | tee -a "$LOG_FILE"
    echo "📊 Quick Summary (last 10 entries):" | tee -a "$LOG_FILE"
    tail -10 results/training_log_cardiomegaly.csv | tee -a "$LOG_FILE"
else
    echo "   ⚠️  No training log found" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "🎉 All sampling ratio experiments completed!" | tee -a "$LOG_FILE"
echo "📊 Check results/training_log_cardiomegaly.csv for detailed results" | tee -a "$LOG_FILE"
echo "📋 Full experiment log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"
#!/bin/bash

# =============================================================================
# Phase 2 Experiments Automation Script
# Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization
# 
# This script orchestrates the complete experimental workflow:
# 1. Ensures pre-trained model exists (runs pre-training if needed)
# 2. Executes all fine-tuning experiments systematically
# 3. Records results in CSV format for analysis
# 
# Usage: Run from project root directory
#   ./scripts/run_phase2_experiments.sh
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Exit when using undefined variables

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Paths (relative to project root)
PRETRAINED_MODEL_PATH="models/pretrained/best_pretrained_model.pth"
PRETRAIN_SCRIPT="src/pipelines/pretrain.py"
FINETUNE_SCRIPT="src/pipelines/finetune.py"
RESULTS_FILE="results/training_log.csv"
RESULTS_DIR="results"

# Experiment parameters
CLASS_ID=0  # 0: Cardiomegaly (primary focus of research)

# All sampling methods to test
SAMPLERS=(
    "none"
    "RandomOverSampler"
    "SMOTE"
    "ADASYN"
    "RandomUnderSampler"
    "TomekLinks"
    "NeighbourhoodCleaningRule"
    "SMOTETomek"
    "SMOTEENN"
)

# Multiple random seeds for statistical significance
SEEDS=(42 123 456)

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#SAMPLERS[@]} * ${#SEEDS[@]}))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  Phase 2 Experiments: Sampling Methods + AUC Optimization${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "Target pathology: ${GREEN}Cardiomegaly (class_id=${CLASS_ID})${NC}"
    echo -e "Sampling methods: ${GREEN}${#SAMPLERS[@]}${NC}"
    echo -e "Random seeds: ${GREEN}${#SEEDS[@]}${NC}"
    echo -e "Total experiments: ${GREEN}${TOTAL_EXPERIMENTS}${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_section() {
    echo -e "\n${YELLOW}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================

check_working_directory() {
    print_section "Verifying working directory"
    
    if [[ ! -f "src/pipelines/pretrain.py" ]] || [[ ! -f "src/pipelines/finetune.py" ]]; then
        print_error "This script must be run from the project root directory"
        print_error "Expected files not found: src/pipelines/pretrain.py, src/pipelines/finetune.py"
        exit 1
    fi
    
    print_success "Working directory verified"
}

check_python_environment() {
    print_section "Checking Python environment"
    
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please ensure Python is installed and activated."
        exit 1
    fi
    
    print_success "Python environment ready"
}

# =============================================================================
# PRE-TRAINING PHASE
# =============================================================================

ensure_pretrained_model() {
    print_section "Checking pre-trained model"
    
    if [[ -f "$PRETRAINED_MODEL_PATH" ]]; then
        print_success "Pre-trained model found at $PRETRAINED_MODEL_PATH"
        print_info "Skipping pre-training phase"
        return 0
    fi
    
    print_info "Pre-trained model not found. Running pre-training..."
    
    # Create models directory if it doesn't exist
    mkdir -p "$(dirname "$PRETRAINED_MODEL_PATH")"
    
    # Run pre-training
    print_info "Executing: python $PRETRAIN_SCRIPT"
    if python "$PRETRAIN_SCRIPT"; then
        print_success "Pre-training completed successfully"
    else
        print_error "Pre-training failed"
        exit 1
    fi
    
    # Verify model was created
    if [[ -f "$PRETRAINED_MODEL_PATH" ]]; then
        print_success "Pre-trained model created at $PRETRAINED_MODEL_PATH"
    else
        print_error "Pre-training script completed but model file not found"
        exit 1
    fi
}

# =============================================================================
# RESULTS FILE MANAGEMENT
# =============================================================================

setup_results_file() {
    print_section "Setting up results file"
    
    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"
    
    # Note: CSV file will be created automatically by finetune.py with proper headers
    # The training_log.csv format is: timestamp,pathology,class_id,sampler,seed,best_val_auc,training_time_sec,total_time_sec,model_path
    if [[ -f "$RESULTS_FILE" ]]; then
        print_info "Results file already exists: $RESULTS_FILE"
        print_info "New results will be appended"
    else
        print_info "Results file will be created automatically by finetune.py"
    fi
}

# Check if experiment already completed (for resumability)
is_experiment_completed() {
    local sampler=$1
    local seed=$2
    
    if [[ -f "$RESULTS_FILE" ]]; then
        # Check if this exact combination already exists in results
        # CSV format: timestamp,pathology,class_id,sampler,seed,best_val_auc,training_time_sec,total_time_sec,model_path
        # Simple grep check for sampler,seed pattern (more reliable)
        if grep -q ",${sampler},${seed}," "$RESULTS_FILE"; then
            return 0  # Experiment already completed
        fi
    fi
    return 1  # Experiment not completed
}

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

run_single_experiment() {
    local sampler=$1
    local seed=$2
    local experiment_num=$3
    local total=$4
    
    print_info "[$experiment_num/$total] Running: sampler=$sampler, seed=$seed"
    
    # Check if experiment already completed
    if is_experiment_completed "$sampler" "$seed"; then
        print_info "Experiment already completed, skipping..."
        return 0
    fi
    
    # Run the fine-tuning experiment
    local output
    if output=$(python "$FINETUNE_SCRIPT" --class_id "$CLASS_ID" --sampler "$sampler" --seed "$seed" 2>&1); then
        # Parse the AUC value from output for display purposes
        local auc_value
        auc_value=$(echo "$output" | grep "Final Best AUC:" | awk '{print $NF}')
        
        if [[ -n "$auc_value" ]] && [[ "$auc_value" =~ ^[0-9]+\.[0-9]+$ ]]; then
            # Results are automatically logged by finetune.py to training_log.csv
            print_success "AUC: $auc_value - Result logged by finetune.py"
        else
            print_error "Failed to parse AUC value from output"
            print_error "Raw output: $output"
            return 1
        fi
    else
        print_error "Fine-tuning experiment failed for sampler=$sampler, seed=$seed"
        return 1
    fi
}

run_all_experiments() {
    print_section "Starting experimental phase"
    print_info "Total experiments to run: $TOTAL_EXPERIMENTS"
    echo ""
    
    local experiment_num=1
    local start_time=$(date +%s)
    
    # Nested loops: seeds × samplers (cycle through all samplers for each seed)
    for seed in "${SEEDS[@]}"; do
        for sampler in "${SAMPLERS[@]}"; do
            echo -e "${BLUE}--- Experiment $experiment_num/$TOTAL_EXPERIMENTS ---${NC}"
            
            if run_single_experiment "$sampler" "$seed" "$experiment_num" "$TOTAL_EXPERIMENTS"; then
                print_success "Experiment completed successfully"
            else
                print_error "Experiment failed, continuing with next..."
            fi
            
            echo ""
            ((experiment_num++))
        done
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    print_success "All experiments completed in ${hours}h ${minutes}m"
}

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

show_results_summary() {
    print_section "Results Summary"
    
    if [[ -f "$RESULTS_FILE" ]]; then
        local total_results
        total_results=$(tail -n +2 "$RESULTS_FILE" | wc -l)
        
        print_info "Results file: $RESULTS_FILE"
        print_info "Total experiments completed: $total_results"
        
        if [[ $total_results -gt 0 ]]; then
            echo ""
            echo "Sample results (sampler, seed, AUC):"
            # Show header with relevant columns
            echo "sampler,seed,best_val_auc"
            # Extract relevant columns (4, 5, 6) from the CSV and show first 5 results
            tail -n +2 "$RESULTS_FILE" | head -n 5 | awk -F',' '{print $4","$5","$6}'
            
            if [[ $total_results -gt 5 ]]; then
                echo "... (showing first 5 results)"
            fi
            
            echo ""
            print_info "For comprehensive evaluation, use: python src/pipelines/evaluate_simple.py"
        fi
    else
        print_error "Results file not found"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Print header
    print_header
    
    # Check prerequisites
    check_working_directory
    check_python_environment
    
    # Ensure pre-trained model exists (run pre-training if needed)
    ensure_pretrained_model
    
    # Setup results file
    setup_results_file
    
    # Run all experiments
    run_all_experiments
    
    # Show results summary
    show_results_summary
    
    print_section "Workflow Complete"
    print_success "All Phase 2 experiments have been completed successfully!"
    print_info "Results saved to: $RESULTS_FILE"
    print_info "You can now proceed with statistical analysis of the results."
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Trap Ctrl+C and provide clean exit
trap 'echo -e "\n${YELLOW}Script interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"
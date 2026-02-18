#!/bin/bash

# =============================================================================
# Multi-Pathology Experiments Automation Script
# Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization
# 
# This script orchestrates experiments for ALL 5 pathologies:
# 1. Ensures pre-trained model exists (runs pre-training if needed)
# 2. Executes all fine-tuning experiments for each pathology systematically
# 3. Records results in separate CSV files per pathology for analysis
# 
# Usage: Run from project root directory
#   ./scripts/run_all_pathologies_experiments.sh
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Exit when using undefined variables

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Paths (relative to project root)
PRETRAINED_MODEL_PATH="models/pretrained/best_pretrained_model.pth"
PRETRAIN_SCRIPT="src/pipelines/pretrain.py"
FINETUNE_SCRIPT="src/pipelines/finetune.py"
RESULTS_DIR="results"

# All pathologies to experiment with
declare -A PATHOLOGIES=(
    [0]="Cardiomegaly"
    [1]="Edema" 
    [2]="Consolidation"
    [3]="Atelectasis"
    [4]="Pleural_Effusion"
)

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
TOTAL_PATHOLOGIES=${#PATHOLOGIES[@]}
TOTAL_SAMPLERS=${#SAMPLERS[@]}
TOTAL_SEEDS=${#SEEDS[@]}
TOTAL_EXPERIMENTS=$((TOTAL_PATHOLOGIES * TOTAL_SAMPLERS * TOTAL_SEEDS))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo -e "${PURPLE}======================================================================${NC}"
    echo -e "${PURPLE}  Multi-Pathology Experiments: Sampling Methods + AUC Optimization${NC}"
    echo -e "${PURPLE}======================================================================${NC}"
    echo -e "Pathologies: ${GREEN}${TOTAL_PATHOLOGIES}${NC} (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion)"
    echo -e "Sampling methods: ${GREEN}${TOTAL_SAMPLERS}${NC}"
    echo -e "Random seeds: ${GREEN}${TOTAL_SEEDS}${NC}"
    echo -e "Total experiments: ${GREEN}${TOTAL_EXPERIMENTS}${NC}"
    echo -e "${PURPLE}======================================================================${NC}"
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

print_pathology_header() {
    local class_id=$1
    local pathology_name=${PATHOLOGIES[$class_id]}
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}  PATHOLOGY: $pathology_name (ID: $class_id)${NC}"
    echo -e "${PURPLE}========================================${NC}"
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

setup_results_directories() {
    print_section "Setting up results directories"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Create model directories for each pathology
    for class_id in "${!PATHOLOGIES[@]}"; do
        local pathology_name=${PATHOLOGIES[$class_id]}
        local pathology_dir="models/${pathology_name,,}"  # lowercase
        mkdir -p "$pathology_dir"
        print_info "Created directory: $pathology_dir"
    done
    
    print_success "Directory structure ready"
}

# Check if experiment already completed (for resumability)
is_experiment_completed() {
    local class_id=$1
    local sampler=$2
    local seed=$3
    local pathology_name=${PATHOLOGIES[$class_id]}
    local results_file="$RESULTS_DIR/training_log_${pathology_name,,}.csv"
    
    if [[ -f "$results_file" ]]; then
        # Check if this exact combination already exists in results
        if grep -q ",${sampler},${seed}," "$results_file"; then
            return 0  # Experiment already completed
        fi
    fi
    return 1  # Experiment not completed
}

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

run_single_experiment() {
    local class_id=$1
    local sampler=$2
    local seed=$3
    local experiment_num=$4
    local total=$5
    local pathology_name=${PATHOLOGIES[$class_id]}
    
    print_info "[$experiment_num/$total] $pathology_name: sampler=$sampler, seed=$seed"
    
    # Check if experiment already completed
    if is_experiment_completed "$class_id" "$sampler" "$seed"; then
        print_info "Experiment already completed, skipping..."
        return 0
    fi
    
    # Run the fine-tuning experiment
    local output
    if output=$(python "$FINETUNE_SCRIPT" --class_id "$class_id" --sampler "$sampler" --seed "$seed" 2>&1); then
        # Parse the AUC value from output for display purposes
        local auc_value
        auc_value=$(echo "$output" | grep -E "Best Validation AUC|Final Best AUC" | tail -1 | awk '{print $NF}')
        
        if [[ -n "$auc_value" ]] && [[ "$auc_value" =~ ^[0-9]+\.[0-9]+$ ]]; then
            print_success "$pathology_name: AUC=$auc_value"
        else
            # Try alternative parsing
            auc_value=$(echo "$output" | grep -oE "Val_AUC=[0-9]+\.[0-9]+" | tail -1 | cut -d'=' -f2)
            if [[ -n "$auc_value" ]]; then
                print_success "$pathology_name: AUC=$auc_value"
            else
                print_success "$pathology_name: Completed (AUC parsing failed)"
            fi
        fi
    else
        print_error "$pathology_name: Fine-tuning failed for sampler=$sampler, seed=$seed"
        print_error "Output: $output"
        return 1
    fi
}

run_pathology_experiments() {
    local class_id=$1
    local pathology_name=${PATHOLOGIES[$class_id]}
    local experiments_for_pathology=$((TOTAL_SAMPLERS * TOTAL_SEEDS))
    
    print_pathology_header "$class_id"
    print_info "Running $experiments_for_pathology experiments for $pathology_name"
    echo ""
    
    local pathology_exp_num=1
    local global_exp_num=$((class_id * experiments_for_pathology + 1))
    
    # Nested loops: seeds × samplers
    for seed in "${SEEDS[@]}"; do
        for sampler in "${SAMPLERS[@]}"; do
            echo -e "${BLUE}--- $pathology_name Experiment $pathology_exp_num/$experiments_for_pathology ---${NC}"
            
            if run_single_experiment "$class_id" "$sampler" "$seed" "$global_exp_num" "$TOTAL_EXPERIMENTS"; then
                print_success "Experiment completed successfully"
            else
                print_error "Experiment failed, continuing with next..."
            fi
            
            echo ""
            ((pathology_exp_num++))
            ((global_exp_num++))
        done
    done
    
    # Show pathology-specific results summary
    local results_file="$RESULTS_DIR/training_log_${pathology_name,,}.csv"
    if [[ -f "$results_file" ]]; then
        local completed_count
        completed_count=$(tail -n +2 "$results_file" | wc -l)
        print_success "$pathology_name: $completed_count experiments completed"
        print_info "Results saved to: $results_file"
    fi
}

run_all_experiments() {
    print_section "Starting multi-pathology experimental phase"
    print_info "Total experiments across all pathologies: $TOTAL_EXPERIMENTS"
    echo ""
    
    local start_time=$(date +%s)
    
    # Run experiments for each pathology
    for class_id in "${!PATHOLOGIES[@]}"; do
        run_pathology_experiments "$class_id"
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    print_success "All pathology experiments completed in ${hours}h ${minutes}m"
}

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

show_results_summary() {
    print_section "Multi-Pathology Results Summary"
    
    echo -e "${BLUE}Results by Pathology:${NC}"
    
    for class_id in "${!PATHOLOGIES[@]}"; do
        local pathology_name=${PATHOLOGIES[$class_id]}
        local results_file="$RESULTS_DIR/training_log_${pathology_name,,}.csv"
        
        if [[ -f "$results_file" ]]; then
            local total_results
            total_results=$(tail -n +2 "$results_file" | wc -l)
            print_info "$pathology_name: $total_results experiments completed"
        else
            print_error "$pathology_name: No results file found"
        fi
    done
    
    echo ""
    print_info "Individual results files located in: $RESULTS_DIR/"
    print_info "For comprehensive evaluation, use: python src/pipelines/evaluate.py"
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
    
    # Ensure pre-trained model exists
    ensure_pretrained_model
    
    # Setup directory structure
    setup_results_directories
    
    # Run all experiments
    run_all_experiments
    
    # Show results summary
    show_results_summary
    
    print_section "Multi-Pathology Workflow Complete"
    print_success "All experiments for 5 pathologies have been completed!"
    print_info "Results saved to pathology-specific CSV files in: $RESULTS_DIR/"
    print_info "Models saved to pathology-specific directories in: models/"
    print_info "You can now proceed with statistical analysis of the results."
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Trap Ctrl+C and provide clean exit
trap 'echo -e "\n${YELLOW}Script interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"
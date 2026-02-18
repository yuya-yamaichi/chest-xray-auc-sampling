#!/bin/bash

# =============================================================================
# Single Pathology Experiments Wrapper Script
# Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization
# 
# This script runs experiments for a SINGLE specified pathology with all sampling methods
# 
# Usage: Run from project root directory
#   ./scripts/run_single_pathology.sh <pathology_id> [sampling_ratio] [seed]
#   ./scripts/run_single_pathology.sh 0              # Cardiomegaly with default ratio and seed
#   ./scripts/run_single_pathology.sh 1 "1:2"       # Edema with 1:2 ratio
#   ./scripts/run_single_pathology.sh 2 "1:3" 456   # Consolidation with 1:3 ratio and seed 456
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

# Default to GPU 0 if not specified
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Paths (relative to project root)
PRETRAINED_MODEL_PATH="models/pretrained/best_pretrained_model.pth"
PRETRAIN_SCRIPT="src/pipelines/pretrain.py"
FINETUNE_SCRIPT="src/pipelines/finetune.py"
RESULTS_DIR="results"

# Pathology mapping
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_usage() {
    echo "Usage: $0 <pathology_id> [sampling_ratio] [seed]"
    echo ""
    echo "Parameters:"
    echo "  pathology_id     Required. Pathology to test (0-4)"
    echo "  sampling_ratio   Optional. Sampling ratio (default: auto)"
    echo "  seed             Optional. Random seed (default: 123)"
    echo ""
    echo "Available pathology IDs:"
    for id in "${!PATHOLOGIES[@]}"; do
        echo "  $id: ${PATHOLOGIES[$id]}"
    done
    echo ""
    echo "Examples:"
    echo "  $0 0              # Cardiomegaly with default ratio and seed"
    echo "  $0 1 '1:2'        # Edema with 1:2 ratio"
    echo "  $0 2 '1:3' 456    # Consolidation with 1:3 ratio and seed 456"
}

print_header() {
    local class_id=$1
    local ratio=$2
    local seed=$3
    local pathology_name=${PATHOLOGIES[$class_id]}
    local total_experiments=${#SAMPLERS[@]}
    
    echo -e "${PURPLE}======================================================================${NC}"
    echo -e "${PURPLE}  Single Pathology Experiments: ${pathology_name}${NC}"
    echo -e "${PURPLE}======================================================================${NC}"
    echo -e "Pathology: ${GREEN}${pathology_name} (ID: $class_id)${NC}"
    echo -e "Sampling ratio: ${GREEN}${ratio}${NC}"
    echo -e "Random seed: ${GREEN}${seed}${NC}"
    echo -e "Sampling methods: ${GREEN}${#SAMPLERS[@]}${NC}"
    echo -e "Total experiments: ${GREEN}${total_experiments}${NC}"
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

# =============================================================================
# VALIDATION
# =============================================================================

validate_arguments() {
    if [[ $# -lt 1 ]] || [[ $# -gt 3 ]]; then
        print_error "Invalid number of arguments. Expected 1-3 arguments: pathology_id [sampling_ratio] [seed]"
        print_usage
        exit 1
    fi
    
    local class_id=$1
    local ratio=${2:-"auto"}
    local seed=${3:-123}
    
    # Validate pathology_id
    if [[ ! "$class_id" =~ ^[0-4]$ ]]; then
        print_error "Invalid pathology_id: $class_id"
        print_error "Must be an integer between 0-4"
        print_usage
        exit 1
    fi
    
    if [[ -z "${PATHOLOGIES[$class_id]:-}" ]]; then
        print_error "Unknown pathology_id: $class_id"
        print_usage
        exit 1
    fi
    
    # Validate sampling ratio format
    if [[ ! "$ratio" =~ ^(auto|[0-9]+:[0-9]+|[0-9]*\.?[0-9]+)$ ]]; then
        print_error "Invalid sampling_ratio format: $ratio"
        print_error "Valid formats: 'auto', '1:2', '1:3', '0.5', etc."
        print_usage
        exit 1
    fi
    
    # Validate seed
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        print_error "Invalid seed: $seed (must be a positive integer)"
        print_usage
        exit 1
    fi
}

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

setup_results_directory() {
    local class_id=$1
    local pathology_name=${PATHOLOGIES[$class_id]}
    
    print_section "Setting up results directory"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Create model directory for this pathology
    local pathology_dir="models/${pathology_name,,}"  # lowercase
    mkdir -p "$pathology_dir"
    print_info "Created directory: $pathology_dir"
    
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
    local ratio=$3
    local seed=$4
    local experiment_num=$5
    local total=$6
    local pathology_name=${PATHOLOGIES[$class_id]}
    
    print_info "[$experiment_num/$total] sampler=$sampler, ratio=$ratio, seed=$seed"
    
    # Check if experiment already completed
    if is_experiment_completed "$class_id" "$sampler" "$seed"; then
        print_info "Experiment already completed, skipping..."
        return 0
    fi
    
    # Run the fine-tuning experiment with sampling ratio
    local output
    if output=$(python "$FINETUNE_SCRIPT" --class_id "$class_id" --sampler "$sampler" --sampling_ratio "$ratio" --seed "$seed" 2>&1); then
        # Parse the AUC value from output for display purposes
        local auc_value
        auc_value=$(echo "$output" | grep -E "Best Validation AUC|Final Best AUC" | tail -1 | awk '{print $NF}')
        
        if [[ -n "$auc_value" ]] && [[ "$auc_value" =~ ^[0-9]+\.[0-9]+$ ]]; then
            print_success "AUC: $auc_value - Result logged automatically"
        else
            # Try alternative parsing
            auc_value=$(echo "$output" | grep -oE "Val_AUC=[0-9]+\.[0-9]+" | tail -1 | cut -d'=' -f2)
            if [[ -n "$auc_value" ]]; then
                print_success "AUC: $auc_value - Result logged automatically"
            else
                print_success "Completed (AUC parsing failed) - Result logged automatically"
            fi
        fi
    else
        print_error "Fine-tuning experiment failed for sampler=$sampler, seed=$seed"
        print_error "Output: $output"
        return 1
    fi
}

run_pathology_experiments() {
    local class_id=$1
    local ratio=$2
    local seed=$3
    local pathology_name=${PATHOLOGIES[$class_id]}
    local total_experiments=${#SAMPLERS[@]}
    
    print_section "Starting experiments for $pathology_name"
    print_info "Running $total_experiments experiments with ratio $ratio and seed $seed"
    echo ""
    
    local experiment_num=1
    local start_time=$(date +%s)
    
    # Run all samplers for this pathology
    for sampler in "${SAMPLERS[@]}"; do
        echo -e "${BLUE}--- Experiment $experiment_num/$total_experiments ---${NC}"
        
        if run_single_experiment "$class_id" "$sampler" "$ratio" "$seed" "$experiment_num" "$total_experiments"; then
            print_success "Experiment completed successfully"
        else
            print_error "Experiment failed, continuing with next..."
        fi
        
        echo ""
        ((experiment_num++))
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    print_success "All $pathology_name experiments completed in ${hours}h ${minutes}m"
}

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

show_results_summary() {
    local class_id=$1
    local pathology_name=${PATHOLOGIES[$class_id]}
    
    print_section "Results Summary for $pathology_name"
    
    local results_file="$RESULTS_DIR/training_log_${pathology_name,,}.csv"
    
    if [[ -f "$results_file" ]]; then
        local total_results
        total_results=$(tail -n +2 "$results_file" | wc -l)
        
        print_info "Results file: $results_file"
        print_info "Total experiments completed: $total_results"
        
        if [[ $total_results -gt 0 ]]; then
            echo ""
            echo "Sample results (sampler, seed, AUC):"
            echo "sampler,seed,best_val_auc"
            # Show first 5 results
            tail -n +2 "$results_file" | head -n 5 | awk -F',' '{print $4","$5","$6}'
            
            if [[ $total_results -gt 5 ]]; then
                echo "... (showing first 5 results)"
            fi
            
            echo ""
            print_info "For comprehensive evaluation, use: python src/pipelines/evaluate.py --class_id $class_id"
        fi
    else
        print_error "Results file not found: $results_file"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    local class_id=$1
    local ratio=${2:-"auto"}
    local seed=${3:-123}
    local pathology_name=${PATHOLOGIES[$class_id]}
    
    # Print header
    print_header "$class_id" "$ratio" "$seed"
    
    # Check prerequisites
    check_working_directory
    check_python_environment
    
    # Ensure pre-trained model exists
    ensure_pretrained_model
    
    # Setup directory structure
    setup_results_directory "$class_id"
    
    # Run experiments
    run_pathology_experiments "$class_id" "$ratio" "$seed"
    
    # Show results summary
    show_results_summary "$class_id"
    
    print_section "Single Pathology Workflow Complete"
    print_success "All experiments for $pathology_name with ratio $ratio and seed $seed have been completed!"
    print_info "Results saved to: $RESULTS_DIR/training_log_${pathology_name,,}.csv"
    print_info "Models saved to: models/${pathology_name,,}/"
    print_info "You can now proceed with analysis of the results."
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Validate arguments
validate_arguments "$@"

# Trap Ctrl+C and provide clean exit
trap 'echo -e "\n${YELLOW}Script interrupted by user${NC}"; exit 130' INT

# Run main function with all arguments
main "$1" "${2:-auto}" "${3:-123}"
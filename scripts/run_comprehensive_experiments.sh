#!/bin/bash

# =============================================================================
# Comprehensive Sampling Method Experiments Master Orchestration Script
# Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization
# 
# This is the MASTER script that orchestrates the complete experimental workflow:
# 1. Same ratio, all pathologies: 1:1 Cardiomegaly → 1:1 Edema → etc.
# 2. Different ratios: 1:2, 1:3, 1:4 across all pathologies  
# 3. Multiple seeds: Execute 5 times with different seeds for statistical significance
# 
# Experimental Order (as requested):
# For each seed:
#   1:1 Cardiomegaly: all sampling methods → 1:1 Edema: all sampling methods → etc.
#   1:2 Cardiomegaly: all sampling methods → 1:2 Edema: all sampling methods → etc.
#   1:3 Cardiomegaly: all sampling methods → 1:3 Edema: all sampling methods → etc.
#   1:4 Cardiomegaly: all sampling methods → 1:4 Edema: all sampling methods → etc.
# 
# Usage: Run from project root directory
#   ./scripts/run_comprehensive_experiments.sh [--resume] [--seeds SEED1,SEED2,...] [--ratios RATIO1,RATIO2,...]
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Exit when using undefined variables

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Paths (relative to project root)
RATIO_PATHOLOGY_SCRIPT="scripts/run_ratio_pathology_experiments.sh"
AUTO_OPT_SCRIPT="scripts/optimize_auto_cleaning_params.sh"
TRACKER_SCRIPT="src/utils/experiment_tracker.py"
RESULTS_DIR="results"

# Default experimental configuration (updated to 3 seeds)
DEFAULT_SEEDS=(42 123 456)
# Default ratios for ratio-grid experiments
# Note: Cleaning/Hybrid methods will be auto-filtered by feasibility in the ratio runner
DEFAULT_RATIOS=("0.5" "0.6" "0.7" "0.8" "0.9" "1.0")

# Pathology mapping
declare -A PATHOLOGY_NAMES=(
    [0]="Cardiomegaly"
    [1]="Edema" 
    [2]="Consolidation"
    [3]="Atelectasis"
    [4]="Pleural_Effusion"
)

# Sampling methods count (for calculations)
SAMPLING_METHODS_COUNT=9
PATHOLOGIES_COUNT=5

# Global variables for configuration
SEEDS=()
RATIOS=()
RESUME_MODE=false
DRY_RUN=false

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --resume                Resume from where experiments left off
    --seeds SEED1,SEED2     Comma-separated list of seeds (default: 42,123,456)
    --ratios RATIO1,RATIO2  Comma-separated list of ratios (default: 0.5,0.6,0.7,0.8,0.9,1.0)
    --dry-run              Show what would be executed without running experiments
    --help                 Show this help message

EXAMPLES:
    # Run complete experimental suite (default configuration)
    $0
    
    # Resume interrupted experiments
    $0 --resume
    
    # Run with custom seeds
    $0 --seeds 42,123,456
    
    # Run with custom ratios  
    $0 --ratios "0.5,0.7,1.0"
    
    # Test configuration without running
    $0 --dry-run --seeds 42,123 --ratios "1:1,1:2"

EXPERIMENTAL ORDER:
The script executes experiments in the exact order requested:

For each seed value:
  Ratio 0.5: Cardiomegaly (all 9 sampling methods) → Edema (all 9 methods) → ... → Pleural Effusion
  Ratio 0.6: Cardiomegaly (all 9 sampling methods) → Edema (all 9 methods) → ... → Pleural Effusion  
  Ratio 0.7: Cardiomegaly (all 9 sampling methods) → Edema (all 9 methods) → ... → Pleural Effusion
  Ratio 0.8: Cardiomegaly (all 9 sampling methods) → Edema (all 9 methods) → ... → Pleural Effusion
  Ratio 0.9: Cardiomegaly (all 9 sampling methods) → Edema (all 9 methods) → ... → Pleural Effusion
  Ratio 1.0: Cardiomegaly (all 9 sampling methods) → Edema (all 9 methods) → ... → Pleural Effusion
  Auto cleaning param optimization (TomekLinks/NCR/SMOTETomek/SMOTEENN small grids)

Total experiments (ratio-grid only): 3 seeds × 6 ratios × 5 pathologies × 9 sampling methods = 810 experiments
EOF
}

print_banner() {
    echo -e "${BOLD}${CYAN}"
    echo "=============================================================================="
    echo "    COMPREHENSIVE SAMPLING METHOD EXPERIMENTS - MASTER ORCHESTRATOR"
    echo "    Master's Thesis Research: Synergistic Effects of Data Sampling & AUC"
    echo "=============================================================================="
    echo -e "${NC}"
}

ensure_default_cuda() {
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
}

print_configuration() {
    local total_experiments=$((${#SEEDS[@]} * ${#RATIOS[@]} * $PATHOLOGIES_COUNT * $SAMPLING_METHODS_COUNT))
    
    echo -e "${BOLD}EXPERIMENTAL CONFIGURATION:${NC}"
    echo -e "Seeds: ${GREEN}${SEEDS[*]}${NC} (${GREEN}${#SEEDS[@]}${NC} seeds)"
    echo -e "Ratios: ${GREEN}${RATIOS[*]}${NC} (${GREEN}${#RATIOS[@]}${NC} ratios)"
    echo -e "Pathologies: ${GREEN}$PATHOLOGIES_COUNT${NC} pathologies"
    echo -e "Sampling methods: ${GREEN}$SAMPLING_METHODS_COUNT${NC} methods per pathology"
    echo -e "Total experiments: ${GREEN}$total_experiments${NC}"
    echo -e "Resume mode: ${GREEN}$RESUME_MODE${NC}"
    echo -e "Dry run mode: ${GREEN}$DRY_RUN${NC}"
    echo ""
}

print_section() {
    echo -e "\n${BOLD}${YELLOW}>>> $1${NC}"
}

print_seed_header() {
    local seed=$1
    local seed_num=$2
    local total_seeds=$3
    echo -e "\n${BOLD}${PURPLE}================================================================${NC}"
    echo -e "${BOLD}${PURPLE}  SEED ITERATION $seed_num/$total_seeds: SEED = $seed${NC}"
    echo -e "${BOLD}${PURPLE}================================================================${NC}"
}

print_ratio_header() {
    local ratio=$1
    local ratio_num=$2
    local total_ratios=$3
    echo -e "\n${BOLD}${BLUE}  ------ RATIO $ratio_num/$total_ratios: $ratio ------${NC}"
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

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parse_arguments() {
    # Set defaults
    SEEDS=("${DEFAULT_SEEDS[@]}")
    RATIOS=("${DEFAULT_RATIOS[@]}")
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --resume)
                RESUME_MODE=true
                shift
                ;;
            --seeds)
                if [[ -n "${2:-}" ]] && [[ ! "$2" =~ ^-- ]]; then
                    IFS=',' read -ra SEEDS <<< "$2"
                    shift 2
                else
                    print_error "--seeds requires a comma-separated list of seeds"
                    exit 1
                fi
                ;;
            --ratios)
                if [[ -n "${2:-}" ]] && [[ ! "$2" =~ ^-- ]]; then
                    IFS=',' read -ra RATIOS <<< "$2"
                    shift 2
                else
                    print_error "--ratios requires a comma-separated list of ratios"
                    exit 1
                fi
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate seeds
    for seed in "${SEEDS[@]}"; do
        if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
            print_error "Invalid seed: $seed (must be positive integer)"
            exit 1
        fi
    done
    
    # Validate ratios
    for ratio in "${RATIOS[@]}"; do
        if [[ ! "$ratio" =~ ^(auto|[0-9]+:[0-9]+|[0-9]*\.?[0-9]+)$ ]]; then
            print_error "Invalid ratio format: $ratio"
            print_error "Valid formats: 'auto', '1:2', '1:3', '0.5', etc."
            exit 1
        fi
    done
}

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================

check_prerequisites() {
    print_section "Checking prerequisites"
    
    # Check working directory
    if [[ ! -f "src/pipelines/pretrain.py" ]] || [[ ! -f "src/pipelines/finetune.py" ]]; then
        print_error "This script must be run from the project root directory"
        print_error "Expected files not found: src/pipelines/pretrain.py, src/pipelines/finetune.py"
        exit 1
    fi
    print_success "Working directory verified"
    
    # Check ratio-pathology script
    if [[ ! -x "$RATIO_PATHOLOGY_SCRIPT" ]]; then
        print_error "Ratio-pathology script not found or not executable: $RATIO_PATHOLOGY_SCRIPT"
        exit 1
    fi
    print_success "Ratio-pathology script found"
    
    # Check experiment tracker
    if [[ ! -f "$TRACKER_SCRIPT" ]]; then
        print_error "Experiment tracker not found: $TRACKER_SCRIPT"
        exit 1
    fi
    print_success "Experiment tracker found"
    
    # Check Python environment
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please ensure Python is installed and activated."
        exit 1
    fi
    
    # Check pandas availability for tracker
    if ! python -c "import pandas" &> /dev/null; then
        print_error "pandas not available. Required for experiment tracking."
        exit 1
    fi
    print_success "Python environment ready"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    print_success "Results directory ready"
}

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

show_overall_progress() {
    print_section "Current experiment progress"
    
    python -c "
import sys
sys.path.append('src')
from utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
print(tracker.generate_summary_report())

# Show pathology breakdown
print('\nPATHOLOGY PROGRESS BREAKDOWN:')
pathology_stats = tracker.get_pathology_progress()
for pathology_id, stats in pathology_stats.items():
    progress_bar = '█' * int(stats['progress'] / 5) + '░' * (20 - int(stats['progress'] / 5))
    print(f'{stats[\"name\"]:15} [{progress_bar}] {stats[\"progress\"]:6.1f}% ({stats[\"completed\"]:3d}/{stats[\"total\"]:3d})')
"
}

get_pending_experiments_for_seed_ratio() {
    local seed=$1
    local ratio=$2
    
    # Count pending experiments for this seed/ratio combination
    python -c "
import sys
sys.path.append('src')
from utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
pending = tracker.get_pending_experiments()
count = sum(1 for s, r, p, m in pending if s == $seed and r == '$ratio')
print(count)
"
}

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

run_seed_iteration() {
    local seed=$1
    local seed_num=$2
    local total_seeds=$3
    
    print_seed_header "$seed" "$seed_num" "$total_seeds"
    
    local seed_start_time=$(date +%s)
    local seed_completed=0
    local seed_failed=0
    
    # Run all ratios for this seed
    local ratio_num=1
    for ratio in "${RATIOS[@]}"; do
        print_ratio_header "$ratio" "$ratio_num" "${#RATIOS[@]}"
        
        # Check if this seed/ratio combination needs work
        local pending_count
        pending_count=$(get_pending_experiments_for_seed_ratio "$seed" "$ratio")
        
        if [[ $pending_count -eq 0 ]] && [[ "$RESUME_MODE" == true ]]; then
            print_info "All experiments for seed $seed, ratio $ratio completed. Skipping..."
        else
            print_info "Running experiments: seed=$seed, ratio=$ratio (pending: $pending_count)"
            
            if [[ "$DRY_RUN" == true ]]; then
                print_info "[DRY RUN] Would execute: $RATIO_PATHOLOGY_SCRIPT '$ratio' $seed"
            else
                local ratio_start_time=$(date +%s)
                
                # Execute ratio-pathology experiments
                if ./"$RATIO_PATHOLOGY_SCRIPT" "$ratio" "$seed"; then
                    local ratio_end_time=$(date +%s)
                    local ratio_duration=$((ratio_end_time - ratio_start_time))
                    local ratio_minutes=$((ratio_duration / 60))
                    print_success "Ratio $ratio completed successfully (${ratio_minutes}m)"
                    
                    # Count successful experiments (rough estimate)
                    ((seed_completed += 45))  # 5 pathologies × 9 methods = 45 per ratio
                else
                    print_error "Ratio $ratio failed for seed $seed"
                    ((seed_failed += 45))
                fi
            fi
        fi
        
        ((ratio_num++))
    done
    
    # After completing all ratios for this seed, run ratio-independent param optimization
    if [[ "$DRY_RUN" == true ]]; then
        print_info "[DRY RUN] Would execute: $AUTO_OPT_SCRIPT --seed $seed"
    else
        print_section "Running ratio-independent parameter optimization (auto) for seed $seed"
        if ./$AUTO_OPT_SCRIPT --seed "$seed"; then
            print_success "Auto cleaning param optimization completed for seed $seed"
        else
            print_warning "Auto cleaning param optimization encountered errors for seed $seed"
        fi
    fi
    
    local seed_end_time=$(date +%s)
    local seed_duration=$((seed_end_time - seed_start_time))
    local seed_hours=$((seed_duration / 3600))
    local seed_minutes=$(((seed_duration % 3600) / 60))
    
    if [[ "$DRY_RUN" == true ]]; then
        print_success "Seed $seed [DRY RUN] completed"
    else
        print_success "Seed $seed iteration completed in ${seed_hours}h ${seed_minutes}m"
        print_info "Seed $seed results: ~$seed_completed completed, ~$seed_failed failed"
    fi
}

run_comprehensive_experiments() {
    print_section "Starting comprehensive experiments"
    
    local overall_start_time=$(date +%s)
    
    # Show initial progress if resuming
    if [[ "$RESUME_MODE" == true ]] && [[ "$DRY_RUN" == false ]]; then
        show_overall_progress
    fi
    
    # Run experiments for each seed
    local seed_num=1
    for seed in "${SEEDS[@]}"; do
        run_seed_iteration "$seed" "$seed_num" "${#SEEDS[@]}"
        ((seed_num++))
        
        # Show progress update after each seed (except in dry run mode)
        if [[ "$DRY_RUN" == false ]]; then
            echo ""
            show_overall_progress
            echo ""
        fi
    done
    
    local overall_end_time=$(date +%s)
    local overall_duration=$((overall_end_time - overall_start_time))
    local overall_hours=$((overall_duration / 3600))
    local overall_minutes=$(((overall_duration % 3600) / 60))
    
    if [[ "$DRY_RUN" == true ]]; then
        print_success "Comprehensive experiments [DRY RUN] completed"
    else
        print_success "ALL comprehensive experiments completed in ${overall_hours}h ${overall_minutes}m"
    fi
}

# =============================================================================
# FINAL RESULTS AND ANALYSIS
# =============================================================================

generate_final_report() {
    if [[ "$DRY_RUN" == true ]]; then
        print_section "Final Report (Dry Run Mode)"
        print_info "This was a dry run. No experiments were actually executed."
        return
    fi
    
    print_section "Generating final comprehensive report"
    
    # Export comprehensive results
    local export_path
    export_path=$(python -c "
import sys
sys.path.append('src')
from utils.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()
path = tracker.export_results_summary()
print(path)
")
    
    print_success "Comprehensive results exported to: $export_path"
    
    # Show final statistics
    echo ""
    show_overall_progress
    
    echo ""
    print_info "=== NEXT STEPS ==="
    print_info "1. Review results: $export_path"
    print_info "2. Run statistical analysis: python src/analysis/statistical_analysis.py"
    print_info "3. Check individual pathology logs in: $RESULTS_DIR/"
    print_info "4. Use tracker for detailed analysis: python $TRACKER_SCRIPT --status"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Print banner
    print_banner
    ensure_default_cuda
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Show configuration
    print_configuration
    
    if [[ "$DRY_RUN" == false ]]; then
        # Check prerequisites (skip for dry run)
        check_prerequisites
    fi
    
    # Show current progress if resuming
    if [[ "$RESUME_MODE" == true ]] && [[ "$DRY_RUN" == false ]]; then
        show_overall_progress
        echo ""
        print_warning "Resume mode enabled. Completed experiments will be skipped."
        echo ""
    fi
    
    # Execute comprehensive experiments
    run_comprehensive_experiments
    
    # Generate final report and analysis
    generate_final_report
    
    print_section "Master Orchestration Complete"
    if [[ "$DRY_RUN" == true ]]; then
        print_success "Dry run completed successfully!"
        print_info "Remove --dry-run flag to execute actual experiments"
    else
        print_success "All comprehensive sampling method experiments completed!"
        print_info "Review the exported results and proceed with thesis analysis"
    fi
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Trap Ctrl+C and provide clean exit
trap 'echo -e "\n${YELLOW}Master orchestration interrupted by user${NC}"; exit 130' INT

# Run main function with all arguments
main "$@"
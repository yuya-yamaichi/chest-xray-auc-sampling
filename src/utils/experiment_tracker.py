#!/usr/bin/env python3
"""
Experiment Tracker Utility for Master's Thesis Research
Tracks experimental progress and manages resumability for comprehensive sampling experiments.
"""

import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path


class ExperimentTracker:
    """
    Tracks experimental progress for comprehensive sampling method experiments.
    Supports resumability and progress monitoring.
    """
    
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        self.tracker_file = self.base_results_dir / "experiment_tracker.json"
        self.progress_file = self.base_results_dir / "experiment_progress.csv"
        
        # Pathology mapping
        self.pathologies = {
            0: "Cardiomegaly",
            1: "Edema", 
            2: "Consolidation",
            3: "Atelectasis",
            4: "Pleural_Effusion"
        }
        
        # Available sampling methods
        self.sampling_methods = [
            "none",
            "RandomOverSampler",
            "SMOTE",
            "ADASYN",
            "RandomUnderSampler",
            "TomekLinks",
            "NeighbourhoodCleaningRule",
            "SMOTETomek",
            "SMOTEENN"
        ]
        
        # Create results directory
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Initialize tracker
        self.tracker_data = self._load_tracker()
        # Migration to consolidate 'none' sampler ratio keys
        self._migrate_none_ratio_keys()
        # Ensure configuration reflects current default ratios
        self._ensure_configuration_defaults()
    
    def _load_tracker(self) -> Dict:
        """Load existing tracker data or initialize new tracker."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return {
            "experiment_id": f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started": datetime.now().isoformat(),
            "completed_experiments": {},
            "failed_experiments": {},
            "skipped_experiments": {},
            "configuration": {
                "ratios": ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                "pathologies": list(self.pathologies.keys()),
                "sampling_methods": self.sampling_methods,
                "seeds": [42, 123, 456]
            }
        }
    
    def _save_tracker(self):
        """Save tracker data to file."""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tracker_data, f, indent=2)
    
    def _normalize_ratio_for_sampler(self, ratio: str, sampler: str) -> str:
        """Normalize ratio for samplers that do not use ratio (e.g., 'none')."""
        if sampler == "none":
            return "N/A"
        return ratio

    def get_experiment_key(self, seed: int, ratio: str, pathology_id: int, sampler: str) -> str:
        """Generate unique key for experiment with ratio normalization for 'none'."""
        normalized_ratio = self._normalize_ratio_for_sampler(ratio, sampler)
        return f"{seed}_{normalized_ratio}_{pathology_id}_{sampler}"
    
    def is_experiment_completed(self, seed: int, ratio: str, pathology_id: int, sampler: str) -> bool:
        """Check if specific experiment has been completed."""
        key = self.get_experiment_key(seed, ratio, pathology_id, sampler)
        return key in self.tracker_data["completed_experiments"]
    
    def is_experiment_failed(self, seed: int, ratio: str, pathology_id: int, sampler: str) -> bool:
        """Check if specific experiment has failed."""
        key = self.get_experiment_key(seed, ratio, pathology_id, sampler)
        return key in self.tracker_data["failed_experiments"]

    def is_experiment_skipped(self, seed: int, ratio: str, pathology_id: int, sampler: str) -> bool:
        """Check if specific experiment has been skipped."""
        key = self.get_experiment_key(seed, ratio, pathology_id, sampler)
        return key in self.tracker_data.get("skipped_experiments", {})
    
    def mark_experiment_completed(self, seed: int, ratio: str, pathology_id: int, sampler: str, 
                                 auc_score: float, execution_time: float, model_path: str = None):
        """Mark experiment as completed with results."""
        normalized_ratio = self._normalize_ratio_for_sampler(ratio, sampler)
        key = self.get_experiment_key(seed, normalized_ratio, pathology_id, sampler)
        
        self.tracker_data["completed_experiments"][key] = {
            "seed": seed,
            "ratio": normalized_ratio,
            "pathology_id": pathology_id,
            "pathology_name": self.pathologies[pathology_id],
            "sampler": sampler,
            "auc_score": auc_score,
            "execution_time": execution_time,
            "model_path": model_path,
            "completed_at": datetime.now().isoformat()
        }
        
        # Remove from failed if it was previously failed
        if key in self.tracker_data["failed_experiments"]:
            del self.tracker_data["failed_experiments"][key]
        
        self._save_tracker()
        self._update_progress_log()
    
    def mark_experiment_failed(self, seed: int, ratio: str, pathology_id: int, sampler: str, 
                              error_message: str = "Unknown error"):
        """Mark experiment as failed."""
        normalized_ratio = self._normalize_ratio_for_sampler(ratio, sampler)
        key = self.get_experiment_key(seed, normalized_ratio, pathology_id, sampler)
        
        self.tracker_data["failed_experiments"][key] = {
            "seed": seed,
            "ratio": normalized_ratio,
            "pathology_id": pathology_id,
            "pathology_name": self.pathologies[pathology_id],
            "sampler": sampler,
            "error_message": error_message,
            "failed_at": datetime.now().isoformat()
        }
        
        self._save_tracker()
        self._update_progress_log()

    def mark_experiment_skipped(self, seed: int, ratio: str, pathology_id: int, sampler: str,
                                reason: str = "infeasible") -> None:
        """Mark experiment as skipped with reason (e.g., infeasible)."""
        normalized_ratio = self._normalize_ratio_for_sampler(ratio, sampler)
        key = self.get_experiment_key(seed, normalized_ratio, pathology_id, sampler)

        # Ensure key spaces exist after legacy trackers without skipped map
        if "skipped_experiments" not in self.tracker_data:
            self.tracker_data["skipped_experiments"] = {}

        self.tracker_data["skipped_experiments"][key] = {
            "seed": seed,
            "ratio": normalized_ratio,
            "pathology_id": pathology_id,
            "pathology_name": self.pathologies[pathology_id],
            "sampler": sampler,
            "reason": reason,
            "feasibility": "infeasible" if reason else "unknown",
            "skipped_at": datetime.now().isoformat()
        }

        # Remove from failed/completed if present under same key
        self.tracker_data["failed_experiments"].pop(key, None)
        self.tracker_data["completed_experiments"].pop(key, None)

        self._save_tracker()
        self._update_progress_log()

    def _migrate_none_ratio_keys(self) -> None:
        """Consolidate existing tracker entries where sampler=='none' but ratio is not normalized.
        Ensures past records are keyed with ratio 'N/A' and avoids duplicates across ratios.
        """
        changed = False
        for section in ["completed_experiments", "failed_experiments"]:
            original_items = list(self.tracker_data[section].items())
            for key, data in original_items:
                sampler = data.get("sampler")
                ratio = data.get("ratio")
                if sampler == "none":
                    normalized_ratio = self._normalize_ratio_for_sampler(ratio, sampler)
                    new_key = self.get_experiment_key(data["seed"], normalized_ratio, data["pathology_id"], sampler)
                    if new_key != key:
                        # Move entry under new key and update ratio field
                        entry = self.tracker_data[section].pop(key)
                        entry["ratio"] = normalized_ratio
                        # If target key already exists, keep the one with later timestamp
                        if new_key in self.tracker_data[section]:
                            # Prefer the entry with later timestamp field
                            ts_field = "completed_at" if section == "completed_experiments" else "failed_at"
                            existing = self.tracker_data[section][new_key]
                            if entry.get(ts_field, "") > existing.get(ts_field, ""):
                                self.tracker_data[section][new_key] = entry
                        else:
                            self.tracker_data[section][new_key] = entry
                        changed = True
        if changed:
            self._save_tracker()
            self._update_progress_log()

    def _ensure_configuration_defaults(self) -> None:
        """Ensure tracker configuration uses the current default ratio set.
        If an older ratio set is present, replace it with the new one to keep
        pending/total counts consistent with the orchestrator.
        """
        desired_ratios = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        cfg = self.tracker_data.get("configuration", {})
        current = cfg.get("ratios")
        if current != desired_ratios:
            cfg["ratios"] = desired_ratios
            self.tracker_data["configuration"] = cfg
            self._save_tracker()
            self._update_progress_log()
    
    def _update_progress_log(self):
        """Update progress CSV file with unified schema including skipped and metadata."""
        progress_data = []

        # Add completed experiments
        for key, data in self.tracker_data["completed_experiments"].items():
            progress_data.append({
                "experiment_key": key,
                "status": "completed",
                "seed": data["seed"],
                "ratio": data["ratio"],
                "pathology_id": data["pathology_id"],
                "pathology_name": data["pathology_name"],
                "sampler": data["sampler"],
                "auc_score": data.get("auc_score", None),
                "execution_time": data.get("execution_time", None),
                "timestamp": data["completed_at"],
                # Optional metadata
                "sampling_applied": data.get("sampling_applied", None),
                "feasibility": data.get("feasibility", None),
                "reason": data.get("reason", None)
            })

        # Add failed experiments
        for key, data in self.tracker_data["failed_experiments"].items():
            progress_data.append({
                "experiment_key": key,
                "status": "failed",
                "seed": data["seed"],
                "ratio": data["ratio"],
                "pathology_id": data["pathology_id"],
                "pathology_name": data["pathology_name"],
                "sampler": data["sampler"],
                "auc_score": None,
                "execution_time": None,
                "timestamp": data["failed_at"],
                "error_message": data.get("error_message", "")
            })

        # Add skipped experiments
        for key, data in self.tracker_data.get("skipped_experiments", {}).items():
            progress_data.append({
                "experiment_key": key,
                "status": "skipped",
                "seed": data["seed"],
                "ratio": data["ratio"],
                "pathology_id": data["pathology_id"],
                "pathology_name": data["pathology_name"],
                "sampler": data["sampler"],
                "auc_score": None,
                "execution_time": None,
                "timestamp": data["skipped_at"],
                "reason": data.get("reason", ""),
                "sampling_applied": False,
                "feasibility": data.get("feasibility", "infeasible")
            })

        if progress_data:
            df = pd.DataFrame(progress_data)
            df.to_csv(self.progress_file, index=False)
    
    def get_total_experiments(self) -> int:
        """Calculate total number of experiments."""
        config = self.tracker_data["configuration"]
        num_seeds = len(config["seeds"])
        num_ratios = len(config["ratios"])
        num_pathologies = len(config["pathologies"])
        num_methods = len(config["sampling_methods"])  # includes 'none'
        # One method ('none') is ratio-independent
        ratio_independent = 1 if "none" in config["sampling_methods"] else 0
        ratio_dependent = num_methods - ratio_independent
        per_pathology = ratio_independent + num_ratios * ratio_dependent
        return num_seeds * num_pathologies * per_pathology
    
    def get_completed_count(self) -> int:
        """Get number of completed experiments."""
        return len(self.tracker_data["completed_experiments"])
    
    def get_failed_count(self) -> int:
        """Get number of failed experiments."""
        return len(self.tracker_data["failed_experiments"])
    
    def get_skipped_count(self) -> int:
        """Get number of skipped experiments."""
        return len(self.tracker_data.get("skipped_experiments", {}))
    
    def get_remaining_count(self) -> int:
        """Get number of remaining experiments."""
        return self.get_total_experiments() - self.get_completed_count() - self.get_failed_count() - self.get_skipped_count()
    
    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        total = self.get_total_experiments()
        if total == 0:
            return 0.0
        denom = total - self.get_skipped_count()
        if denom <= 0:
            return 100.0
        return (self.get_completed_count() / denom) * 100
    
    def get_pending_experiments(self) -> List[Tuple[int, str, int, str]]:
        """Get list of pending experiments."""
        pending = []
        config = self.tracker_data["configuration"]
        
        for seed in config["seeds"]:
            for ratio in config["ratios"]:
                for pathology_id in config["pathologies"]:
                    for sampler in config["sampling_methods"]:
                        if not (self.is_experiment_completed(seed, ratio, pathology_id, sampler) or 
                               self.is_experiment_failed(seed, ratio, pathology_id, sampler)):
                            pending.append((seed, ratio, pathology_id, sampler))
        
        return pending
    
    def get_failed_experiments(self) -> List[Tuple[int, str, int, str]]:
        """Get list of failed experiments that can be retried."""
        failed = []
        for key, data in self.tracker_data["failed_experiments"].items():
            failed.append((data["seed"], data["ratio"], data["pathology_id"], data["sampler"]))
        return failed
    
    def generate_summary_report(self) -> str:
        """Generate a detailed summary report."""
        total = self.get_total_experiments()
        completed = self.get_completed_count()
        failed = self.get_failed_count()
        remaining = self.get_remaining_count()
        progress = self.get_progress_percentage()
        
        report = f"""
=== COMPREHENSIVE SAMPLING EXPERIMENTS PROGRESS ===
Experiment ID: {self.tracker_data['experiment_id']}
Started: {self.tracker_data['started']}

PROGRESS SUMMARY:
- Total Experiments: {total}
- Completed: {completed}
- Failed: {failed}
- Skipped: {self.get_skipped_count()}
- Remaining: {remaining}
- Progress: {progress:.1f}%

CONFIGURATION:
- Seeds: {self.tracker_data['configuration']['seeds']}
- Ratios: {self.tracker_data['configuration']['ratios']}
- Pathologies: {len(self.tracker_data['configuration']['pathologies'])} pathologies
- Sampling Methods: {len(self.tracker_data['configuration']['sampling_methods'])} methods

"""
        
        if completed > 0:
            report += "RECENT COMPLETIONS:\n"
            completed_exp = list(self.tracker_data["completed_experiments"].values())
            completed_exp.sort(key=lambda x: x["completed_at"], reverse=True)
            for exp in completed_exp[:5]:  # Show last 5
                report += f"- {exp['pathology_name']} | {exp['sampler']} | ratio={exp['ratio']} | seed={exp['seed']} | AUC={exp['auc_score']:.3f}\n"
        
        if failed > 0:
            report += f"\nFAILED EXPERIMENTS: {failed}\n"
            failed_exp = list(self.tracker_data["failed_experiments"].values())
            failed_exp.sort(key=lambda x: x["failed_at"], reverse=True)
            for exp in failed_exp[:3]:  # Show last 3 failures
                report += f"- {exp['pathology_name']} | {exp['sampler']} | ratio={exp['ratio']} | seed={exp['seed']} | Error: {exp['error_message']}\n"
        
        return report
    
    def get_pathology_progress(self) -> Dict[int, Dict]:
        """Get progress breakdown by pathology."""
        pathology_stats = {}
        
        for pathology_id in self.pathologies.keys():
            completed = sum(1 for exp in self.tracker_data["completed_experiments"].values() 
                          if exp["pathology_id"] == pathology_id)
            failed = sum(1 for exp in self.tracker_data["failed_experiments"].values() 
                        if exp["pathology_id"] == pathology_id)
            
            # Total experiments per pathology
            config = self.tracker_data["configuration"]
            num_seeds = len(config["seeds"]) 
            num_ratios = len(config["ratios"])
            num_methods = len(config["sampling_methods"])  # includes 'none'
            ratio_independent = 1 if "none" in config["sampling_methods"] else 0
            ratio_dependent = num_methods - ratio_independent
            total_per_pathology = num_seeds * (ratio_independent + num_ratios * ratio_dependent)
            
            pathology_stats[pathology_id] = {
                "name": self.pathologies[pathology_id],
                "completed": completed,
                "failed": failed,
                "total": total_per_pathology,
                "remaining": total_per_pathology - completed - failed,
                "progress": (completed / total_per_pathology) * 100 if total_per_pathology > 0 else 0
            }
        
        return pathology_stats
    
    def reset_failed_experiments(self):
        """Reset failed experiments to allow retry."""
        self.tracker_data["failed_experiments"] = {}
        self._save_tracker()
        self._update_progress_log()

    def update_completed_with_sampling_metadata(self, seed: int, ratio: str, pathology_id: int, sampler: str) -> bool:
        """Augment a completed experiment with sampling metadata from pathology CSV logs.

        Returns True if augmentation succeeded, False otherwise.
        """
        try:
            normalized_ratio = self._normalize_ratio_for_sampler(ratio, sampler)
            key = self.get_experiment_key(seed, normalized_ratio, pathology_id, sampler)
            if key not in self.tracker_data["completed_experiments"]:
                return False

            # Determine pathology name and log path
            pathology_name = self.pathologies[pathology_id]
            # Support both underscore and space variants used across the codebase
            candidates = [
                self.base_results_dir / f"training_log_{pathology_name.lower()}.csv",
                self.base_results_dir / f"training_log_{pathology_name.lower().replace('_', ' ')}.csv",
                self.base_results_dir / f"training_log_{pathology_name.lower().replace(' ', '_')}.csv",
            ]
            log_path = None
            for p in candidates:
                if p.exists():
                    log_path = p
                    break
            if log_path is None:
                return False

            df = pd.read_csv(log_path)
            # Filter matching rows
            df_match = df[(df.get('seed') == seed) &
                          (df.get('sampler') == sampler) &
                          (df.get('class_id') == pathology_id)]
            if df_match.empty:
                return False

            # Use the last matching entry
            row = df_match.iloc[-1]
            sampling_strategy_used = str(row.get('sampling_strategy_used', 'unknown'))
            sampling_applied = sampling_strategy_used not in ['none', 'failed', 'unknown']

            # Update tracker record
            self.tracker_data["completed_experiments"][key]["sampling_applied"] = bool(sampling_applied)
            # Keep feasibility unknown at completion stage
            if "feasibility" not in self.tracker_data["completed_experiments"][key]:
                self.tracker_data["completed_experiments"][key]["feasibility"] = "unknown"

            self._save_tracker()
            self._update_progress_log()
            return True
        except Exception:
            return False
    
    def export_results_summary(self, output_path: str = None) -> str:
        """Export comprehensive results summary to CSV."""
        if output_path is None:
            output_path = self.base_results_dir / f"comprehensive_results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        results = []
        for exp in self.tracker_data["completed_experiments"].values():
            results.append({
                "seed": exp["seed"],
                "ratio": exp["ratio"],
                "pathology_id": exp["pathology_id"],
                "pathology_name": exp["pathology_name"],
                "sampler": exp["sampler"],
                "auc_score": exp["auc_score"],
                "execution_time": exp["execution_time"],
                "completed_at": exp["completed_at"]
            })
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            return str(output_path)
        
        return "No completed experiments to export"


def main():
    """Command line interface for experiment tracker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Tracker Utility")
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--export', action='store_true', help='Export results summary')
    parser.add_argument('--reset-failed', action='store_true', help='Reset failed experiments')
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker()
    
    if args.status:
        print(tracker.generate_summary_report())
    
    if args.export:
        output_path = tracker.export_results_summary()
        print(f"Results exported to: {output_path}")
    
    if args.reset_failed:
        tracker.reset_failed_experiments()
        print("Failed experiments reset. They can now be retried.")


if __name__ == "__main__":
    main()
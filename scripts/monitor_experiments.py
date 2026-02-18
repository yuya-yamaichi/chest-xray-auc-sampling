#!/usr/bin/env python3
"""
Real-time Experiment Progress Monitor
Master's Thesis Research: Synergistic Effects of Sampling Methods and AUC Optimization

This script provides real-time monitoring and progress visualization for comprehensive experiments.
"""

import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.append('src')

try:
    from utils.experiment_tracker import ExperimentTracker
except ImportError:
    print("Error: Cannot import experiment tracker. Please ensure you're in the project root directory.")
    sys.exit(1)


class ExperimentMonitor:
    """Real-time monitor for comprehensive sampling experiments."""
    
    def __init__(self, refresh_interval: int = 30):
        self.tracker = ExperimentTracker()
        self.refresh_interval = refresh_interval
        self.start_time = datetime.now()
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def format_time_elapsed(self, seconds: int) -> str:
        """Format elapsed time in human-readable format."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def estimate_time_remaining(self) -> str:
        """Estimate time remaining based on current progress."""
        total_experiments = self.tracker.get_total_experiments()
        completed_experiments = self.tracker.get_completed_count()
        
        if completed_experiments == 0:
            return "Calculating..."
        
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        rate = completed_experiments / elapsed_seconds  # experiments per second
        
        remaining_experiments = total_experiments - completed_experiments - self.tracker.get_failed_count()
        
        if rate > 0 and remaining_experiments > 0:
            remaining_seconds = remaining_experiments / rate
            return self.format_time_elapsed(int(remaining_seconds))
        
        return "Unknown"
    
    def get_progress_bar(self, percentage: float, width: int = 40) -> str:
        """Generate a visual progress bar."""
        filled_width = int(width * percentage / 100)
        bar = '█' * filled_width + '░' * (width - filled_width)
        return f"[{bar}] {percentage:5.1f}%"
    
    def format_pathology_table(self) -> str:
        """Format pathology progress as a table."""
        pathology_stats = self.tracker.get_pathology_progress()
        
        table = "\n┌─────────────────┬─────────┬─────────┬─────────┬──────────────────────────────────────────┐\n"
        table += "│ Pathology       │ Complete│ Failed  │ Remain  │ Progress                                 │\n"
        table += "├─────────────────┼─────────┼─────────┼─────────┼──────────────────────────────────────────┤\n"
        
        for pathology_id in sorted(pathology_stats.keys()):
            stats = pathology_stats[pathology_id]
            name = stats['name'][:15].ljust(15)
            completed = f"{stats['completed']:3d}".rjust(7)
            failed = f"{stats['failed']:3d}".rjust(7)
            remaining = f"{stats['remaining']:3d}".rjust(7)
            progress_bar = self.get_progress_bar(stats['progress'], 40)
            
            table += f"│ {name} │ {completed} │ {failed} │ {remaining} │ {progress_bar} │\n"
        
        table += "└─────────────────┴─────────┴─────────┴─────────┴──────────────────────────────────────────┘\n"
        
        return table
    
    def format_recent_completions(self, limit: int = 5) -> str:
        """Format recent completions."""
        completed_experiments = self.tracker.tracker_data.get("completed_experiments", {})
        
        if not completed_experiments:
            return "No completed experiments yet.\n"
        
        # Sort by completion time
        sorted_experiments = sorted(
            completed_experiments.values(),
            key=lambda x: x["completed_at"],
            reverse=True
        )
        
        result = f"\n🎯 RECENT COMPLETIONS (last {limit}):\n"
        result += "─" * 60 + "\n"
        
        for exp in sorted_experiments[:limit]:
            timestamp = exp["completed_at"][:19].replace("T", " ")
            pathology = exp["pathology_name"][:12].ljust(12)
            sampler = exp["sampler"][:15].ljust(15)
            auc = f"{exp['auc_score']:.3f}"
            ratio = exp["ratio"]
            seed = exp["seed"]
            
            result += f"{timestamp} │ {pathology} │ {sampler} │ R:{ratio:>4} │ S:{seed:>4} │ AUC:{auc}\n"
        
        return result
    
    def format_failures(self, limit: int = 3) -> str:
        """Format recent failures."""
        failed_experiments = self.tracker.tracker_data.get("failed_experiments", {})
        
        if not failed_experiments:
            return ""
        
        # Sort by failure time
        sorted_failures = sorted(
            failed_experiments.values(),
            key=lambda x: x["failed_at"],
            reverse=True
        )
        
        result = f"\n❌ RECENT FAILURES (last {limit}):\n"
        result += "─" * 60 + "\n"
        
        for exp in sorted_failures[:limit]:
            timestamp = exp["failed_at"][:19].replace("T", " ")
            pathology = exp["pathology_name"][:12].ljust(12)
            sampler = exp["sampler"][:15].ljust(15)
            ratio = exp["ratio"]
            seed = exp["seed"]
            error = exp.get("error_message", "Unknown error")[:30]
            
            result += f"{timestamp} │ {pathology} │ {sampler} │ R:{ratio:>4} │ S:{seed:>4} │ {error}\n"
        
        return result
    
    def get_experiment_rate(self) -> str:
        """Calculate current experiment completion rate."""
        completed = self.tracker.get_completed_count()
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        
        if elapsed_seconds > 0:
            rate_per_hour = (completed / elapsed_seconds) * 3600
            return f"{rate_per_hour:.1f} exp/hour"
        
        return "0.0 exp/hour"
    
    def display_status(self):
        """Display comprehensive experiment status."""
        # Get current statistics
        total = self.tracker.get_total_experiments()
        completed = self.tracker.get_completed_count()
        failed = self.tracker.get_failed_count()
        remaining = self.tracker.get_remaining_count()
        progress = self.tracker.get_progress_percentage()
        
        # Calculate times
        elapsed = (datetime.now() - self.start_time).total_seconds()
        elapsed_formatted = self.format_time_elapsed(int(elapsed))
        eta = self.estimate_time_remaining()
        rate = self.get_experiment_rate()
        
        # Clear screen and display header
        self.clear_screen()
        print("╔" + "═" * 78 + "╗")
        print("║" + " COMPREHENSIVE SAMPLING EXPERIMENTS - REAL-TIME MONITOR".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Overall progress
        print(f"\n🔄 OVERALL PROGRESS:")
        print(f"   {self.get_progress_bar(progress, 60)}")
        print(f"   Total: {total} │ Completed: {completed:3d} │ Failed: {failed:3d} │ Remaining: {remaining:3d}")
        
        # Time information
        print(f"\n⏱️  TIME INFORMATION:")
        print(f"   Elapsed: {elapsed_formatted} │ ETA: {eta} │ Rate: {rate}")
        print(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Pathology breakdown table
        print(f"\n📊 PATHOLOGY BREAKDOWN:")
        print(self.format_pathology_table())
        
        # Recent completions
        print(self.format_recent_completions())
        
        # Recent failures (if any)
        if failed > 0:
            print(self.format_failures())
        
        # Footer
        print("\n" + "─" * 80)
        print(f"🔄 Auto-refresh every {self.refresh_interval}s │ Press Ctrl+C to exit")
        print("─" * 80)
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring with auto-refresh."""
        print("Starting experiment monitor...")
        print(f"Refresh interval: {self.refresh_interval} seconds")
        print("Press Ctrl+C to exit")
        time.sleep(2)
        
        try:
            while True:
                self.display_status()
                
                # Check if all experiments are done
                if self.tracker.get_remaining_count() == 0:
                    print("\n🎉 ALL EXPERIMENTS COMPLETED! 🎉")
                    break
                
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user")
    
    def run_single_status(self):
        """Display status once without continuous monitoring."""
        self.display_status()
    
    def export_progress_report(self, output_path: str = None) -> str:
        """Export detailed progress report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"results/progress_report_{timestamp}.txt"
        
        # Capture current display
        original_start_time = self.start_time
        self.start_time = datetime.now() - timedelta(seconds=1)  # Avoid zero elapsed time
        
        # Redirect print to capture output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            self.display_status()
        
        report_content = f.getvalue()
        self.start_time = original_start_time  # Restore original
        
        # Write to file
        with open(output_path, 'w') as file:
            file.write(f"Experiment Progress Report\n")
            file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 80 + "\n\n")
            file.write(report_content)
            file.write("\n\nEnd of Report\n")
        
        return output_path


def main():
    """Command line interface for experiment monitor."""
    parser = argparse.ArgumentParser(
        description="Real-time monitor for comprehensive sampling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start real-time monitoring (default 30s refresh)
    python scripts/monitor_experiments.py
    
    # Monitor with custom refresh interval
    python scripts/monitor_experiments.py --refresh 60
    
    # Show status once without continuous monitoring
    python scripts/monitor_experiments.py --once
    
    # Export progress report to file
    python scripts/monitor_experiments.py --export progress_report.txt
        """
    )
    
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Show status once without continuous monitoring')
    parser.add_argument('--export', type=str,
                       help='Export progress report to specified file')
    
    args = parser.parse_args()
    
    # Validate refresh interval
    if args.refresh < 5:
        print("Warning: Minimum refresh interval is 5 seconds")
        args.refresh = 5
    
    try:
        monitor = ExperimentMonitor(refresh_interval=args.refresh)
        
        if args.export:
            report_path = monitor.export_progress_report(args.export)
            print(f"Progress report exported to: {report_path}")
        elif args.once:
            monitor.run_single_status()
        else:
            monitor.run_continuous_monitoring()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
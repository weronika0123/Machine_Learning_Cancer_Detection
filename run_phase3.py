"""
Phase 3 Master Script
Orchestrates all Phase 3 tasks: hyperparameter search, analysis, and reporting.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time


def print_banner(text):
    """Print a formatted banner."""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print_banner(description)
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path.cwd()
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed/60:.1f} minutes!")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{description} failed after {elapsed/60:.1f} minutes!")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n{description} interrupted by user!")
        return False


def check_prerequisites():
    """Check if required files exist."""
    print_banner("CHECKING PREREQUISITES")
    
    required_files = [
        'src/main.py',
        'src/hyperparameter_search.py',
        'src/analyze_experiments.py',
        'src/demo.py',
        'src/data_sources/liquid_biopsy_data.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"{file_path}")
        else:
            print(f"{file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Main orchestration routine."""
    print(f"\n{'#'*80}")
    print(f"#{'':^78}#")
    print(f"#{'PHASE 3 - HYPERPARAMETER SEARCH & ANALYSIS':^78}#")
    print(f"#{'':^78}#")
    print(f"#{'Deep Neural Networks for Data Analysis':^78}#")
    print(f"#{'Cancer Detection Project':^78}#")
    print(f"#{'':^78}#")
    print(f"{'#'*80}\n")
    
    start_time = datetime.now()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nMissing required files. Please ensure all files are present.")
        sys.exit(1)
    
    print("\nAll prerequisites met!\n")
    
    # Ask for confirmation
    print("This script will:")
    print("  1. Run 25 hyperparameter search experiments (~4-6 hours)")
    print("  2. Analyze results and generate visualizations")
    print("  3. Create comprehensive reports")
    print("  4. Identify the best model\n")
    
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n Operation cancelled by user.")
        sys.exit(0)
    
    # Step 1: Hyperparameter Search
    success = run_command(
        [sys.executable, 'src/hyperparameter_search.py'],
        "STEP 1/3: HYPERPARAMETER SEARCH"
    )
    
    if not success:
        print("\n Hyperparameter search failed. Stopping execution.")
        sys.exit(1)
    
    # Check if experiments were successful
    experiments_log = Path('experiments') / 'experiments_log.csv'
    if not experiments_log.exists():
        print(f"\n Error: {experiments_log} not found!")
        print("Hyperparameter search did not produce expected output.")
        sys.exit(1)
    
    # Step 2: Analyze Experiments
    success = run_command(
        [sys.executable, 'src/analyze_experiments.py'],
        "STEP 2/3: EXPERIMENT ANALYSIS"
    )
    
    if not success:
        print("\n Warning: Analysis failed, but continuing...")
    
    # Check if analysis outputs exist
    plots_dir = Path('experiments') / 'plots'
    summary_file = Path('experiments') / 'experiments_summary.md'
    
    if plots_dir.exists() and summary_file.exists():
        print("\n Analysis outputs generated successfully!")
    else:
        print("\n Warning: Some analysis outputs may be missing.")
    
    # Step 3: Generate Best Run Report
    print_banner("STEP 3/3: GENERATING BEST RUN REPORT")
    
    best_info_path = Path('experiments') / 'best_run_info.json'
    if best_info_path.exists():
        import json
        with open(best_info_path, 'r', encoding='utf-8') as f:
            best_info = json.load(f)
        
        print("BEST MODEL FOUND:")
        print(f"   Experiment ID: {best_info['experiment_id']}")
        print(f"   AUC ROC: {best_info['auc_roc']:.4f}")
        print(f"   Accuracy: {best_info['accuracy']:.4f}")
        print(f"   F1 Score: {best_info['f1']:.4f}")
        print(f"\n   Configuration:")
        print(f"   - Hidden Layers: {best_info['hidden_layers']}")
        print(f"   - Learning Rate: {best_info['learning_rate']}")
        print(f"   - Dropout Rate: {best_info['dropout_rate']}")
        print(f"   - Batch Size: {best_info['batch_size']}")
        print(f"   - Activation: {best_info['activation']}")
        print(f"   - Epochs: {best_info['epochs']}")
        
        # Generate best run markdown report
        report_path = Path('experiments') / 'best_run_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Best Model Report - Phase 3\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write("## Best Model\n\n")
            f.write(f"**Experiment ID:** {best_info['experiment_id']}\n")
            f.write(f"**Timestamp:** {best_info['timestamp']}\n\n")
            
            f.write("##Performance Metrics\n\n")
            f.write(f"- **AUC ROC:** {best_info['auc_roc']:.4f}\n")
            f.write(f"- **Accuracy:** {best_info['accuracy']:.4f}\n")
            f.write(f"- **F1 Score:** {best_info['f1']:.4f}\n")
            if best_info.get('precision'):
                f.write(f"- **Precision:** {best_info['precision']:.4f}\n")
            if best_info.get('recall'):
                f.write(f"- **Recall:** {best_info['recall']:.4f}\n")
            if best_info.get('auc_pr'):
                f.write(f"- **AUC PR:** {best_info['auc_pr']:.4f}\n")
            
            f.write("\n## Hyperparameters\n\n")
            f.write(f"- **Hidden Layers:** {best_info['hidden_layers']}\n")
            f.write(f"- **Learning Rate:** {best_info['learning_rate']}\n")
            f.write(f"- **Dropout Rate:** {best_info['dropout_rate']}\n")
            f.write(f"- **Batch Size:** {best_info['batch_size']}\n")
            f.write(f"- **Activation Function:** {best_info['activation']}\n")
            f.write(f"- **Epochs:** {best_info['epochs']}\n")
            
            f.write("\n## Model Artifacts\n\n")
            f.write(f"The trained model and associated files can be found in:\n")
            f.write(f"```\noutput/.../{{timestamp}}/\n```\n\n")
            f.write("Files include:\n")
            f.write("- `model_best.keras` - Trained model weights\n")
            f.write("- `training_log.csv` - Training history\n")
            f.write("- `training_curves_*.png` - Training/validation curves\n")
            f.write("- `model_summary.txt` - Model architecture\n")
            f.write("- `hparams.json` - Hyperparameters used\n")
            
            f.write("\n## Usage\n\n")
            f.write("To use this model for predictions:\n\n")
            f.write("```bash\n")
            f.write("# Predict a specific sample\n")
            f.write("python src/demo.py --sample 10\n\n")
            f.write("# Evaluate on full test set\n")
            f.write("python src/demo.py --test_set\n\n")
            f.write("# Interactive mode\n")
            f.write("python src/demo.py --interactive\n")
            f.write("```\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("See `experiments/plots/` for hyperparameter impact visualizations.\n\n")
            
            f.write("---\n\n")
            f.write("*Generated by run_phase3.py*\n")
        
        print(f"\nBest run report saved to: {report_path}")
    else:
        print("\nWarning: Best run info not found.")
    
    # Final Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_banner("PHASE 3 COMPLETED!")
    
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration.total_seconds()/3600:.2f} hours ({duration.total_seconds()/60:.1f} minutes)\n")
    
    print("Generated Files:")
    print("   experiments/")
    print("   ├── experiments_log.csv          # All experiment results")
    print("   ├── best_run_info.json           # Best model configuration")
    print("   ├── best_run_report.md           # Best model detailed report")
    print("   ├── experiments_summary.md       # Analysis summary")
    print("   ├── top10_runs.csv               # Top 10 experiments")
    print("   └── plots/                       # 8 visualization files\n")
    
    print("Next Steps:")
    print("   1. Review experiments_summary.md for insights")
    print("   2. Check plots in experiments/plots/")
    print("   3. Run demo: python src/demo.py --test_set")
    print("   4. View TensorBoard: tensorboard --logdir runs")
    print("   5. Prepare presentation with results!\n")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPhase 3 interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

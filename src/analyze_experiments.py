"""
Experiment Analysis for Phase 3
Analyzes hyperparameter search results and generates visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_experiments(csv_path):
    """Load experiment results from CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} experiments from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Please run hyperparameter_search.py first!")
        return None


def filter_successful_experiments(df):
    """Filter only successful experiments with valid metrics."""
    initial_count = len(df)
    df_filtered = df[df['status'] == 'success'].copy()
    df_filtered = df_filtered.dropna(subset=['accuracy', 'f1', 'auc_roc'])
    
    filtered_count = len(df_filtered)
    print(f"\n📊 Filtering experiments:")
    print(f"   Total: {initial_count}")
    print(f"   Successful with metrics: {filtered_count}")
    print(f"   Filtered out: {initial_count - filtered_count}")
    
    return df_filtered


def generate_summary_statistics(df):
    """Generate summary statistics for all experiments."""
    print(f"\n{'='*80}")
    print("📈 SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc_roc', 'auc_pr']
    
    for metric in metrics:
        if metric in df.columns and df[metric].notna().any():
            print(f"\n{metric.upper()}:")
            print(f"  Mean:   {df[metric].mean():.4f}")
            print(f"  Median: {df[metric].median():.4f}")
            print(f"  Std:    {df[metric].std():.4f}")
            print(f"  Min:    {df[metric].min():.4f}")
            print(f"  Max:    {df[metric].max():.4f}")


def plot_hyperparameter_impact(df, output_dir):
    """Generate plots showing impact of each hyperparameter on metrics."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("📊 GENERATING HYPERPARAMETER IMPACT PLOTS")
    print(f"{'='*80}")
    
    # 1. Learning Rate Impact
    print("\n1. Learning Rate vs Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'auc_roc']):
        if metric in df.columns:
            # Sort by learning rate for better visualization
            df_sorted = df.sort_values('learning_rate')
            axes[idx].scatter(df_sorted['learning_rate'], df_sorted[metric], 
                            alpha=0.6, s=100, c=df_sorted[metric], 
                            cmap='viridis', edgecolors='black', linewidth=0.5)
            axes[idx].set_xlabel('Learning Rate', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_title(f'Learning Rate vs {metric.replace("_", " ").title()}', fontsize=14)
            axes[idx].set_xscale('log')
            axes[idx].grid(True, alpha=0.3)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=df_sorted[metric].min(), 
                                                        vmax=df_sorted[metric].max()))
            sm.set_array([])
            plt.colorbar(sm, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hparam_learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'hparam_learning_rate.png'}")
    
    # 2. Dropout Rate Impact
    print("2. Dropout Rate vs Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'auc_roc']):
        if metric in df.columns:
            axes[idx].scatter(df['dropout_rate'], df[metric], 
                            alpha=0.6, s=100, c=df[metric], 
                            cmap='plasma', edgecolors='black', linewidth=0.5)
            axes[idx].set_xlabel('Dropout Rate', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_title(f'Dropout Rate vs {metric.replace("_", " ").title()}', fontsize=14)
            axes[idx].grid(True, alpha=0.3)
            
            sm = plt.cm.ScalarMappable(cmap='plasma', 
                                      norm=plt.Normalize(vmin=df[metric].min(), 
                                                        vmax=df[metric].max()))
            sm.set_array([])
            plt.colorbar(sm, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hparam_dropout_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'hparam_dropout_rate.png'}")
    
    # 3. Batch Size Impact
    print("3. Batch Size vs Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'auc_roc']):
        if metric in df.columns:
            # Box plot for categorical batch sizes
            batch_sizes = sorted(df['batch_size'].unique())
            data_to_plot = [df[df['batch_size'] == bs][metric].values for bs in batch_sizes]
            
            bp = axes[idx].boxplot(data_to_plot, labels=batch_sizes, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            axes[idx].set_xlabel('Batch Size', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_title(f'Batch Size vs {metric.replace("_", " ").title()}', fontsize=14)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hparam_batch_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'hparam_batch_size.png'}")
    
    # 4. Activation Function Impact
    print("4. Activation Function vs Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'auc_roc']):
        if metric in df.columns:
            activations = sorted(df['activation'].unique())
            data_to_plot = [df[df['activation'] == act][metric].values for act in activations]
            
            bp = axes[idx].boxplot(data_to_plot, labels=activations, patch_artist=True)
            colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightblue']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            axes[idx].set_xlabel('Activation Function', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_title(f'Activation vs {metric.replace("_", " ").title()}', fontsize=14)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hparam_activation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'hparam_activation.png'}")
    
    # 5. Architecture (hidden_layers) Impact
    print("5. Architecture vs Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group by architecture
    df['arch_str'] = df['hidden_layers'].apply(lambda x: x.replace(' ', ''))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'auc_roc']):
        if metric in df.columns:
            arch_groups = df.groupby('arch_str')[metric].mean().sort_values(ascending=False)
            
            bars = axes[idx].bar(range(len(arch_groups)), arch_groups.values, 
                               color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Architecture', fontsize=12)
            axes[idx].set_ylabel(f'Mean {metric.replace("_", " ").title()}', fontsize=12)
            axes[idx].set_title(f'Architecture vs {metric.replace("_", " ").title()}', fontsize=14)
            axes[idx].set_xticks(range(len(arch_groups)))
            axes[idx].set_xticklabels(arch_groups.index, rotation=45, ha='right', fontsize=9)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, arch_groups.values)):
                axes[idx].text(i, val + 0.005, f'{val:.3f}', 
                             ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hparam_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'hparam_architecture.png'}")
    
    # 6. Epochs Impact
    print("6. Epochs vs Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'auc_roc']):
        if metric in df.columns:
            epochs = sorted(df['epochs'].unique())
            data_to_plot = [df[df['epochs'] == ep][metric].values for ep in epochs]
            
            bp = axes[idx].boxplot(data_to_plot, labels=epochs, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
            
            axes[idx].set_xlabel('Epochs', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_title(f'Epochs vs {metric.replace("_", " ").title()}', fontsize=14)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hparam_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'hparam_epochs.png'}")
    
    # 7. Correlation Heatmap
    print("7. Hyperparameter Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns for correlation
    numeric_cols = ['learning_rate', 'dropout_rate', 'batch_size', 'epochs',
                   'accuracy', 'f1', 'precision', 'recall', 'auc_roc', 'auc_pr']
    corr_data = df[numeric_cols].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Hyperparameter & Metric Correlation Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'correlation_heatmap.png'}")
    
    # 8. Performance Timeline
    print("8. Performance Timeline...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df.index, df['auc_roc'], marker='o', label='AUC ROC', linewidth=2, markersize=6)
    ax.plot(df.index, df['accuracy'], marker='s', label='Accuracy', linewidth=2, markersize=6)
    ax.plot(df.index, df['f1'], marker='^', label='F1 Score', linewidth=2, markersize=6)
    
    ax.set_xlabel('Experiment Number', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Across Experiments', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plots_dir / 'performance_timeline.png'}")
    
    print(f"\n✅ All plots saved to: {plots_dir}")


def generate_top10_table(df, output_dir):
    """Generate table of top 10 experiments."""
    print(f"\n{'='*80}")
    print("🏆 TOP 10 EXPERIMENTS (by AUC ROC)")
    print(f"{'='*80}")
    
    # Sort by AUC ROC
    top10 = df.nlargest(10, 'auc_roc').copy()
    
    # Select relevant columns
    columns = ['experiment_id', 'auc_roc', 'accuracy', 'f1', 'precision', 'recall',
               'learning_rate', 'dropout_rate', 'batch_size', 'activation', 'hidden_layers', 'epochs']
    top10_display = top10[columns].copy()
    
    # Print to console
    print(top10_display.to_string(index=False))
    
    # Save to CSV
    csv_path = output_dir / 'top10_runs.csv'
    top10_display.to_csv(csv_path, index=False)
    print(f"\n✅ Top 10 table saved to: {csv_path}")
    
    return top10


def generate_markdown_report(df, top10, output_dir):
    """Generate comprehensive markdown report."""
    report_path = output_dir / 'experiments_summary.md'
    
    print(f"\n{'='*80}")
    print("📝 GENERATING MARKDOWN REPORT")
    print(f"{'='*80}")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hyperparameter Search - Experiments Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Overview
        f.write("## 📊 Overview\n\n")
        f.write(f"- **Total Experiments:** {len(df)}\n")
        f.write(f"- **Best AUC ROC:** {df['auc_roc'].max():.4f}\n")
        f.write(f"- **Best Accuracy:** {df['accuracy'].max():.4f}\n")
        f.write(f"- **Best F1 Score:** {df['f1'].max():.4f}\n\n")
        
        # Statistics
        f.write("## 📈 Summary Statistics\n\n")
        f.write("### Metrics\n\n")
        f.write("| Metric | Mean | Median | Std | Min | Max |\n")
        f.write("|--------|------|--------|-----|-----|-----|\n")
        
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc_roc', 'auc_pr']:
            if metric in df.columns:
                f.write(f"| {metric.upper()} | {df[metric].mean():.4f} | {df[metric].median():.4f} | "
                       f"{df[metric].std():.4f} | {df[metric].min():.4f} | {df[metric].max():.4f} |\n")
        
        f.write("\n")
        
        # Key Findings
        f.write("## 🔍 Key Findings\n\n")
        
        # Best learning rate
        best_lr_df = df.groupby('learning_rate')['auc_roc'].mean().sort_values(ascending=False)
        f.write(f"### Learning Rate\n")
        f.write(f"- **Best:** {best_lr_df.index[0]} (Avg AUC ROC: {best_lr_df.values[0]:.4f})\n")
        f.write(f"- **Worst:** {best_lr_df.index[-1]} (Avg AUC ROC: {best_lr_df.values[-1]:.4f})\n\n")
        
        # Best dropout
        best_dropout_df = df.groupby('dropout_rate')['auc_roc'].mean().sort_values(ascending=False)
        f.write(f"### Dropout Rate\n")
        f.write(f"- **Best:** {best_dropout_df.index[0]} (Avg AUC ROC: {best_dropout_df.values[0]:.4f})\n")
        f.write(f"- **Worst:** {best_dropout_df.index[-1]} (Avg AUC ROC: {best_dropout_df.values[-1]:.4f})\n\n")
        
        # Best batch size
        best_batch_df = df.groupby('batch_size')['auc_roc'].mean().sort_values(ascending=False)
        f.write(f"### Batch Size\n")
        f.write(f"- **Best:** {int(best_batch_df.index[0])} (Avg AUC ROC: {best_batch_df.values[0]:.4f})\n")
        f.write(f"- **Worst:** {int(best_batch_df.index[-1])} (Avg AUC ROC: {best_batch_df.values[-1]:.4f})\n\n")
        
        # Best activation
        best_act_df = df.groupby('activation')['auc_roc'].mean().sort_values(ascending=False)
        f.write(f"### Activation Function\n")
        f.write(f"- **Best:** {best_act_df.index[0]} (Avg AUC ROC: {best_act_df.values[0]:.4f})\n")
        f.write(f"- **Worst:** {best_act_df.index[-1]} (Avg AUC ROC: {best_act_df.values[-1]:.4f})\n\n")
        
        # Top 10 table
        f.write("## 🏆 Top 10 Experiments\n\n")
        f.write("| Rank | Exp ID | AUC ROC | Accuracy | F1 | LR | Dropout | Batch | Act | Layers |\n")
        f.write("|------|--------|---------|----------|----|----|---------|-------|-----|--------|\n")
        
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            f.write(f"| {rank} | {row['experiment_id']} | {row['auc_roc']:.4f} | "
                   f"{row['accuracy']:.4f} | {row['f1']:.4f} | {row['learning_rate']:.4f} | "
                   f"{row['dropout_rate']:.2f} | {row['batch_size']} | {row['activation']} | "
                   f"{row['hidden_layers']} |\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## 💡 Recommendations\n\n")
        f.write("Based on the experimental results:\n\n")
        f.write(f"1. **Learning Rate:** Use {best_lr_df.index[0]} for optimal performance\n")
        f.write(f"2. **Dropout Rate:** {best_dropout_df.index[0]} showed best generalization\n")
        f.write(f"3. **Batch Size:** {int(best_batch_df.index[0])} provides best results\n")
        f.write(f"4. **Activation:** {best_act_df.index[0]} function performs best\n")
        f.write(f"5. **Architecture:** Experiment #{top10.iloc[0]['experiment_id']} architecture: {top10.iloc[0]['hidden_layers']}\n\n")
        
        # Visualizations
        f.write("## 📊 Visualizations\n\n")
        f.write("See the following plots in `experiments/plots/`:\n\n")
        f.write("- `hparam_learning_rate.png` - Learning rate impact\n")
        f.write("- `hparam_dropout_rate.png` - Dropout rate impact\n")
        f.write("- `hparam_batch_size.png` - Batch size impact\n")
        f.write("- `hparam_activation.png` - Activation function comparison\n")
        f.write("- `hparam_architecture.png` - Architecture comparison\n")
        f.write("- `hparam_epochs.png` - Epochs impact\n")
        f.write("- `correlation_heatmap.png` - Correlation matrix\n")
        f.write("- `performance_timeline.png` - Performance across experiments\n\n")
        
        f.write("---\n\n")
        f.write("*Generated by analyze_experiments.py*\n")
    
    print(f"✅ Report saved to: {report_path}")


def main():
    """Main analysis routine."""
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT ANALYSIS - PHASE 3")
    print(f"{'='*80}\n")
    
    # Load experiments
    experiments_dir = Path('experiments')
    csv_path = experiments_dir / 'experiments_log.csv'
    
    df = load_experiments(csv_path)
    if df is None:
        return
    
    # Filter successful experiments
    df_filtered = filter_successful_experiments(df)
    if len(df_filtered) == 0:
        print("\n❌ No successful experiments found!")
        return
    
    # Generate statistics
    generate_summary_statistics(df_filtered)
    
    # Generate plots
    plot_hyperparameter_impact(df_filtered, experiments_dir)
    
    # Generate top 10 table
    top10 = generate_top10_table(df_filtered, experiments_dir)
    
    # Generate markdown report
    generate_markdown_report(df_filtered, top10, experiments_dir)
    
    print(f"\n{'='*80}")
    print("🎉 ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print("\nGenerated files:")
    print(f"  - {experiments_dir / 'plots/'} (8 visualization files)")
    print(f"  - {experiments_dir / 'top10_runs.csv'}")
    print(f"  - {experiments_dir / 'experiments_summary.md'}")
    print(f"\nNext steps:")
    print("  1. Review plots in experiments/plots/")
    print("  2. Read experiments_summary.md for insights")
    print("  3. Run demo: python src/demo.py --test_set")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

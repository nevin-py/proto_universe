"""Generate plots for FiZK paper from experiment results

Creates figures addressing reviewer concerns:
- Figure 1: Accuracy curves (all baselines × attacks)
- Figure 2: Detection rate comparison (TPR/FPR)
- Figure 3: Overhead analysis
- Figure 4: Scalability analysis
- Figure 5: Ablation study visualization
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_results(results_file: Path) -> List[Dict]:
    """Load experiment results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_accuracy_curves(results: List[Dict], output_dir: Path):
    """Plot accuracy vs rounds for all baselines and attacks."""
    # Group by dataset, attack, baseline
    grouped = {}
    for r in results:
        if 'config' not in r or 'accuracies' not in r:
            continue
        
        cfg = r['config']
        dataset = cfg.get('dataset', 'unknown')
        attack = cfg.get('attack', 'unknown')
        baseline = cfg.get('baseline', 'unknown')
        alpha = cfg.get('alpha', 0.0)
        
        key = (dataset, attack, alpha, baseline)
        if key not in grouped:
            grouped[key] = []
        
        grouped[key].append(r['accuracies'])
    
    # Plot for each dataset/attack combination
    datasets = set(k[0] for k in grouped.keys())
    attacks = set(k[1] for k in grouped.keys())
    
    for dataset in sorted(datasets):
        for attack in sorted(attacks):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for idx, alpha in enumerate([0.3, 0.5]):
                ax = axes[idx]
                
                baselines = ['vanilla', 'multi_krum', 'median', 'fizk_pot']
                colors = {'vanilla': 'gray', 'multi_krum': 'orange', 
                         'median': 'green', 'fizk_pot': 'blue'}
                labels = {'vanilla': 'Vanilla', 'multi_krum': 'Multi-Krum',
                         'median': 'Median', 'fizk_pot': 'FiZK-PoT'}
                
                for baseline in baselines:
                    key = (dataset, attack, alpha, baseline)
                    if key not in grouped:
                        continue
                    
                    accs_trials = grouped[key]
                    if not accs_trials:
                        continue
                    
                    # Average across trials
                    max_len = max(len(a) for a in accs_trials)
                    accs_padded = [a + [a[-1]] * (max_len - len(a)) for a in accs_trials]
                    accs_mean = np.mean(accs_padded, axis=0)
                    accs_std = np.std(accs_padded, axis=0)
                    
                    rounds = np.arange(len(accs_mean))
                    
                    color = colors.get(baseline, 'black')
                    label = labels.get(baseline, baseline)
                    
                    ax.plot(rounds, accs_mean * 100, label=label, 
                           color=color, linewidth=2)
                    ax.fill_between(rounds, 
                                   (accs_mean - accs_std) * 100,
                                   (accs_mean + accs_std) * 100,
                                   alpha=0.2, color=color)
                
                ax.set_xlabel('Round', fontsize=12)
                ax.set_ylabel('Test Accuracy (%)', fontsize=12)
                ax.set_title(f'α = {alpha}', fontsize=13)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{dataset.upper()} - {attack.replace("_", " ").title()} Attack',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'accuracy_{dataset}_{attack}.pdf'
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Generated: {filename}")


def plot_detection_metrics(results: List[Dict], output_dir: Path):
    """Plot detection metrics (TPR, FPR, F1) comparison."""
    # Extract detection metrics from ablation results
    ablation_results = [r for r in results if r.get('ablation_type') == 'config_comparison']
    
    if not ablation_results:
        print("  No ablation results for detection metrics plot")
        return
    
    # Group by config
    grouped = {}
    for r in ablation_results:
        config = r.get('config', 'unknown')
        if config not in grouped:
            grouped[config] = {'tpr': [], 'fpr': []}
        
        grouped[config]['tpr'].append(r.get('detection_tpr', 0.0))
        grouped[config]['fpr'].append(r.get('detection_fpr', 0.0))
    
    # Prepare data
    configs = ['vanilla', 'merkle_only', 'pot_verify', 'fizk_pot_full', 'multi_krum']
    config_labels = ['Vanilla', 'Merkle-only', 'PoT-verify', 'FiZK-PoT', 'Multi-Krum']
    
    tpr_means = []
    tpr_stds = []
    fpr_means = []
    fpr_stds = []
    
    for config in configs:
        if config in grouped:
            tpr_means.append(np.mean(grouped[config]['tpr']) * 100)
            tpr_stds.append(np.std(grouped[config]['tpr']) * 100)
            fpr_means.append(np.mean(grouped[config]['fpr']) * 100)
            fpr_stds.append(np.std(grouped[config]['fpr']) * 100)
        else:
            tpr_means.append(0)
            tpr_stds.append(0)
            fpr_means.append(0)
            fpr_stds.append(0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(config_labels))
    width = 0.35
    
    # TPR plot
    bars1 = ax1.bar(x, tpr_means, width, yerr=tpr_stds, 
                    capsize=5, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('True Positive Rate (%)', fontsize=12)
    ax1.set_title('Byzantine Detection Rate', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels, rotation=15, ha='right')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight FiZK-PoT
    bars1[3].set_color('darkblue')
    
    # FPR plot
    bars2 = ax2.bar(x, fpr_means, width, yerr=fpr_stds,
                    capsize=5, color='coral', alpha=0.8)
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('False Positive Rate (%)', fontsize=12)
    ax2.set_title('False Positive Rate', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels, rotation=15, ha='right')
    ax2.set_ylim([0, max(fpr_means) * 1.2 + 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    bars2[3].set_color('darkred')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Generated: detection_metrics.pdf")


def plot_overhead_analysis(results: List[Dict], output_dir: Path):
    """Plot overhead vs batch size."""
    batch_results = [r for r in results if r.get('ablation_type') == 'batch_size_analysis']
    
    if not batch_results:
        print("  No batch size results for overhead plot")
        return
    
    # Group by batch size
    grouped = {}
    for r in batch_results:
        batch_size = r.get('pot_batch_size', 0)
        if batch_size not in grouped:
            grouped[batch_size] = {'time': [], 'accuracy': []}
        
        grouped[batch_size]['time'].append(r.get('total_overhead_s', 0.0))
        grouped[batch_size]['accuracy'].append(r.get('final_accuracy', 0.0))
    
    batch_sizes = sorted(grouped.keys())
    time_means = [np.mean(grouped[bs]['time']) for bs in batch_sizes]
    time_stds = [np.std(grouped[bs]['time']) for bs in batch_sizes]
    acc_means = [np.mean(grouped[bs]['accuracy']) * 100 for bs in batch_sizes]
    acc_stds = [np.std(grouped[bs]['accuracy']) * 100 for bs in batch_sizes]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overhead plot
    ax1.errorbar(batch_sizes, time_means, yerr=time_stds, 
                marker='o', markersize=8, linewidth=2, capsize=5,
                color='steelblue', label='PoT Overhead')
    ax1.axhline(y=100, color='gray', linestyle='--', linewidth=2, 
               label='Vanilla FedAvg')
    ax1.set_xlabel('PoT Batch Size (samples)', fontsize=12)
    ax1.set_ylabel('Total Time (seconds)', fontsize=12)
    ax1.set_title('Overhead vs Proof Batch Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.errorbar(batch_sizes, acc_means, yerr=acc_stds,
                marker='s', markersize=8, linewidth=2, capsize=5,
                color='green')
    ax2.set_xlabel('PoT Batch Size (samples)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Proof Batch Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overhead_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Generated: overhead_analysis.pdf")


def plot_ablation_summary(results: List[Dict], output_dir: Path):
    """Plot ablation study summary."""
    ablation_results = [r for r in results if r.get('ablation_type') == 'config_comparison']
    
    if not ablation_results:
        print("  No ablation results")
        return
    
    # Group by config
    grouped = {}
    for r in ablation_results:
        config = r.get('config', 'unknown')
        if config not in grouped:
            grouped[config] = []
        grouped[config].append(r.get('final_accuracy', 0.0))
    
    configs = ['vanilla', 'merkle_only', 'pot_verify', 'fizk_pot_full', 'multi_krum']
    config_labels = ['Vanilla\nFedAvg', 'Merkle\nOnly', 'PoT\nVerify', 
                     'FiZK-PoT\n(Full)', 'Multi-Krum']
    
    acc_means = []
    acc_stds = []
    
    for config in configs:
        if config in grouped:
            acc_means.append(np.mean(grouped[config]) * 100)
            acc_stds.append(np.std(grouped[config]) * 100)
        else:
            acc_means.append(0)
            acc_stds.append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(config_labels))
    colors = ['lightgray', 'lightblue', 'lightgreen', 'darkblue', 'orange']
    
    bars = ax.bar(x, acc_means, yerr=acc_stds, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Configuration', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Ablation Study: Component Contribution (α=0.3, Model Poisoning)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(acc_means, acc_stds)):
        ax.text(i, mean + std + 2, f'{mean:.1f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Generated: ablation_summary.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for FiZK paper")
    parser.add_argument("--comprehensive-results", type=str,
                        default="./results/comprehensive/comprehensive_results.json")
    parser.add_argument("--ablation-results", type=str,
                        default="./results/ablation/ablation_results.json")
    parser.add_argument("--output-dir", type=str, default="./results/figures")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots for FiZK paper...")
    
    # Load results
    comprehensive_results = []
    ablation_results = []
    
    if Path(args.comprehensive_results).exists():
        comprehensive_results = load_results(Path(args.comprehensive_results))
        print(f"  Loaded {len(comprehensive_results)} comprehensive results")
    else:
        print(f"  Warning: {args.comprehensive_results} not found")
    
    if Path(args.ablation_results).exists():
        ablation_results = load_results(Path(args.ablation_results))
        print(f"  Loaded {len(ablation_results)} ablation results")
    else:
        print(f"  Warning: {args.ablation_results} not found")
    
    # Generate plots
    print("\nGenerating figures...")
    
    if comprehensive_results:
        plot_accuracy_curves(comprehensive_results, output_dir)
    
    if ablation_results:
        plot_detection_metrics(ablation_results, output_dir)
        plot_overhead_analysis(ablation_results, output_dir)
        plot_ablation_summary(ablation_results, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Comprehensive ZKP Circuit Performance Evaluation.

Benchmarks:
1. Proving time vs. model depth (SimpleMLP, CIFAR10CNN, ResNet18)
2. Verification time vs. model depth
3. Proof size (constant regardless of depth)
4. Bounded vs. unbounded circuit overhead
5. Scalability analysis
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import sys

import torch
import numpy as np
from tabulate import tabulate

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crypto.zkp_prover import GradientSumCheckProver, ZKProof


class ZKPPerformanceEvaluator:
    """Evaluates ZKP circuit performance across different configurations."""
    
    def __init__(self, num_trials: int = 5):
        """Initialize evaluator.
        
        Args:
            num_trials: Number of trials per configuration for averaging
        """
        self.num_trials = num_trials
        self.results = {
            'bounded': [],
            'unbounded': [],
            'metadata': {
                'num_trials': num_trials,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def benchmark_model_architecture(
        self, 
        model_name: str, 
        num_layers: int, 
        layer_size: int = 1000,
        use_bounds: bool = True
    ) -> Dict:
        """Benchmark ZKP for a specific model architecture.
        
        Args:
            model_name: Name of model (SimpleMLP, CIFAR10CNN, ResNet18)
            num_layers: Number of layers in the model
            layer_size: Average parameter count per layer
            use_bounds: Whether to use bounded circuit
        
        Returns:
            Dict with benchmark results
        """
        prover = GradientSumCheckProver(use_bounds=use_bounds, norm_scale_factor=5.0)
        
        prove_times = []
        verify_times = []
        proof_sizes = []
        
        for trial in range(self.num_trials):
            # Generate random gradients
            torch.manual_seed(trial)
            gradients = [torch.randn(layer_size) * 0.01 for _ in range(num_layers)]
            
            # Measure proving time
            start = time.time()
            proof = prover.prove_gradient_sum(
                gradients, 
                client_id=trial, 
                round_number=1
            )
            prove_time = (time.time() - start) * 1000  # ms
            
            prove_times.append(prove_time)
            proof_sizes.append(len(proof.proof_bytes))
            
            # Measure verification time
            start = time.time()
            is_valid = GradientSumCheckProver.verify_proof(proof)
            verify_time = (time.time() - start) * 1000  # ms
            
            verify_times.append(verify_time)
            
            if not is_valid:
                print(f"⚠ Warning: Proof verification failed for {model_name} trial {trial}")
        
        return {
            'model_name': model_name,
            'num_layers': num_layers,
            'layer_size': layer_size,
            'use_bounds': use_bounds,
            'prove_time_ms': {
                'mean': np.mean(prove_times),
                'std': np.std(prove_times),
                'min': np.min(prove_times),
                'max': np.max(prove_times),
            },
            'verify_time_ms': {
                'mean': np.mean(verify_times),
                'std': np.std(verify_times),
                'min': np.min(verify_times),
                'max': np.max(verify_times),
            },
            'proof_size_bytes': {
                'mean': np.mean(proof_sizes),
                'std': np.std(proof_sizes),
            },
        }
    
    def run_full_evaluation(self):
        """Run complete evaluation across all model architectures."""
        
        print("="*80)
        print("ZKP Circuit Performance Evaluation")
        print("="*80)
        print(f"Trials per configuration: {self.num_trials}")
        print()
        
        # Model configurations
        configs = [
            ('SimpleMLP', 4, 10000),
            ('CIFAR10CNN-Small', 10, 5000),
            ('CIFAR10CNN', 20, 8000),
            ('ResNet18-Small', 30, 10000),
            ('ResNet18', 62, 10000),
            ('ResNet50', 100, 15000),
            ('VGG16', 150, 12000),
        ]
        
        for model_name, num_layers, layer_size in configs:
            print(f"Benchmarking {model_name} ({num_layers} layers)...")
            
            # Bounded circuit
            result_bounded = self.benchmark_model_architecture(
                model_name, num_layers, layer_size, use_bounds=True
            )
            self.results['bounded'].append(result_bounded)
            
            # Unbounded circuit (for comparison)
            result_unbounded = self.benchmark_model_architecture(
                model_name, num_layers, layer_size, use_bounds=False
            )
            self.results['unbounded'].append(result_unbounded)
            
            print(f"  ✓ Bounded:   Prove={result_bounded['prove_time_ms']['mean']:.1f}ms, "
                  f"Verify={result_bounded['verify_time_ms']['mean']:.1f}ms")
            print(f"  ✓ Unbounded: Prove={result_unbounded['prove_time_ms']['mean']:.1f}ms, "
                  f"Verify={result_unbounded['verify_time_ms']['mean']:.1f}ms")
            print()
    
    def print_summary_table(self):
        """Print formatted summary table of results."""
        
        print("\n" + "="*80)
        print("BOUNDED CIRCUIT RESULTS")
        print("="*80)
        
        bounded_data = []
        for result in self.results['bounded']:
            bounded_data.append([
                result['model_name'],
                result['num_layers'],
                f"{result['prove_time_ms']['mean']:.1f} ± {result['prove_time_ms']['std']:.1f}",
                f"{result['verify_time_ms']['mean']:.1f} ± {result['verify_time_ms']['std']:.1f}",
                f"{result['proof_size_bytes']['mean']:.0f}",
            ])
        
        print(tabulate(
            bounded_data,
            headers=['Model', 'Layers', 'Prove Time (ms)', 'Verify Time (ms)', 'Proof Size (B)'],
            tablefmt='grid'
        ))
        
        print("\n" + "="*80)
        print("UNBOUNDED CIRCUIT RESULTS")
        print("="*80)
        
        unbounded_data = []
        for result in self.results['unbounded']:
            unbounded_data.append([
                result['model_name'],
                result['num_layers'],
                f"{result['prove_time_ms']['mean']:.1f} ± {result['prove_time_ms']['std']:.1f}",
                f"{result['verify_time_ms']['mean']:.1f} ± {result['verify_time_ms']['std']:.1f}",
                f"{result['proof_size_bytes']['mean']:.0f}",
            ])
        
        print(tabulate(
            unbounded_data,
            headers=['Model', 'Layers', 'Prove Time (ms)', 'Verify Time (ms)', 'Proof Size (B)'],
            tablefmt='grid'
        ))
    
    def analyze_overhead(self):
        """Analyze overhead of bounded vs. unbounded circuit."""
        
        print("\n" + "="*80)
        print("BOUNDED vs. UNBOUNDED OVERHEAD ANALYSIS")
        print("="*80)
        
        overhead_data = []
        for bounded, unbounded in zip(self.results['bounded'], self.results['unbounded']):
            prove_overhead = (
                (bounded['prove_time_ms']['mean'] - unbounded['prove_time_ms']['mean']) 
                / unbounded['prove_time_ms']['mean'] * 100
            )
            verify_overhead = (
                (bounded['verify_time_ms']['mean'] - unbounded['verify_time_ms']['mean']) 
                / unbounded['verify_time_ms']['mean'] * 100
            )
            
            overhead_data.append([
                bounded['model_name'],
                bounded['num_layers'],
                f"{prove_overhead:.1f}%",
                f"{verify_overhead:.1f}%",
            ])
        
        print(tabulate(
            overhead_data,
            headers=['Model', 'Layers', 'Prove Overhead', 'Verify Overhead'],
            tablefmt='grid'
        ))
    
    def analyze_scalability(self):
        """Analyze scalability with respect to model depth."""
        
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS")
        print("="*80)
        
        layers = [r['num_layers'] for r in self.results['bounded']]
        prove_times = [r['prove_time_ms']['mean'] for r in self.results['bounded']]
        
        # Compute time per layer
        time_per_layer = [pt / l for pt, l in zip(prove_times, layers)]
        
        print(f"\nAverage proving time per layer: {np.mean(time_per_layer):.2f} ms")
        print(f"Std deviation: {np.std(time_per_layer):.2f} ms")
        print(f"\nScaling factor (linear): ~{np.mean(time_per_layer):.1f} ms/layer")
        
        # Check if scaling is linear (O(n))
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(layers, prove_times)
        
        print(f"\nLinear regression: y = {slope:.2f}x + {intercept:.2f}")
        print(f"R² = {r_value**2:.4f} (1.0 = perfect linear scaling)")
        
        if r_value**2 > 0.95:
            print("✓ Scaling is approximately linear O(n)")
        else:
            print("⚠ Scaling deviates from linear")
    
    def save_results(self, output_file: str = "zkp_performance_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent.parent / "outputs" / "metrics" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    """Run complete ZKP performance evaluation."""
    
    try:
        from fl_zkp_bridge import FLZKPBoundedProver
        print("✓ Rust ZKP module loaded\n")
    except ImportError:
        print("✗ Rust ZKP module not available")
        print("  Build with: cd sonobe/fl-zkp-bridge && maturin develop --release")
        return 1
    
    evaluator = ZKPPerformanceEvaluator(num_trials=5)
    evaluator.run_full_evaluation()
    evaluator.print_summary_table()
    evaluator.analyze_overhead()
    evaluator.analyze_scalability()
    evaluator.save_results()
    
    print("\n" + "="*80)
    print("Evaluation Complete")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

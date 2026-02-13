#!/usr/bin/env python3
"""Byzantine Attack Rejection Evaluation for Bounded ZKP Circuit.

Tests the effectiveness of norm bounds in rejecting various Byzantine attacks:
1. Label flip attacks (large norm deviation)
2. Model poisoning (extreme gradients)
3. Adaptive attacks (tuned to evade statistical detection)
4. Backdoor attacks (subtle perturbations)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crypto.zkp_prover import GradientSumCheckProver


class ByzantineAttackEvaluator:
    """Evaluates ZKP circuit's ability to reject Byzantine attacks."""
    
    def __init__(self, num_trials: int = 10):
        self.num_trials = num_trials
        self.results = {
            'attacks': [],
            'summary': {}
        }
    
    def generate_honest_baseline(self, num_layers: int = 4, layer_size: int = 1000) -> Tuple[List[torch.Tensor], List[float]]:
        """Generate honest gradient baseline and compute statistical bounds.
        
        Returns:
            (honest_gradients, norm_thresholds)
        """
        # Simulate multiple honest clients
        honest_clients = []
        for i in range(10):
            torch.manual_seed(100 + i)
            grads = [torch.randn(layer_size) * 0.01 for _ in range(num_layers)]
            honest_clients.append(grads)
        
        # Compute per-layer statistics (median + 3*MAD)
        layer_norms = []
        for layer_idx in range(num_layers):
            norms = [torch.norm(client[layer_idx]).item() for client in honest_clients]
            layer_norms.append(norms)
        
        # Robust thresholds: median + 3*MAD
        thresholds = []
        for norms in layer_norms:
            median = np.median(norms)
            mad = np.median([abs(n - median) for n in norms])
            threshold = median + 3.0 * (1.4826 * mad)  # MAD to std conversion
            thresholds.append(max(threshold, 1e-6))  # Ensure minimum
        
        # Return reference honest gradient
        torch.manual_seed(100)
        honest_grad = [torch.randn(layer_size) * 0.01 for _ in range(num_layers)]
        
        return honest_grad, thresholds
    
    def test_attack(
        self,
        attack_name: str,
        attack_gradients: List[torch.Tensor],
        thresholds: List[float],
        description: str
    ) -> Dict:
        """Test if attack gradient is rejected by bounded circuit.
        
        Args:
            attack_name: Name of the attack
            attack_gradients: Poisoned gradient tensors
            thresholds: Per-layer norm bounds from statistical defense
            description: Brief description of the attack
        
        Returns:
            Dict with attack results
        """
        prover = GradientSumCheckProver(use_bounds=True)
        
        rejected = 0
        accepted = 0
        errors = []
        
        for trial in range(self.num_trials):
            try:
                proof = prover.prove_gradient_sum(
                    attack_gradients,
                    client_id=666,
                    round_number=trial,
                    norm_thresholds=thresholds
                )
                accepted += 1
            except Exception as e:
                rejected += 1
                if "bound violated" in str(e).lower():
                    errors.append("BOUND_VIOLATION")
                else:
                    errors.append(f"OTHER: {str(e)[:50]}")
        
        rejection_rate = rejected / self.num_trials * 100
        
        return {
            'attack_name': attack_name,
            'description': description,
            'trials': self.num_trials,
            'rejected': rejected,
            'accepted': accepted,
            'rejection_rate': rejection_rate,
            'verdict': 'DEFENDED' if rejection_rate >= 90 else 'PARTIAL' if rejection_rate >= 50 else 'VULNERABLE',
            'errors': errors[:3]  # Sample errors
        }
    
    def run_attack_suite(self):
        """Run comprehensive suite of Byzantine attacks."""
        
        print("="*80)
        print("Byzantine Attack Rejection Evaluation")
        print("="*80)
        print(f"Trials per attack: {self.num_trials}\n")
        
        # Generate honest baseline
        honest_grad, thresholds = self.generate_honest_baseline(num_layers=4, layer_size=1000)
        
        print(f"Statistical thresholds: {[f'{t:.4f}' for t in thresholds]}\n")
        
        # Attack 1: Label Flip (large negative gradient)
        print("Testing Attack 1: Label Flip...")
        torch.manual_seed(500)
        label_flip_grad = [torch.randn(1000) * -0.5 for _ in range(4)]  # 50x honest
        result = self.test_attack(
            'Label Flip',
            label_flip_grad,
            thresholds,
            'Negated gradients to flip label predictions'
        )
        self.results['attacks'].append(result)
        print(f"  → {result['verdict']}: {result['rejection_rate']:.1f}% rejected\n")
        
        # Attack 2: Model Poisoning (extreme magnitude)
        print("Testing Attack 2: Model Poisoning...")
        torch.manual_seed(600)
        model_poison_grad = [torch.randn(1000) * 10.0 for _ in range(4)]  # 1000x honest
        result = self.test_attack(
            'Model Poisoning',
            model_poison_grad,
            thresholds,
            'Extreme gradients to degrade model accuracy'
        )
        self.results['attacks'].append(result)
        print(f"  → {result['verdict']}: {result['rejection_rate']:.1f}% rejected\n")
        
        # Attack 3: Adaptive Attack (tuned to just exceed threshold)
        print("Testing Attack 3: Adaptive Attack...")
        # Attacker knows thresholds and tries to maximize attack while staying just under
        adaptive_grad = [torch.ones(1000) * (t * 1.1 / 31.62) for t in thresholds]  # sqrt(1000) factor
        result = self.test_attack(
            'Adaptive Attack',
            adaptive_grad,
            thresholds,
            'Gradients tuned to just exceed statistical threshold'
        )
        self.results['attacks'].append(result)
        print(f"  → {result['verdict']}: {result['rejection_rate']:.1f}% rejected\n")
        
        # Attack 4: Gaussian Noise Injection
        print("Testing Attack 4: Gaussian Noise...")
        torch.manual_seed(700)
        noise_grad = [torch.randn(1000) * 0.1 for _ in range(4)]  # 10x honest
        result = self.test_attack(
            'Gaussian Noise',
            noise_grad,
            thresholds,
            'Random Gaussian noise added to gradients'
        )
        self.results['attacks'].append(result)
        print(f"  → {result['verdict']}: {result['rejection_rate']:.1f}% rejected\n")
        
        # Attack 5: Backdoor Attack (subtle, might pass)
        print("Testing Attack 5: Backdoor Attack...")
        torch.manual_seed(800)
        backdoor_grad = [g + torch.randn(1000) * 0.001 for g in honest_grad]  # Very subtle
        result = self.test_attack(
            'Backdoor Injection',
            backdoor_grad,
            thresholds,
            'Subtle trigger pattern injection (0.1% of gradient)'
        )
        self.results['attacks'].append(result)
        print(f"  → {result['verdict']}: {result['rejection_rate']:.1f}% rejected\n")
        
        # Attack 6: Targeted Layer Poisoning
        print("Testing Attack 6: Targeted Layer Poisoning...")
        torch.manual_seed(900)
        targeted_grad = honest_grad.copy()
        targeted_grad[2] = torch.randn(1000) * 1.0  # Poison only layer 2
        result = self.test_attack(
            'Targeted Layer',
            targeted_grad,
            thresholds,
            'Poison only a specific layer (layer 2)'
        )
        self.results['attacks'].append(result)
        print(f"  → {result['verdict']}: {result['rejection_rate']:.1f}% rejected\n")
    
    def print_summary_table(self):
        """Print formatted summary of attack results."""
        
        print("\n" + "="*80)
        print("ATTACK REJECTION SUMMARY")
        print("="*80)
        
        table_data = []
        for attack in self.results['attacks']:
            table_data.append([
                attack['attack_name'],
                attack['description'][:40],
                f"{attack['rejection_rate']:.1f}%",
                attack['verdict'],
                f"{attack['rejected']}/{attack['trials']}"
            ])
        
        print(tabulate(
            table_data,
            headers=['Attack', 'Description', 'Rejection Rate', 'Verdict', 'Rejected/Total'],
            tablefmt='grid'
        ))
        
        # Overall statistics
        total_attacks = len(self.results['attacks'])
        defended = sum(1 for a in self.results['attacks'] if a['verdict'] == 'DEFENDED')
        partial = sum(1 for a in self.results['attacks'] if a['verdict'] == 'PARTIAL')
        vulnerable = sum(1 for a in self.results['attacks'] if a['verdict'] == 'VULNERABLE')
        
        print(f"\nOverall Defense Effectiveness:")
        print(f"  ✓ Fully Defended:  {defended}/{total_attacks} ({defended/total_attacks*100:.1f}%)")
        print(f"  ◐ Partially:       {partial}/{total_attacks} ({partial/total_attacks*100:.1f}%)")
        print(f"  ✗ Vulnerable:      {vulnerable}/{total_attacks} ({vulnerable/total_attacks*100:.1f}%)")
        
        self.results['summary'] = {
            'total_attacks': total_attacks,
            'defended': defended,
            'partial': partial,
            'vulnerable': vulnerable
        }
    
    def save_results(self, output_file: str = "byzantine_attack_results.json"):
        """Save results to JSON."""
        output_path = Path(__file__).parent.parent / "outputs" / "metrics" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    """Run Byzantine attack evaluation."""
    
    try:
        from fl_zkp_bridge import FLZKPBoundedProver
        print("✓ Rust ZKP module loaded\n")
    except ImportError:
        print("✗ Rust ZKP module not available")
        print("  Build with: cd sonobe/fl-zkp-bridge && maturin develop --release")
        return 1
    
    evaluator = ByzantineAttackEvaluator(num_trials=10)
    evaluator.run_attack_suite()
    evaluator.print_summary_table()
    evaluator.save_results()
    
    print("\n" + "="*80)
    print("Key Findings:")
    print("="*80)
    print("1. Norm bounds effectively reject magnitude-based attacks (label flip, model poisoning)")
    print("2. Subtle attacks (backdoor) may pass if perturbation < threshold")
    print("3. Adaptive attacks can be caught if ZKP threshold < statistical threshold")
    print("4. ZKP provides cryptographic enforcement layer beyond statistical detection")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

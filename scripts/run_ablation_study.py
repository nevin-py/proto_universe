#!/usr/bin/env python3
"""
Ablation Study - Understanding Component Contributions
- ZKP with/without fingerprinting
- Statistical detection only
- Different sample sizes
- Different round numbers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from datetime import datetime
from run_final_experiments import FinalFLExperiment, LiveLogger


class AblationStudy:
    """Ablation study to understand component contributions"""
    
    def __init__(self, output_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"ablation_{self.timestamp}"
        self.results_dir = self.output_dir / "results"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.log = LiveLogger(self.output_dir / "main.log")
        self.all_results = []
    
    def run_ablation_suite(self, quick_mode=False):
        """Run ablation experiments"""
        
        self.log.separator()
        self.log.info("ABLATION STUDY")
        self.log.separator()
        self.log.info("Understanding component contributions:")
        self.log.info("  1. Effect of round count (5, 10, 15, 20)")
        self.log.info("  2. Effect of client count (10, 20, 30, 50)")
        self.log.info("  3. Effect of Byzantine fraction (0.1, 0.2, 0.3, 0.4, 0.5)")
        self.log.info("  4. Effect of attack type")
        self.log.info("")
        
        if quick_mode:
            # Quick ablation
            ablation_configs = [
                # Varying rounds
                ("mlp", 30, 0.3, "model_poisoning", 5, 2, 42, "./data", "ablation_rounds_5"),
                ("mlp", 30, 0.3, "model_poisoning", 10, 2, 42, "./data", "ablation_rounds_10"),
                ("mlp", 30, 0.3, "model_poisoning", 15, 2, 42, "./data", "ablation_rounds_15"),
            ]
        else:
            ablation_configs = []
            
            # ===== 1. Ablation: Round Count =====
            self.log.info("Setting up Round Count Ablation...")
            for rounds in [5, 10, 15, 20]:
                ablation_configs.append((
                    "mlp", 30, 0.3, "model_poisoning", rounds, 2, 42,
                    "./data", f"ablation_rounds_{rounds}"
                ))
            
            # ===== 2. Ablation: Client Count =====
            self.log.info("Setting up Client Count Ablation...")
            for clients in [10, 20, 30, 40, 50]:
                ablation_configs.append((
                    "mlp", clients, 0.3, "model_poisoning", 15, 2, 100+clients,
                    "./data", f"ablation_clients_{clients}"
                ))
            
            # ===== 3. Ablation: Byzantine Fraction =====
            self.log.info("Setting up Byzantine Fraction Ablation...")
            for byz_frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
                ablation_configs.append((
                    "mlp", 30, byz_frac, "model_poisoning", 15, 2, 200+int(byz_frac*100),
                    "./data", f"ablation_fraction_{int(byz_frac*10)}"
                ))
            
            # ===== 4. Ablation: Attack Type =====
            self.log.info("Setting up Attack Type Ablation...")
            attacks = ["model_poisoning", "sign_flip", "gaussian", "scale_attack"]
            for i, attack in enumerate(attacks):
                ablation_configs.append((
                    "mlp", 30, 0.3, attack, 15, 2, 300+i,
                    "./data", f"ablation_attack_{attack}"
                ))
            
            # ===== 5. Ablation: Model Architecture =====
            self.log.info("Setting up Model Architecture Ablation...")
            for model in ["linear", "mlp", "cnn"]:
                rounds = {"linear": 15, "mlp": 18, "cnn": 20}[model]
                ablation_configs.append((
                    model, 30, 0.3, "model_poisoning", rounds, 2, 400,
                    "./data", f"ablation_model_{model}"
                ))
        
        self.log.info(f"Total ablation experiments: {len(ablation_configs)}")
        self.log.separator()
        self.log.info("")
        
        # Run ablation experiments
        suite = FinalFLExperiment(str(self.output_dir / "ablation_runs"))
        
        for config in ablation_configs:
            model_type, clients, byz_frac, attack, rounds, epochs, seed, data_dir, ablation_id = config
            
            self.log.info(f"Running: {ablation_id}")
            suite.run_experiment(
                model_type=model_type,
                num_clients=clients,
                byzantine_fraction=byz_frac,
                attack_type=attack,
                num_rounds=rounds,
                local_epochs=epochs,
                seed=seed,
                data_dir=data_dir
            )
            
            # Save ablation result
            if suite.all_results:
                result = suite.all_results[-1]
                result['ablation_id'] = ablation_id
                self.all_results.append(result)
        
        # Save all ablation results
        with open(self.results_dir / "ablation_study.json", 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'experiments': self.all_results
            }, f, indent=2)
        
        self.log.separator()
        self.log.info("ABLATION STUDY COMPLETE")
        self.log.separator()
        self.log.info(f"Results: {self.results_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()
    
    ablation = AblationStudy(args.output)
    ablation.run_ablation_suite(quick_mode=args.quick)


if __name__ == "__main__":
    main()

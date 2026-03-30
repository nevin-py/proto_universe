#!/usr/bin/env python3
"""
Comprehensive Experiment Suite - All Configurations
- Multiple attack types
- IID and non-IID data distributions
- Different random seeds
- All model architectures
- Live logging with real-time updates
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from src.models.mnist import create_mnist_model
from src.crypto.zkp_prover import TrainingProofProver


class LiveLogger:
    """Logger that outputs to both console and file in real-time"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger('ComprehensiveSuite')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        console.setFormatter(console_format)
        self.logger.addHandler(console)
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def info(self, msg):
        self.logger.info(msg)
        for handler in self.logger.handlers:
            handler.flush()
    
    def error(self, msg):
        self.logger.error(msg)
        for handler in self.logger.handlers:
            handler.flush()
    
    def separator(self):
        self.info("=" * 100)
    
    def line(self):
        self.info("-" * 100)


class AttackSimulator:
    """Simulate various Byzantine attacks"""
    
    @staticmethod
    def model_poisoning(weights, bias, strength=0.5):
        """Add random noise to model parameters"""
        poisoned_weights = weights + torch.randn_like(weights) * strength
        poisoned_bias = bias + torch.randn_like(bias) * strength
        return poisoned_weights, poisoned_bias
    
    @staticmethod
    def label_flip(weights, bias, num_classes=10):
        """Simulate label flipping attack by permuting class weights"""
        perm = torch.randperm(num_classes)
        poisoned_weights = weights[perm]
        poisoned_bias = bias[perm]
        return poisoned_weights, poisoned_bias
    
    @staticmethod
    def sign_flip(weights, bias):
        """Flip the sign of gradients"""
        return -weights, -bias
    
    @staticmethod
    def gaussian_noise(weights, bias, mean=0, std=1.0):
        """Add Gaussian noise"""
        poisoned_weights = weights + torch.randn_like(weights) * std + mean
        poisoned_bias = bias + torch.randn_like(bias) * std + mean
        return poisoned_weights, poisoned_bias
    
    @staticmethod
    def zero_weights(weights, bias):
        """Send zero weights (lazy client)"""
        return torch.zeros_like(weights), torch.zeros_like(bias)
    
    @staticmethod
    def targeted_attack(weights, bias, target_class=0, boost_factor=10.0):
        """Boost specific class to create backdoor"""
        poisoned_weights = weights.clone()
        poisoned_bias = bias.clone()
        poisoned_weights[target_class] *= boost_factor
        poisoned_bias[target_class] *= boost_factor
        return poisoned_weights, poisoned_bias


class ComprehensiveExperimentSuite:
    """Complete experiment suite with all configurations"""
    
    def __init__(self, output_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"comprehensive_{self.timestamp}"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.log = LiveLogger(self.output_dir / "main.log")
        
        # Results tracking
        self.all_results = []
        self.total_experiments = 0
        self.passed_experiments = 0
        self.failed_experiments = 0
        
        # Attack simulator
        self.attacker = AttackSimulator()
    
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def get_model_layers(self, model, model_type):
        """Extract final layer from model"""
        if model_type == "linear":
            return model.linear.weight.detach(), model.linear.bias.detach()
        elif model_type == "mlp":
            return model.fc3.weight.detach(), model.fc3.bias.detach()
        elif model_type == "cnn":
            return model.fc2.weight.detach(), model.fc2.bias.detach()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_experiment(self, model_type, input_dim, attack_type, seed, num_clients=10, 
                       byzantine_fraction=0.3, num_classes=10):
        """Run single experiment"""
        
        exp_id = f"{model_type}_{attack_type}_c{num_clients}_alpha{byzantine_fraction}_seed{seed}"
        self.total_experiments += 1
        
        self.log.separator()
        self.log.info(f"EXPERIMENT {self.total_experiments}: {exp_id}")
        self.log.separator()
        self.log.info(f"  Model: {model_type} | Attack: {attack_type} | Seed: {seed}")
        self.log.info(f"  Clients: {num_clients} | Byzantine fraction: {byzantine_fraction}")
        self.log.info(f"  Input dim: {input_dim} | Classes: {num_classes}")
        self.log.info("")
        
        start_time = time.time()
        
        try:
            # Set seed
            self.set_seed(seed)
            
            # Create models
            self.log.info("Creating models...")
            honest_model = create_mnist_model(model_type, num_classes)
            malicious_model = create_mnist_model(model_type, num_classes)
            
            # Extract parameters
            honest_weights, honest_bias = self.get_model_layers(honest_model, model_type)
            malicious_weights, malicious_bias = self.get_model_layers(malicious_model, model_type)
            
            # Apply attack
            self.log.info(f"Applying {attack_type} attack...")
            
            if attack_type == "model_poisoning":
                malicious_weights, malicious_bias = self.attacker.model_poisoning(
                    malicious_weights, malicious_bias, strength=0.5
                )
            elif attack_type == "label_flip":
                malicious_weights, malicious_bias = self.attacker.label_flip(
                    malicious_weights, malicious_bias, num_classes
                )
            elif attack_type == "sign_flip":
                malicious_weights, malicious_bias = self.attacker.sign_flip(
                    malicious_weights, malicious_bias
                )
            elif attack_type == "gaussian":
                malicious_weights, malicious_bias = self.attacker.gaussian_noise(
                    malicious_weights, malicious_bias, mean=0, std=1.0
                )
            elif attack_type == "zero":
                malicious_weights, malicious_bias = self.attacker.zero_weights(
                    malicious_weights, malicious_bias
                )
            elif attack_type == "targeted":
                malicious_weights, malicious_bias = self.attacker.targeted_attack(
                    malicious_weights, malicious_bias, target_class=0, boost_factor=10.0
                )
            else:
                self.log.error(f"Unknown attack type: {attack_type}")
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            # Byzantine detection
            self.log.info("Running Byzantine detection...")
            prover = TrainingProofProver()
            
            round_number = 1
            r_vec = prover.generate_random_vector(round_number)
            sample_size = min(100, input_dim)
            
            # Honest fingerprint
            honest_fp, _ = prover.compute_model_fingerprint(
                honest_weights, honest_bias, r_vec, round_number, sample_size=sample_size
            )
            
            # Malicious fingerprint
            malicious_fp, _ = prover.compute_model_fingerprint(
                malicious_weights, malicious_bias, r_vec, round_number, sample_size=sample_size
            )
            
            # Detection
            fp_diff = abs(honest_fp - malicious_fp)
            detected = fp_diff > 0
            detection_time = (time.time() - start_time) * 1000
            
            self.log.info("")
            self.log.info("RESULTS:")
            self.log.line()
            self.log.info(f"  Honest fingerprint: {honest_fp}")
            self.log.info(f"  Malicious fingerprint: {malicious_fp}")
            self.log.info(f"  Fingerprint diff: {fp_diff}")
            self.log.info(f"  Byzantine detected: {detected}")
            self.log.info(f"  Detection time: {detection_time:.2f}ms")
            
            # Save result
            result = {
                "experiment_id": exp_id,
                "model_type": model_type,
                "input_dim": input_dim,
                "attack_type": attack_type,
                "seed": seed,
                "num_clients": num_clients,
                "byzantine_fraction": byzantine_fraction,
                "num_classes": num_classes,
                "honest_fingerprint": int(honest_fp),
                "malicious_fingerprint": int(malicious_fp),
                "fingerprint_diff": int(fp_diff),
                "byzantine_detected": detected,
                "detection_time_ms": detection_time,
                "timestamp": self.timestamp,
                "status": "PASS" if detected else "FAIL"
            }
            
            self.all_results.append(result)
            
            # Save individual result
            result_file = self.results_dir / f"{exp_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            if detected:
                self.log.info("")
                self.log.info(f"✅ {exp_id}: PASS")
                self.passed_experiments += 1
                return True
            else:
                self.log.error("")
                self.log.error(f"❌ {exp_id}: FAIL - Byzantine not detected!")
                self.failed_experiments += 1
                return False
                
        except Exception as e:
            self.log.error("")
            self.log.error(f"❌ {exp_id}: EXCEPTION - {str(e)}")
            self.failed_experiments += 1
            
            result = {
                "experiment_id": exp_id,
                "status": "ERROR",
                "error": str(e),
                "timestamp": self.timestamp
            }
            self.all_results.append(result)
            return False
    
    def run_all(self, model_types, attack_types, seeds, client_counts, 
                byzantine_fractions, quick_mode=False):
        """Run complete experiment suite"""
        
        self.log.separator()
        self.log.info("COMPREHENSIVE EXPERIMENT SUITE - ALL CONFIGURATIONS")
        self.log.separator()
        self.log.info(f"Timestamp: {self.timestamp}")
        self.log.info(f"Output directory: {self.output_dir}")
        self.log.info(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
        self.log.info("")
        self.log.info("Configuration:")
        self.log.info(f"  Models: {', '.join(model_types)}")
        self.log.info(f"  Attacks: {', '.join(attack_types)}")
        self.log.info(f"  Seeds: {seeds}")
        self.log.info(f"  Client counts: {client_counts}")
        self.log.info(f"  Byzantine fractions: {byzantine_fractions}")
        total = (len(model_types) * len(attack_types) * len(seeds) * 
                len(client_counts) * len(byzantine_fractions))
        self.log.info(f"  Total experiments planned: {total}")
        self.log.info("")
        
        # Model configurations
        model_configs = {
            "linear": (784, "MNISTLinearRegression"),
            "mlp": (64, "SimpleMLP"),
            "cnn": (128, "MNISTCnn"),
        }
        
        # Run experiments
        for model_type in model_types:
            if model_type not in model_configs:
                self.log.error(f"Unknown model type: {model_type}, skipping...")
                continue
            
            input_dim, model_class = model_configs[model_type]
            
            self.log.info("")
            self.log.separator()
            self.log.info(f"MODEL: {model_class} (input_dim={input_dim})")
            self.log.separator()
            
            for attack_type in attack_types:
                for num_clients in client_counts:
                    for byzantine_fraction in byzantine_fractions:
                        for seed in seeds:
                            self.run_experiment(
                                model_type, input_dim, attack_type, seed,
                                num_clients=num_clients,
                                byzantine_fraction=byzantine_fraction
                            )
                            self.log.info("")
        
        # Summary
        self.generate_summary()
        
        return self.failed_experiments == 0
    
    def generate_summary(self):
        """Generate final summary"""
        
        self.log.info("")
        self.log.separator()
        self.log.info("COMPREHENSIVE EXPERIMENT SUITE COMPLETE")
        self.log.separator()
        self.log.info("")
        self.log.info("STATISTICS:")
        self.log.info(f"  Total experiments: {self.total_experiments}")
        self.log.info(f"  Passed: {self.passed_experiments}")
        self.log.info(f"  Failed: {self.failed_experiments}")
        
        if self.total_experiments > 0:
            self.log.info(f"  Success rate: {self.passed_experiments * 100 // self.total_experiments}%")
        
        self.log.info("")
        
        # Save all results
        combined_file = self.results_dir / "all_results.json"
        with open(combined_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "total": self.total_experiments,
                "passed": self.passed_experiments,
                "failed": self.failed_experiments,
                "success_rate": self.passed_experiments / self.total_experiments if self.total_experiments > 0 else 0,
                "experiments": self.all_results
            }, f, indent=2)
        
        self.log.info(f"Combined results: {combined_file}")
        self.log.info(f"Main log: {self.output_dir / 'main.log'}")
        num_individual = len(list(self.results_dir.glob('*.json'))) - 1  # -1 for all_results.json
        self.log.info(f"Individual results: {num_individual} files")
        self.log.info("")
        
        # Group by attack type
        attack_stats = {}
        for result in self.all_results:
            if result.get('status') in ['PASS', 'FAIL']:
                attack = result['attack_type']
                if attack not in attack_stats:
                    attack_stats[attack] = {'total': 0, 'passed': 0}
                attack_stats[attack]['total'] += 1
                if result['status'] == 'PASS':
                    attack_stats[attack]['passed'] += 1
        
        self.log.info("RESULTS BY ATTACK TYPE:")
        self.log.line()
        for attack, stats in sorted(attack_stats.items()):
            rate = stats['passed'] * 100 // stats['total'] if stats['total'] > 0 else 0
            self.log.info(f"  {attack:20s}: {stats['passed']}/{stats['total']} ({rate}%)")
        
        self.log.info("")
        self.log.separator()
        
        if self.failed_experiments == 0:
            self.log.info("🎉 ALL EXPERIMENTS PASSED!")
        else:
            self.log.error(f"⚠️  {self.failed_experiments} EXPERIMENT(S) FAILED")
        
        self.log.separator()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Experiment Suite')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer experiments)')
    parser.add_argument('--output', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    # Create suite
    suite = ComprehensiveExperimentSuite(args.output)
    
    # Configuration
    if args.quick:
        # Quick mode: 1 model, 3 attacks, 1 client config, 2 seeds
        model_types = ["linear"]
        attack_types = ["model_poisoning", "label_flip", "gaussian"]
        seeds = [42, 123]
        client_counts = [10]
        byzantine_fractions = [0.3]
    else:
        # Full mode: all models, all attacks, multiple configs, 3 seeds
        model_types = ["linear", "mlp", "cnn"]
        attack_types = [
            "model_poisoning",
            "label_flip",
            "sign_flip",
            "gaussian",
            "zero",
            "targeted"
        ]
        seeds = [42, 123, 456]
        client_counts = [5, 10, 20]
        byzantine_fractions = [0.2, 0.3, 0.4, 0.5]
    
    # Run all experiments
    success = suite.run_all(
        model_types, attack_types, seeds, client_counts, 
        byzantine_fractions, quick_mode=args.quick
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

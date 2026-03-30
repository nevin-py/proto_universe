#!/usr/bin/env python3
"""
FINAL Comprehensive Experiment Suite - CORRECT VERSION
- Proper Byzantine detection (compares updates, not absolute models)
- Complete configuration logging
- Real FL training with multiple rounds
- Detailed per-client and per-round logging
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from src.models.mnist import create_mnist_model
from src.crypto.zkp_prover import TrainingProofProver
from src.client.trainer import Trainer


class LiveLogger:
    """Logger with immediate console + file output"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger('FinalSuite')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(console)
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(file_handler)
    
    def info(self, msg):
        self.logger.info(msg)
        for handler in self.logger.handlers:
            handler.flush()
    
    def separator(self): self.info("=" * 100)
    def line(self): self.info("-" * 100)


class FinalFLExperiment:
    """Final FL experiment with correct Byzantine detection"""
    
    def __init__(self, output_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"final_{self.timestamp}"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.log = LiveLogger(self.output_dir / "main.log")
        self.all_results = []
        self.total_experiments = 0
        self.passed_experiments = 0
        
    def load_mnist_data(self, num_clients=10, data_dir="./data"):
        """Load and partition MNIST data for clients"""
        self.log.info(f"Loading MNIST dataset from {data_dir}...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
        # IID partition
        num_items = len(train_dataset) // num_clients
        client_datasets = []
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        
        for i in range(num_clients):
            start = i * num_items
            end = start + num_items
            client_indices = indices[start:end]
            client_datasets.append(Subset(train_dataset, client_indices))
        
        self.log.info(f"  Total training samples: {len(train_dataset)}")
        self.log.info(f"  Partitioned for {num_clients} clients ({num_items} samples each)")
        self.log.info(f"  Data distribution: IID")
        
        return client_datasets, test_dataset
    
    def apply_attack(self, model, attack_type, model_type):
        """Apply Byzantine attack to model"""
        if model_type == "linear":
            weights = model.linear.weight.data
            bias = model.linear.bias.data
        elif model_type == "mlp":
            weights = model.fc3.weight.data
            bias = model.fc3.bias.data
        elif model_type == "cnn":
            weights = model.fc2.weight.data
            bias = model.fc2.bias.data
        
        if attack_type == "model_poisoning":
            weights += torch.randn_like(weights) * 0.5
            bias += torch.randn_like(bias) * 0.5
        elif attack_type == "sign_flip":
            weights.mul_(-1)
            bias.mul_(-1)
        elif attack_type == "gaussian":
            weights += torch.randn_like(weights) * 1.0
            bias += torch.randn_like(bias) * 1.0
        elif attack_type == "scale_attack":
            weights.mul_(10.0)
            bias.mul_(10.0)
    
    def get_model_params(self, model, model_type):
        """Extract final layer parameters"""
        if model_type == "linear":
            return model.linear.weight.detach(), model.linear.bias.detach()
        elif model_type == "mlp":
            return model.fc3.weight.detach(), model.fc3.bias.detach()
        elif model_type == "cnn":
            return model.fc2.weight.detach(), model.fc2.bias.detach()
    
    def run_fl_round(self, global_model, client_datasets, model_type, byzantine_clients,
                     attack_type, round_num, local_epochs=2):
        """Run one FL round with proper Byzantine detection"""
        
        self.log.info(f"  Round {round_num}:")
        self.log.line()
        self.log.info(f"    Training {len(client_datasets)} clients (Byzantine: {byzantine_clients})")
        self.log.info(f"    Attack type: {attack_type}")
        self.log.info("")
        
        # Get global model params for computing updates
        global_weights, global_bias = self.get_model_params(global_model, model_type)
        
        # Initialize ZKP prover
        prover = TrainingProofProver()
        r_vec = prover.generate_random_vector(round_num)
        input_dim = global_weights.shape[1]
        sample_size = min(100, input_dim)
        
        client_updates = []
        detected_byzantine = []
        update_norms = []  # Collect all update norms for statistical detection
        
        for client_id, client_data in enumerate(client_datasets):
            start_time = time.time()
            is_byzantine = client_id in byzantine_clients
            
            self.log.info(f"    Client {client_id} {'[BYZANTINE]' if is_byzantine else '[HONEST]'}:")
            
            # Create local model (copy global)
            local_model = create_mnist_model(model_type, num_classes=10)
            local_model.load_state_dict(global_model.state_dict())
            
            # Train locally
            self.log.info(f"      Training: {local_epochs} epochs, {len(client_data)} samples")
            trainer = Trainer(local_model, learning_rate=0.01)
            train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            for epoch in range(local_epochs):
                metrics = trainer.train(train_loader, num_epochs=1, verbose=False)
                self.log.info(f"        Epoch {epoch+1}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
            
            # Apply attack if Byzantine
            if is_byzantine:
                self.log.info(f"      ⚠️  Applying {attack_type} attack...")
                self.apply_attack(local_model, attack_type, model_type)
            
            # Get trained weights (move to CPU to match global model)
            trained_weights, trained_bias = self.get_model_params(local_model, model_type)
            trained_weights = trained_weights.cpu()
            trained_bias = trained_bias.cpu()
            
            # Compute UPDATE (delta) not absolute model
            update_weights = trained_weights - global_weights
            update_bias = trained_bias - global_bias
            
            # Generate ZKP proof on the UPDATE
            self.log.info(f"      Generating ZKP proof on update...")
            proof_start = time.time()
            
            try:
                # Compute fingerprint of UPDATE
                update_fp, _ = prover.compute_model_fingerprint(
                    update_weights, update_bias, r_vec, round_num, sample_size=sample_size
                )
                
                # Compute update statistics
                update_norm = torch.norm(update_weights).item()
                update_norms.append(update_norm)
                
                proof_time = (time.time() - proof_start) * 1000
                
                # Store update info (detection happens after all clients report)
                client_updates.append({
                    'client_id': client_id,
                    'model': local_model,
                    'is_byzantine': is_byzantine,
                    'detected': False,  # Will be updated after statistical analysis
                    'update_norm': update_norm,
                    'fingerprint': int(update_fp),
                    'proof_time_ms': proof_time,
                    'update_weights': update_weights,
                    'update_bias': update_bias
                })
                
                self.log.info(f"      Update norm: {update_norm:.2f}")
                self.log.info(f"      Fingerprint: {int(update_fp)}")
                self.log.info(f"      Proof time: {proof_time:.2f}ms")
                
            except Exception as e:
                self.log.info(f"      ❌ Proof generation failed: {e}")
            
            total_time = (time.time() - start_time) * 1000
            self.log.info(f"      Client time: {total_time:.2f}ms")
            self.log.info("")
        
        # Statistical Byzantine detection after all clients report
        self.log.line()
        self.log.info(f"  Running Byzantine detection (statistical analysis)...")
        
        # Compute median and std of update norms
        median_norm = np.median(update_norms)
        std_norm = np.std(update_norms)
        
        self.log.info(f"    Update norm statistics:")
        self.log.info(f"      Median: {median_norm:.2f}")
        self.log.info(f"      Std dev: {std_norm:.2f}")
        self.log.info(f"      Detection threshold: {median_norm + 2*std_norm:.2f}")
        self.log.info("")
        
        # Mark suspicious updates (those > median + 2*std)
        for update in client_updates:
            threshold = median_norm + 2 * std_norm
            is_suspicious = update['update_norm'] > threshold or update['update_norm'] > 50.0
            
            update['detected'] = is_suspicious
            
            if is_suspicious:
                detected_byzantine.append(update['client_id'])
                
            # Log detection result
            client_id = update['client_id']
            is_byz = update['is_byzantine']
            
            if is_suspicious and is_byz:
                self.log.info(f"    Client {client_id}: ✅ BYZANTINE DETECTED (norm: {update['update_norm']:.2f})")
            elif is_suspicious and not is_byz:
                self.log.info(f"    Client {client_id}: ⚠️  FALSE POSITIVE (norm: {update['update_norm']:.2f})")
            elif not is_suspicious and is_byz:
                self.log.info(f"    Client {client_id}: ❌ MISSED (norm: {update['update_norm']:.2f})")
            else:
                self.log.info(f"    Client {client_id}: ✅ Verified honest (norm: {update['update_norm']:.2f})")
        
        self.log.info("")
        
        # Filter accepted updates
        accepted_updates = [u for u in client_updates if not u['detected']]
        
        self.log.line()
        self.log.info(f"  Round {round_num} Summary:")
        self.log.info(f"    Total clients: {len(client_datasets)}")
        self.log.info(f"    Byzantine clients: {len(byzantine_clients)} - {byzantine_clients}")
        self.log.info(f"    Detected Byzantine: {len(detected_byzantine)} - {detected_byzantine}")
        
        # Calculate true positives, false positives, etc.
        true_positives = len([c for c in detected_byzantine if c in byzantine_clients])
        false_positives = len([c for c in detected_byzantine if c not in byzantine_clients])
        false_negatives = len([c for c in byzantine_clients if c not in detected_byzantine])
        
        self.log.info(f"    True positives: {true_positives}")
        self.log.info(f"    False positives: {false_positives}")
        self.log.info(f"    False negatives: {false_negatives}")
        self.log.info(f"    Accepted for aggregation: {len(accepted_updates)}")
        
        # Aggregate accepted updates
        if accepted_updates:
            self.log.info(f"  Aggregating {len(accepted_updates)} updates...")
            global_dict = global_model.state_dict()
            
            for key in global_dict.keys():
                global_dict[key] = torch.stack([
                    u['model'].state_dict()[key].float() for u in accepted_updates
                ]).mean(dim=0)
            
            global_model.load_state_dict(global_dict)
            self.log.info(f"  ✅ Global model updated")
        else:
            self.log.info(f"  ⚠️  No updates accepted - global model unchanged")
        
        self.log.info("")
        
        return {
            'round': round_num,
            'byzantine_clients': byzantine_clients,
            'detected_byzantine': detected_byzantine,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'num_accepted': len(accepted_updates)
        }
    
    def run_experiment(self, model_type, num_clients=10, byzantine_fraction=0.3,
                       attack_type="model_poisoning", num_rounds=3, local_epochs=2,
                       seed=42, data_dir="./data"):
        """Run complete FL experiment"""
        
        exp_id = f"{model_type}_c{num_clients}_alpha{byzantine_fraction:.1f}_{attack_type}_r{num_rounds}_seed{seed}"
        self.total_experiments += 1
        
        self.log.separator()
        self.log.info(f"EXPERIMENT {self.total_experiments}: {exp_id}")
        self.log.separator()
        self.log.info("")
        self.log.info("CONFIGURATION:")
        self.log.line()
        self.log.info(f"  Model architecture: {model_type}")
        self.log.info(f"  Number of clients: {num_clients}")
        self.log.info(f"  Byzantine fraction: {byzantine_fraction} ({int(num_clients * byzantine_fraction)} Byzantine clients)")
        self.log.info(f"  Attack type: {attack_type}")
        self.log.info(f"  FL rounds: {num_rounds}")
        self.log.info(f"  Local epochs per round: {local_epochs}")
        self.log.info(f"  Batch size: 32")
        self.log.info(f"  Learning rate: 0.01")
        self.log.info(f"  Random seed: {seed}")
        self.log.info(f"  Data directory: {data_dir}")
        self.log.info(f"  Data distribution: IID")
        self.log.info(f"  Dataset: MNIST")
        self.log.info("")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        start_time = time.time()
        
        try:
            # Load data
            client_datasets, test_dataset = self.load_mnist_data(num_clients, data_dir)
            
            # Determine Byzantine clients
            num_byzantine = int(num_clients * byzantine_fraction)
            byzantine_clients = list(range(num_byzantine))
            
            self.log.info(f"Byzantine clients designated: {byzantine_clients}")
            self.log.info("")
            
            # Initialize global model
            self.log.info(f"Initializing global {model_type} model...")
            global_model = create_mnist_model(model_type, num_classes=10)
            self.log.info("")
            
            # Run FL rounds
            self.log.separator()
            self.log.info("FEDERATED LEARNING ROUNDS")
            self.log.separator()
            
            round_results = []
            for round_num in range(1, num_rounds + 1):
                round_result = self.run_fl_round(
                    global_model, client_datasets, model_type,
                    byzantine_clients, attack_type, round_num, local_epochs
                )
                round_results.append(round_result)
            
            # Calculate overall metrics
            total_tp = sum(r['true_positives'] for r in round_results)
            total_fp = sum(r['false_positives'] for r in round_results)
            total_fn = sum(r['false_negatives'] for r in round_results)
            total_byzantine = len(byzantine_clients) * num_rounds
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            total_time = time.time() - start_time
            
            self.log.separator()
            self.log.info("EXPERIMENT COMPLETE")
            self.log.separator()
            self.log.info(f"Total time: {total_time:.2f}s")
            self.log.info("")
            self.log.info("DETECTION METRICS:")
            self.log.line()
            self.log.info(f"  True Positives: {total_tp}")
            self.log.info(f"  False Positives: {total_fp}")
            self.log.info(f"  False Negatives: {total_fn}")
            self.log.info(f"  Precision: {precision*100:.1f}%")
            self.log.info(f"  Recall: {recall*100:.1f}%")
            self.log.info(f"  F1 Score: {f1_score:.3f}")
            self.log.info("")
            
            # Save results
            result = {
                'experiment_id': exp_id,
                'configuration': {
                    'model_type': model_type,
                    'num_clients': num_clients,
                    'byzantine_fraction': byzantine_fraction,
                    'attack_type': attack_type,
                    'num_rounds': num_rounds,
                    'local_epochs': local_epochs,
                    'seed': seed,
                    'data_dir': data_dir,
                    'data_distribution': 'IID',
                    'dataset': 'MNIST'
                },
                'byzantine_clients': byzantine_clients,
                'round_results': round_results,
                'metrics': {
                    'true_positives': total_tp,
                    'false_positives': total_fp,
                    'false_negatives': total_fn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                },
                'total_time_seconds': total_time,
                'status': 'PASS' if recall > 0.8 else 'PARTIAL' if recall > 0 else 'FAIL'
            }
            
            self.all_results.append(result)
            
            result_file = self.results_dir / f"{exp_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            if recall > 0.8:
                self.log.info(f"✅ {exp_id}: PASS (Recall: {recall*100:.1f}%)")
                self.passed_experiments += 1
            else:
                self.log.info(f"⚠️  {exp_id}: PARTIAL (Recall: {recall*100:.1f}%)")
            
            self.log.separator()
            self.log.info("")
            
        except Exception as e:
            self.log.info(f"❌ {exp_id}: ERROR - {e}")
            import traceback
            self.log.info(traceback.format_exc())
    
    def run_all(self, quick_mode=False):
        """Run experiment suite"""
        
        self.log.separator()
        self.log.info("FINAL COMPREHENSIVE FL EXPERIMENT SUITE - ADDRESSING REVIEWER CONCERNS")
        self.log.separator()
        self.log.info(f"Timestamp: {self.timestamp}")
        self.log.info(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
        self.log.info(f"Output: {self.output_dir}")
        self.log.info("")
        
        # All attack types
        attacks = ["model_poisoning", "sign_flip", "gaussian", "scale_attack"]
        
        if quick_mode:
            # Quick: One attack per model (testing infrastructure)
            configs = [
                ("linear", 10, 0.3, "model_poisoning", 3, 2, 42, "./data"),
                ("mlp", 10, 0.3, "sign_flip", 3, 2, 123, "./data"),
                ("cnn", 10, 0.3, "gaussian", 3, 2, 456, "./data"),
            ]
        else:
            # FULL COMPREHENSIVE: All attacks × All models × Multiple scales
            # Rounds: Linear=15, MLP=18, CNN=20 (based on model complexity)
            # Format: (model, clients, byz_frac, attack, rounds, local_epochs, seed, data_dir)
            configs = []
            
            # ===== LINEAR MODEL (15 rounds) =====
            # Scale experiments: 15, 30, 50 clients
            for attack in attacks:
                configs.append(("linear", 15, 0.2, attack, 15, 2, 42, "./data"))
                configs.append(("linear", 30, 0.2, attack, 15, 2, 123, "./data"))
                configs.append(("linear", 50, 0.2, attack, 15, 2, 456, "./data"))
            
            # Byzantine fraction experiments: 0.2, 0.3, 0.4
            for attack in attacks:
                configs.append(("linear", 30, 0.3, attack, 15, 2, 789, "./data"))
                configs.append(("linear", 30, 0.4, attack, 15, 2, 101, "./data"))
            
            # ===== MLP MODEL (18 rounds - more complex) =====
            # Scale experiments: 15, 30, 50 clients
            for attack in attacks:
                configs.append(("mlp", 15, 0.2, attack, 18, 2, 42, "./data"))
                configs.append(("mlp", 30, 0.2, attack, 18, 2, 123, "./data"))
                configs.append(("mlp", 50, 0.2, attack, 18, 2, 456, "./data"))
            
            # Byzantine fraction experiments: 0.2, 0.3, 0.4
            for attack in attacks:
                configs.append(("mlp", 30, 0.3, attack, 18, 2, 789, "./data"))
                configs.append(("mlp", 30, 0.4, attack, 18, 2, 101, "./data"))
            
            # ===== CNN MODEL (20 rounds - most complex) =====
            # Scale experiments: 15, 30, 50 clients
            for attack in attacks:
                configs.append(("cnn", 15, 0.2, attack, 20, 2, 42, "./data"))
                configs.append(("cnn", 30, 0.2, attack, 20, 2, 123, "./data"))
                configs.append(("cnn", 50, 0.2, attack, 20, 2, 456, "./data"))
            
            # Byzantine fraction experiments: 0.2, 0.3, 0.4
            for attack in attacks:
                configs.append(("cnn", 30, 0.3, attack, 20, 2, 789, "./data"))
                configs.append(("cnn", 30, 0.4, attack, 20, 2, 101, "./data"))
        
        total_configs = len(configs)
        self.log.info(f"Total experiments to run: {total_configs}")
        self.log.info("")
        
        # Breakdown
        if not quick_mode:
            self.log.info("Experiment breakdown:")
            self.log.info(f"  Linear (15 rounds): {sum(1 for c in configs if c[0]=='linear')} experiments")
            self.log.info(f"  MLP (18 rounds): {sum(1 for c in configs if c[0]=='mlp')} experiments")
            self.log.info(f"  CNN (20 rounds): {sum(1 for c in configs if c[0]=='cnn')} experiments")
            self.log.info("")
            self.log.info("Scale coverage:")
            self.log.info(f"  15 clients: {sum(1 for c in configs if c[1]==15)} experiments")
            self.log.info(f"  30 clients: {sum(1 for c in configs if c[1]==30)} experiments")
            self.log.info(f"  50 clients: {sum(1 for c in configs if c[1]==50)} experiments")
            self.log.info("")
            self.log.info("Attack type coverage (per model):")
            for attack in attacks:
                count = sum(1 for c in configs if c[3]==attack)
                self.log.info(f"  {attack}: {count} experiments")
            self.log.info("")
            self.log.separator()
            self.log.info("")
        
        for config in configs:
            self.run_experiment(*config)
        
        # Final summary
        self.log.separator()
        self.log.info("SUITE COMPLETE")
        self.log.separator()
        self.log.info(f"Total experiments: {self.total_experiments}")
        self.log.info(f"Passed: {self.passed_experiments}")
        self.log.info(f"Results directory: {self.results_dir}")
        self.log.info(f"Main log: {self.output_dir / 'main.log'}")
        self.log.separator()
        
        # Save combined
        with open(self.results_dir / "all_results.json", 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'total': self.total_experiments,
                'passed': self.passed_experiments,
                'experiments': self.all_results
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Final FL Experiment Suite')
    parser.add_argument('--quick', action='store_true', help='Quick mode (3 experiments)')
    parser.add_argument('--output', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    suite = FinalFLExperiment(args.output)
    suite.run_all(quick_mode=args.quick)


if __name__ == "__main__":
    main()

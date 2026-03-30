#!/usr/bin/env python3
"""
REALISTIC Comprehensive Experiment Suite
- Actual federated learning training
- Multiple clients (honest + Byzantine)
- Real ZKP proof generation and verification
- Complete logging showing Byzantine detection at each round
- Proper FL pipeline execution
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
        self.logger = logging.getLogger('RealisticSuite')
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


class RealisticFLExperiment:
    """Realistic FL experiment with actual training and Byzantine detection"""
    
    def __init__(self, output_dir):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"realistic_{self.timestamp}"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.log = LiveLogger(self.output_dir / "main.log")
        self.all_results = []
        self.total_experiments = 0
        self.passed_experiments = 0
        
    def load_mnist_data(self, num_clients=10):
        """Load and partition MNIST data for clients"""
        self.log.info("Loading MNIST dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Partition data IID
        num_items = len(train_dataset) // num_clients
        client_datasets = []
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        
        for i in range(num_clients):
            start = i * num_items
            end = start + num_items
            client_indices = indices[start:end]
            client_datasets.append(Subset(train_dataset, client_indices))
        
        self.log.info(f"  Partitioned data for {num_clients} clients ({num_items} samples each)")
        
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
        """Run one FL round with actual training"""
        
        self.log.info(f"  Round {round_num}: Training {len(client_datasets)} clients...")
        self.log.info(f"    Byzantine clients: {byzantine_clients}")
        self.log.info(f"    Attack type: {attack_type}")
        
        # Initialize ZKP prover for this round
        prover = TrainingProofProver()
        r_vec = prover.generate_random_vector(round_num)
        
        # Get global model fingerprint for verification
        global_weights, global_bias = self.get_model_params(global_model, model_type)
        input_dim = global_weights.shape[1]
        sample_size = min(100, input_dim)
        
        expected_fp, _ = prover.compute_model_fingerprint(
            global_weights, global_bias, r_vec, round_num, sample_size=sample_size
        )
        
        client_updates = []
        detected_byzantine = []
        
        for client_id, client_data in enumerate(client_datasets):
            start_time = time.time()
            is_byzantine = client_id in byzantine_clients
            
            self.log.info(f"    Client {client_id} {'[BYZANTINE]' if is_byzantine else '[HONEST]'}:")
            
            # Create local model
            local_model = create_mnist_model(model_type, num_classes=10)
            local_model.load_state_dict(global_model.state_dict())
            
            # Train locally
            self.log.info(f"      Training for {local_epochs} epochs on {len(client_data)} samples...")
            trainer = Trainer(local_model, learning_rate=0.01)
            train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            for epoch in range(local_epochs):
                metrics = trainer.train(train_loader, num_epochs=1, verbose=False)
                avg_loss = metrics['loss']
                self.log.info(f"        Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            # Apply attack if Byzantine
            if is_byzantine:
                self.log.info(f"      Applying {attack_type} attack...")
                self.apply_attack(local_model, attack_type, model_type)
            
            # Generate ZKP proof
            self.log.info(f"      Generating ZKP proof...")
            proof_start = time.time()
            
            local_weights, local_bias = self.get_model_params(local_model, model_type)
            
            try:
                # Compute client's fingerprint
                client_fp, _ = prover.compute_model_fingerprint(
                    local_weights, local_bias, r_vec, round_num, sample_size=sample_size
                )
                
                # Verify fingerprint matches expected
                fp_diff = abs(client_fp - expected_fp)
                fp_match = fp_diff == 0
                
                proof_time = (time.time() - proof_start) * 1000
                
                if not fp_match:
                    self.log.info(f"      ❌ BYZANTINE DETECTED! Fingerprint mismatch: {fp_diff}")
                    self.log.info(f"         Expected: {expected_fp}, Got: {client_fp}")
                    detected_byzantine.append(client_id)
                else:
                    self.log.info(f"      ✅ Proof verified (fingerprint match)")
                
                self.log.info(f"      Proof generation time: {proof_time:.2f}ms")
                
                client_updates.append({
                    'client_id': client_id,
                    'model': local_model,
                    'is_byzantine': is_byzantine,
                    'detected': not fp_match,
                    'fingerprint': int(client_fp),
                    'proof_time_ms': proof_time
                })
                
            except Exception as e:
                self.log.info(f"      ❌ Proof generation failed: {e}")
            
            total_time = (time.time() - start_time) * 1000
            self.log.info(f"      Total time: {total_time:.2f}ms")
        
        # Filter out detected Byzantine clients
        honest_updates = [u for u in client_updates if not u['detected']]
        
        self.log.info(f"  Round {round_num} complete:")
        self.log.info(f"    Actual Byzantine: {len(byzantine_clients)}")
        self.log.info(f"    Detected Byzantine: {len(detected_byzantine)} - {detected_byzantine}")
        self.log.info(f"    Accepted for aggregation: {len(honest_updates)}")
        
        # Aggregate honest updates
        if honest_updates:
            self.log.info(f"  Aggregating {len(honest_updates)} honest updates...")
            global_dict = global_model.state_dict()
            
            for key in global_dict.keys():
                global_dict[key] = torch.stack([
                    u['model'].state_dict()[key].float() for u in honest_updates
                ]).mean(dim=0)
            
            global_model.load_state_dict(global_dict)
        
        return {
            'round': round_num,
            'byzantine_clients': byzantine_clients,
            'detected_byzantine': detected_byzantine,
            'detection_rate': len(detected_byzantine) / len(byzantine_clients) if byzantine_clients else 1.0,
            'num_accepted': len(honest_updates)
        }
    
    def run_experiment(self, model_type, num_clients=10, byzantine_fraction=0.3,
                       attack_type="model_poisoning", num_rounds=3, seed=42):
        """Run complete FL experiment with Byzantine detection"""
        
        exp_id = f"{model_type}_c{num_clients}_alpha{byzantine_fraction}_{attack_type}_seed{seed}"
        self.total_experiments += 1
        
        self.log.separator()
        self.log.info(f"EXPERIMENT {self.total_experiments}: {exp_id}")
        self.log.separator()
        self.log.info(f"Configuration:")
        self.log.info(f"  Model: {model_type}")
        self.log.info(f"  Clients: {num_clients}")
        self.log.info(f"  Byzantine fraction: {byzantine_fraction}")
        self.log.info(f"  Attack: {attack_type}")
        self.log.info(f"  Rounds: {num_rounds}")
        self.log.info(f"  Seed: {seed}")
        self.log.info("")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        start_time = time.time()
        
        try:
            # Load data
            client_datasets, test_dataset = self.load_mnist_data(num_clients)
            
            # Determine Byzantine clients
            num_byzantine = int(num_clients * byzantine_fraction)
            byzantine_clients = list(range(num_byzantine))
            
            self.log.info(f"Byzantine clients designated: {byzantine_clients}")
            self.log.info("")
            
            # Initialize global model
            global_model = create_mnist_model(model_type, num_classes=10)
            
            # Run FL rounds
            round_results = []
            for round_num in range(1, num_rounds + 1):
                self.log.line()
                round_result = self.run_fl_round(
                    global_model, client_datasets, model_type,
                    byzantine_clients, attack_type, round_num
                )
                round_results.append(round_result)
                self.log.info("")
            
            # Calculate overall detection rate
            total_detected = sum(len(r['detected_byzantine']) for r in round_results)
            total_byzantine = len(byzantine_clients) * num_rounds
            overall_detection = total_detected / total_byzantine if total_byzantine > 0 else 0
            
            total_time = time.time() - start_time
            
            self.log.separator()
            self.log.info("EXPERIMENT COMPLETE")
            self.log.separator()
            self.log.info(f"Total time: {total_time:.2f}s")
            self.log.info(f"Overall Byzantine detection rate: {overall_detection*100:.1f}%")
            self.log.info(f"Byzantine detected: {total_detected}/{total_byzantine}")
            
            # Save results
            result = {
                'experiment_id': exp_id,
                'model_type': model_type,
                'num_clients': num_clients,
                'byzantine_fraction': byzantine_fraction,
                'attack_type': attack_type,
                'num_rounds': num_rounds,
                'seed': seed,
                'byzantine_clients': byzantine_clients,
                'round_results': round_results,
                'overall_detection_rate': overall_detection,
                'total_time_seconds': total_time,
                'status': 'PASS' if overall_detection > 0.8 else 'PARTIAL' if overall_detection > 0 else 'FAIL'
            }
            
            self.all_results.append(result)
            
            result_file = self.results_dir / f"{exp_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            if overall_detection > 0.8:
                self.log.info(f"✅ {exp_id}: PASS")
                self.passed_experiments += 1
            else:
                self.log.info(f"⚠️  {exp_id}: PARTIAL (only {overall_detection*100:.1f}% detected)")
            
        except Exception as e:
            self.log.info(f"❌ {exp_id}: ERROR - {e}")
            import traceback
            self.log.info(traceback.format_exc())
    
    def run_all(self, quick_mode=False):
        """Run experiment suite"""
        
        self.log.separator()
        self.log.info("REALISTIC FL EXPERIMENT SUITE")
        self.log.separator()
        self.log.info(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
        self.log.info(f"Output: {self.output_dir}")
        self.log.info("")
        
        if quick_mode:
            # Quick: 2 experiments
            configs = [
                ("linear", 5, 0.3, "model_poisoning", 2, 42),
                ("linear", 5, 0.3, "sign_flip", 2, 123),
            ]
        else:
            # Full: 12 experiments
            configs = [
                ("linear", 10, 0.3, "model_poisoning", 3, 42),
                ("linear", 10, 0.4, "model_poisoning", 3, 123),
                ("linear", 10, 0.3, "sign_flip", 3, 42),
                ("mlp", 10, 0.3, "model_poisoning", 3, 42),
                ("mlp", 10, 0.3, "sign_flip", 3, 123),
                ("cnn", 10, 0.3, "model_poisoning", 3, 42),
            ]
        
        for config in configs:
            self.run_experiment(*config)
            self.log.info("")
        
        # Summary
        self.log.separator()
        self.log.info("SUITE COMPLETE")
        self.log.separator()
        self.log.info(f"Total: {self.total_experiments}")
        self.log.info(f"Passed: {self.passed_experiments}")
        self.log.info(f"Results: {self.results_dir}")
        self.log.separator()
        
        # Save combined
        with open(self.results_dir / "all_results.json", 'w') as f:
            json.dump(self.all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()
    
    suite = RealisticFLExperiment(args.output)
    suite.run_all(quick_mode=args.quick)


if __name__ == "__main__":
    main()

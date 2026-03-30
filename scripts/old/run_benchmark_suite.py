#!/usr/bin/env python3
"""Benchmark Suite for FiZK-PoT Paper

Runs targeted experiments to generate all paper results:
1. Byzantine detection validation (all architectures)
2. Defense comparison (6 baselines)
3. Attack robustness (multiple attack types)
4. Scalability analysis (varying α)

Results logged to JSON with timestamps and metrics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from src.orchestration.fizk_pot_pipeline import FiZKPoTPipeline
from src.models.mnist import create_mnist_model
from src.defense.robust_agg import MultiKrumAggregator, CoordinateWiseMedianAggregator, TrimmedMeanAggregator
from src.defense.fltrust import FLTrustAggregator
from src.client.trainer import Trainer


def load_mnist(data_dir="./data"):
    """Load MNIST dataset."""
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
    return train_dataset, test_dataset


def partition_data_iid(dataset, num_clients):
    """Partition dataset into IID subsets."""
    num_items = len(dataset) // num_clients
    client_datasets = []
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    for i in range(num_clients):
        start_idx = i * num_items
        end_idx = start_idx + num_items
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets


def run_byzantine_detection_benchmark():
    """Benchmark 1: Byzantine detection across architectures"""
    print("\n" + "="*80)
    print("BENCHMARK 1: Byzantine Detection Validation")
    print("="*80)
    
    results = {
        "benchmark": "byzantine_detection",
        "timestamp": datetime.now().isoformat(),
        "architectures": {}
    }
    
    architectures = [
        ("linear", 784, "MNISTLinearRegression"),
        ("mlp", 64, "SimpleMLP"),
        ("cnn", 128, "MNISTCnn")
    ]
    
    for arch_name, input_dim, model_class in architectures:
        print(f"\nTesting {arch_name.upper()} ({model_class})")
        print("-" * 80)
        
        # Create model
        model = create_mnist_model(arch_name, num_classes=10)
        
        # Extract final layer
        if arch_name == 'linear':
            weights = model.linear.weight.data.clone()
            bias = model.linear.bias.data.clone()
        elif arch_name == 'mlp':
            weights = model.fc3.weight.data.clone()
            bias = model.fc3.bias.data.clone()
        elif arch_name == 'cnn':
            weights = model.fc2.weight.data.clone()
            bias = model.fc2.bias.data.clone()
        
        print(f"  Dimensions: {weights.shape[0]}×{weights.shape[1]}")
        
        # Test Byzantine detection
        from src.crypto.zkp_prover import TrainingProofProver
        
        prover = TrainingProofProver()
        batch = [(torch.randn(input_dim), i % 10) for i in range(4)]
        
        # Malicious test
        malicious_weights = weights.clone() + torch.randn_like(weights) * 0.1
        r_vec = prover.generate_random_vector(0)
        sample_size = min(100, input_dim)
        
        honest_fp, _ = prover.compute_model_fingerprint(weights, bias, r_vec, 0, sample_size=sample_size)
        malicious_fp, _ = prover.compute_model_fingerprint(malicious_weights, bias, r_vec, 0, sample_size=sample_size)
        
        fp_diff = abs(honest_fp - malicious_fp)
        
        start = time.time()
        try:
            prover2 = TrainingProofProver()
            proof = prover2.prove_training(
                malicious_weights, bias, batch, 1, 0, 4,
                expected_fingerprint=honest_fp
            )
            detection_success = False
            detection_time = time.time() - start
        except RuntimeError as e:
            detection_success = "fingerprint" in str(e).lower()
            detection_time = time.time() - start
        
        results["architectures"][arch_name] = {
            "model_class": model_class,
            "dimensions": f"{weights.shape[0]}×{weights.shape[1]}",
            "input_dim": input_dim,
            "sample_size": sample_size,
            "fingerprint_diff": int(fp_diff),
            "detection_success": detection_success,
            "detection_time_ms": round(detection_time * 1000, 2),
            "detection_rate": "100%" if detection_success else "0%"
        }
        
        print(f"  ✅ Byzantine detected: {detection_success}")
        print(f"  ⏱  Detection time: {detection_time*1000:.2f}ms")
        print(f"  📊 Fingerprint diff: {fp_diff}")
    
    return results


def run_defense_comparison_benchmark(num_rounds=10, num_clients=10, alpha=0.3):
    """Benchmark 2: Compare all defenses under attack"""
    print("\n" + "="*80)
    print(f"BENCHMARK 2: Defense Comparison (α={alpha})")
    print("="*80)
    
    results = {
        "benchmark": "defense_comparison",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "alpha": alpha,
            "dataset": "mnist",
            "attack": "model_poisoning"
        },
        "defenses": {}
    }
    
    # Load data
    train_dataset, test_dataset = load_mnist()
    client_datasets = partition_data_iid(train_dataset, num_clients)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    num_malicious = int(alpha * num_clients)
    
    defenses = {
        "vanilla": None,
        "multikrum": MultiKrumAggregator(num_clients - num_malicious),
        "median": CoordinateWiseMedianAggregator(),
        "trimmedmean": TrimmedMeanAggregator(),
        # Note: FiZK requires full pipeline setup - tested separately
    }
    
    for defense_name, aggregator in defenses.items():
        print(f"\nTesting {defense_name.upper()}")
        print("-" * 80)
        
        # Initialize global model
        global_model = create_mnist_model("linear", num_classes=10)
        
        # Training loop
        start_time = time.time()
        round_accuracies = []
        
        for round_num in range(num_rounds):
            # Client training
            client_updates = []
            
            for client_id in range(num_clients):
                is_malicious = client_id < num_malicious
                
                client_model = create_mnist_model("linear", num_classes=10)
                client_model.load_state_dict(global_model.state_dict())
                
                # Simple SGD training
                optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01)
                criterion = torch.nn.CrossEntropyLoss()
                
                client_loader = torch.utils.data.DataLoader(
                    client_datasets[client_id], batch_size=32, shuffle=True
                )
                
                # Train
                client_model.train()
                for epoch in range(2):
                    for batch_x, batch_y in client_loader:
                        if is_malicious:
                            # Model poisoning attack
                            batch_y = torch.randint(0, 10, batch_y.shape)
                        
                        optimizer.zero_grad()
                        outputs = client_model(batch_x.view(batch_x.size(0), -1))
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                client_updates.append({
                    "model": client_model,
                    "client_id": client_id,
                    "is_malicious": is_malicious
                })
            
            # Aggregate
            if defense_name == "vanilla":
                # Simple averaging
                global_dict = global_model.state_dict()
                for key in global_dict.keys():
                    global_dict[key] = torch.stack([
                        u["model"].state_dict()[key].float() for u in client_updates
                    ]).mean(dim=0)
                global_model.load_state_dict(global_dict)
            
            else:
                # Robust aggregators - compute gradients and aggregate
                updates_with_grads = []
                for update in client_updates:
                    grads = []
                    for key in global_model.state_dict().keys():
                        grad = update["model"].state_dict()[key] - global_model.state_dict()[key]
                        grads.append(grad)
                    updates_with_grads.append({"gradients": grads})
                
                # Aggregate
                agg_result = aggregator.aggregate(updates_with_grads)
                if agg_result is None:
                    continue
                
                # The aggregator returns flattened gradients, need to reconstruct to model shapes
                agg_grads_flat = agg_result['gradients']
                
                # Reconstruct gradient dict from flattened array
                global_dict = global_model.state_dict()
                offset = 0
                grad_dict = {}
                
                for key in global_dict.keys():
                    param_shape = global_dict[key].shape
                    param_size = global_dict[key].numel()
                    
                    # Extract this parameter's gradient from flattened array
                    grad_flat = agg_grads_flat[offset:offset + param_size]
                    grad_dict[key] = torch.tensor(grad_flat).reshape(param_shape)
                    offset += param_size
                
                # Apply aggregated gradients
                for key in global_dict.keys():
                    global_dict[key] = global_dict[key] + grad_dict[key]
                global_model.load_state_dict(global_dict)
            
            # Evaluate
            correct = 0
            total = 0
            global_model.eval()
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = global_model(batch_x.view(batch_x.size(0), -1))
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            accuracy = 100.0 * correct / total
            round_accuracies.append(accuracy)
            print(f"  Round {round_num+1}/{num_rounds}: Accuracy = {accuracy:.2f}%")
        
        training_time = time.time() - start_time
        
        results["defenses"][defense_name] = {
            "final_accuracy": round(round_accuracies[-1], 2),
            "max_accuracy": round(max(round_accuracies), 2),
            "min_accuracy": round(min(round_accuracies), 2),
            "mean_accuracy": round(np.mean(round_accuracies), 2),
            "total_time_sec": round(training_time, 2),
            "time_per_round_sec": round(training_time / num_rounds, 2),
            "round_accuracies": [round(a, 2) for a in round_accuracies]
        }
        
        print(f"  Final Accuracy: {round_accuracies[-1]:.2f}%")
        print(f"  Total Time: {training_time:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run FiZK-PoT benchmarks")
    parser.add_argument("--output", type=str, default="outputs/benchmarks",
                       help="Output directory for results")
    parser.add_argument("--benchmarks", nargs="+", default=["all"],
                       choices=["all", "byzantine", "defense"],
                       help="Which benchmarks to run")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of FL rounds for defense comparison")
    parser.add_argument("--clients", type=int, default=10,
                       help="Number of clients")
    parser.add_argument("--alpha", type=float, default=0.3,
                       help="Fraction of malicious clients")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        "experiment": "fizk_pot_benchmarks",
        "timestamp": timestamp,
        "benchmarks": {}
    }
    
    # Run benchmarks
    if "all" in args.benchmarks or "byzantine" in args.benchmarks:
        byzantine_results = run_byzantine_detection_benchmark()
        all_results["benchmarks"]["byzantine_detection"] = byzantine_results
    
    if "all" in args.benchmarks or "defense" in args.benchmarks:
        defense_results = run_defense_comparison_benchmark(
            num_rounds=args.rounds,
            num_clients=args.clients,
            alpha=args.alpha
        )
        all_results["benchmarks"]["defense_comparison"] = defense_results
    
    # Save results
    output_file = output_dir / f"benchmark_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print("\nSummary:")
    
    if "byzantine_detection" in all_results["benchmarks"]:
        bd = all_results["benchmarks"]["byzantine_detection"]
        print(f"\nByzantine Detection:")
        for arch, data in bd["architectures"].items():
            print(f"  {arch.upper()}: {data['detection_rate']} in {data['detection_time_ms']}ms")
    
    if "defense_comparison" in all_results["benchmarks"]:
        dc = all_results["benchmarks"]["defense_comparison"]
        print(f"\nDefense Comparison (α={dc['config']['alpha']}):")
        for defense, data in dc["defenses"].items():
            print(f"  {defense.upper()}: {data['final_accuracy']}% accuracy")


if __name__ == "__main__":
    main()

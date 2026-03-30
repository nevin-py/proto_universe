"""Test script for FiZK-PoT pipeline validation

Quick test to verify the PoT pipeline works correctly before running
comprehensive experiments.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.fizk_pot_pipeline import FiZKPoTPipeline
from src.models.mnist import create_mnist_model
from src.client.trainer import Trainer
import copy


def test_pot_pipeline():
    """Test FiZK-PoT pipeline with minimal configuration."""
    print("="*80)
    print("FiZK-PoT Pipeline Test")
    print("="*80)
    
    # Configuration
    num_clients = 5
    num_byzantine = 2
    num_rounds = 3
    pot_batch_size = 4  # Small for fast testing
    
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients} ({num_byzantine} Byzantine)")
    print(f"  Rounds: {num_rounds}")
    print(f"  PoT batch size: {pot_batch_size}")
    
    # Load MNIST
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    # Partition data (IID)
    n = len(train_dataset)
    indices = np.random.permutation(n).tolist()
    shard_size = n // num_clients
    client_data = {}
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        client_data[i] = Subset(train_dataset, indices[start:end])
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Create model
    print("Creating model...")
    global_model = create_mnist_model('linear')
    
    # Evaluate initial accuracy
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    initial_accuracy = correct / total
    print(f"Initial accuracy: {initial_accuracy:.4f}")
    
    # Initialize pipeline
    print("\nInitializing FiZK-PoT pipeline...")
    pipeline = FiZKPoTPipeline(
        global_model=global_model,
        num_clients=num_clients,
        pot_batch_size=pot_batch_size
    )
    
    byzantine_ids = set(range(num_byzantine))
    print(f"Byzantine clients: {sorted(byzantine_ids)}")
    
    accuracy = initial_accuracy  # Initialize
    byzantine_caught = 0
    
    # Run training rounds
    for round_num in range(num_rounds):
        print(f"\n{'─'*80}")
        print(f"Round {round_num}")
        print(f"{'─'*80}")
        
        pipeline.current_round = round_num
        pipeline.reset_round()
        
        # Get model parameters for PoT
        params = list(global_model.parameters())
        weights = params[0]  # Linear layer weights
        bias = params[1]     # Linear layer bias
        
        # Phase 1: Data commitments
        print("Phase 1: Collecting data commitments...")
        commitments = {}
        client_train_data = {}
        
        for cid in range(num_clients):
            # Get a small batch of training data for PoT
            loader = DataLoader(client_data[cid], batch_size=pot_batch_size, shuffle=True)
            batch_data = []
            for x, y in loader:
                for i in range(len(x)):
                    batch_data.append((x[i], int(y[i])))
                    if len(batch_data) >= pot_batch_size:
                        break
                if len(batch_data) >= pot_batch_size:
                    break
            
            client_train_data[cid] = batch_data
            
            commitment = pipeline.phase1_client_commit_data(
                client_id=cid,
                train_data=batch_data,
                round_number=round_num
            )
            commitments[cid] = commitment
        
        merkle_root = pipeline.phase1_collect_commitments(commitments, round_num)
        print(f"  Merkle root: {merkle_root[:16]}...")
        
        # Phase 2: Training + PoT proof generation
        print("Phase 2: Training and PoT proof generation...")
        
        for cid in range(num_clients):
            # Train client model
            client_model = copy.deepcopy(global_model)
            trainer = Trainer(model=client_model, learning_rate=0.01)
            loader = DataLoader(client_data[cid], batch_size=64, shuffle=True)
            trainer.train(loader, num_epochs=1)
            gradients = trainer.get_gradients()
            
            # Apply attack if Byzantine
            if cid in byzantine_ids:
                print(f"  Client {cid}: Applying model poisoning attack")
                gradients = [g * (-10.0) for g in gradients]
            
            # Generate PoT proof
            try:
                pot_proof = pipeline.phase2_client_generate_pot_proof(
                    client_id=cid,
                    weights=weights,
                    bias=bias,
                    train_data=client_train_data[cid],
                    round_number=round_num
                )
                
                # Submit
                pipeline.phase2_client_submit(
                    client_id=cid,
                    round_number=round_num,
                    gradients=gradients,
                    pot_proof=pot_proof,
                    data_commitment=commitments[cid]
                )
                
                print(f"  Client {cid}: Submitted (proof_size={pot_proof.proof_size} bytes)")
                
            except Exception as e:
                print(f"  Client {cid}: PoT proof generation failed: {e}")
        
        # Phase 2: Server verification
        print("Phase 2: Server verifying PoT proofs...")
        verified_clients, rejected_clients = pipeline.phase2_server_verify_all(
            weights=weights,
            bias=bias
        )
        
        print(f"  Verified: {len(verified_clients)} clients")
        print(f"  Rejected: {len(rejected_clients)} clients")
        if rejected_clients:
            print(f"  Rejected IDs: {sorted(rejected_clients.keys())}")
            print(f"  Reasons: {rejected_clients}")
        
        # Check if Byzantine clients were caught
        byzantine_caught = len([cid for cid in byzantine_ids if cid in rejected_clients])
        honest_rejected = len([cid for cid in verified_clients if cid in byzantine_ids])
        
        print(f"  Byzantine caught: {byzantine_caught}/{len(byzantine_ids)}")
        print(f"  Honest incorrectly rejected: {len(rejected_clients) - byzantine_caught}")
        
        # Phase 3: Simple averaging
        print("Phase 3: Aggregating verified gradients...")
        aggregated_gradients = pipeline.phase3_simple_average(verified_clients)
        
        if aggregated_gradients is None:
            print("  Warning: No verified clients - skipping round")
            continue
        
        # Phase 4: Update model
        print("Phase 4: Updating global model...")
        pipeline.phase4_update_global_model(aggregated_gradients)
        
        # Evaluate
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = global_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        
        print(f"  Test accuracy: {accuracy:.4f}")
        
        # Get round stats
        stats = pipeline.get_round_stats()
        print(f"  Detection TPR: {stats['byzantine_detection_rate_tpr']:.2%}")
        print(f"  Detection FPR: {stats['false_positive_rate_fpr']:.2%}")
    
    print(f"\n{'='*80}")
    print("Test Complete!")
    print(f"{'='*80}")
    print(f"Initial accuracy: {initial_accuracy:.4f}")
    print(f"Final accuracy:   {accuracy:.4f}")
    print(f"Improvement:      {(accuracy - initial_accuracy):.4f}")
    
    # Verify Byzantine detection
    if byzantine_caught == len(byzantine_ids):
        print("\n✓ SUCCESS: All Byzantine clients were detected!")
    else:
        print(f"\n✗ WARNING: Only {byzantine_caught}/{len(byzantine_ids)} Byzantine clients detected")
    
    if len(rejected_clients) - byzantine_caught == 0:
        print("✓ SUCCESS: No false positives!")
    else:
        print(f"✗ WARNING: {len(rejected_clients) - byzantine_caught} false positives")


if __name__ == "__main__":
    test_pot_pipeline()

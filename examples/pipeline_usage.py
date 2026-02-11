"""
ProtoGalaxy Pipeline - Quick Start Guide
=========================================

This guide shows how to use the complete end-to-end FL pipeline.
"""

from src.orchestration.pipeline import ProtoGalaxyPipeline, ClientSubmission
from src.client.trainer import Trainer  
from src.models.mnist import SimpleMLP
from src.data.loader import create_client_loaders
import torch

# ============================================================================
# Example 1: Basic FL Training with ProtoGalaxy
# ============================================================================

def basic_fl_training():
    """Run basic federated learning with full ProtoGalaxy pipeline"""
    
    # 1. Setup
    model = SimpleMLP()
    num_clients = 100
    num_galaxies = 10
    num_rounds = 20
    
    # 2. Initialize pipeline
    pipeline = ProtoGalaxyPipeline(
        global_model=model,
        num_clients=num_clients,
        num_galaxies=num_galaxies,
        defense_config={
            'use_full_analyzer': True,  # Enable 3-metric statistical analyzer
            'layer3_method': 'multi_krum',  # or 'trimmed_mean'
            'layer3_krum_f': 30,  # Tolerate 30% Byzantine clients
            'layer4_decay': 0.9  # Reputation decay factor
        }
    )
    
    # 3. Setup client data
    train_loaders, test_loader = create_client_loaders(
        dataset_name='mnist',
        num_clients=num_clients,
        batch_size=32,
        partition_type='iid'
    )
    
    # 4. Create client trainers
    trainers = {}
    for client_id in range(num_clients):
        trainers[client_id] = Trainer(
            model=model.create_copy(),  # Each client gets model copy
            train_loader=train_loaders[client_id],
            optimizer='sgd',
            learning_rate=0.01,
            device='cpu'
        )
    
    # 5. Run FL rounds
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        
        # Each client trains locally
        for client_id, trainer in trainers.items():
            trainer.train(epochs=1)
        
        # Execute complete ProtoGalaxy round
        round_stats = pipeline.execute_round(
            client_trainers=trainers,
            round_number=round_num
        )
        
        # Print statistics
        print(f"Global Merkle Root: {round_stats['global_root'][:16]}...")
        print(f"Verified clients: {round_stats['verified_clients']}/{round_stats['total_clients']}")
        print(f"Rejected clients: {round_stats['rejected_clients']}")
        print(f"Flagged galaxies: {round_stats['flagged_galaxies']}")
        print(f"Model hash: {round_stats['model_hash'][:16]}...")
        print(f"Round time: {round_stats['round_time']:.2f}s")
        
        # Distribute updated model to all clients
        for trainer in trainers.values():
            trainer.model.load_state_dict(model.state_dict())
        
        # Evaluate global model
        accuracy = evaluate_model(model, test_loader)
        print(f"Global accuracy: {accuracy:.2%}")
    
    return pipeline


# ============================================================================
# Example 2: FL with Byzantine Attacks
# ============================================================================

def fl_with_byzantine_attacks():
    """Run FL with simulated Byzantine attackers"""
    
    model = SimpleMLP()
    num_clients = 100
    num_byzantine = 30  # 30% Byzantine
    
    pipeline = ProtoGalaxyPipeline(
        global_model=model,
        num_clients=num_clients,
        num_galaxies=10,
        defense_config={
            'use_full_analyzer': True,
            'layer3_method': 'multi_krum',
            'layer3_krum_f': num_byzantine,
            'norm_threshold_sigma': 3.0,
            'cosine_threshold': 0.5
        }
    )
    
    # Setup clients (70 honest, 30 Byzantine)
    trainers = {}
    byzantine_clients = set(range(num_byzantine))
    
    for client_id in range(num_clients):
        trainer = Trainer(model.create_copy(), data_loader, device='cpu')
        
        # Byzantine clients perform gradient poisoning
        if client_id in byzantine_clients:
            trainer.set_attack_mode('gradient_poison', strength=2.0)
        
        trainers[client_id] = trainer
    
    # Run rounds - defense pipeline will detect and filter Byzantine clients
    for round_num in range(20):
        round_stats = pipeline.execute_round(trainers, round_num)
        
        # Check detection results
        total_flagged = sum(
            len(pipeline.galaxy_defense_coordinators[g].get_suspicious_clients())
            for g in range(10)
        )
        print(f"Round {round_num}: {total_flagged} clients flagged as suspicious")


# ============================================================================
# Example 3: Manual Phase-by-Phase Execution
# ============================================================================

def manual_phase_execution():
    """Execute each phase manually for fine-grained control"""
    
    pipeline = ProtoGalaxyPipeline(
        global_model=SimpleMLP(),
        num_clients=50,
        num_galaxies=5
    )
    
    # Get client gradients
    client_gradients = {}
    for client_id, trainer in trainers.items():
        trainer.train(epochs=1)
        client_gradients[client_id] = trainer.get_gradients()
    
    # === PHASE 1: COMMITMENT ===
    
    # Step 1: Clients generate commitments
    commitments_by_galaxy = {}
    metadata_by_client = {}
    
    for client_id, gradients in client_gradients.items():
        commit_hash, metadata = pipeline.phase1_client_commitment(
            client_id, gradients, round_number=0
        )
        metadata_by_client[client_id] = metadata
        
        galaxy_id = client_id % 5
        if galaxy_id not in commitments_by_galaxy:
            commitments_by_galaxy[galaxy_id] = {}
        commitments_by_galaxy[galaxy_id][client_id] = commit_hash
    
    # Step 2: Galaxies build Merkle trees
    galaxy_roots = {}
    for galaxy_id, commitments in commitments_by_galaxy.items():
        root = pipeline.phase1_galaxy_collect_commitments(
            galaxy_id, commitments, round_number=0
        )
        galaxy_roots[galaxy_id] = root
    
    # Step 3: Global builds tree
    global_root = pipeline.phase1_global_collect_galaxy_roots(
        galaxy_roots, round_number=0
    )
    print(f"Global Merkle root: {global_root}")
    
    # === PHASE 2: REVELATION ===
    
    # Clients submit gradients with proofs
    submissions_by_galaxy = {}
    for client_id, gradients in client_gradients.items():
        galaxy_id = client_id % 5
        commit_hash = commitments_by_galaxy[galaxy_id][client_id]
        metadata = metadata_by_client[client_id]
        
        submission = pipeline.phase2_client_submit_gradients(
            client_id, galaxy_id, gradients, commit_hash, metadata, round_number=0
        )
        
        if galaxy_id not in submissions_by_galaxy:
            submissions_by_galaxy[galaxy_id] = {}
        submissions_by_galaxy[galaxy_id][client_id] = submission
    
    # Galaxies verify submissions
    verified_by_galaxy = {}
    for galaxy_id, submissions in submissions_by_galaxy.items():
        verified, rejected = pipeline.phase2_galaxy_verify_and_collect(
            galaxy_id, submissions
        )
        verified_by_galaxy[galaxy_id] = verified
        print(f"Galaxy {galaxy_id}: {len(verified)} verified, {len(rejected)} rejected")
    
    # === PHASE 3: DEFENSE ===
    
    # Run defense pipeline for each galaxy
    galaxy_submissions = {}
    for galaxy_id, verified_updates in verified_by_galaxy.items():
        agg_grads, defense_report = pipeline.phase3_galaxy_defense_pipeline(
            galaxy_id, verified_updates
        )
        
        client_ids = [u['client_id'] for u in verified_updates]
        galaxy_sub = pipeline.phase3_galaxy_submit_to_global(
            galaxy_id, agg_grads, defense_report, client_ids
        )
        galaxy_submissions[galaxy_id] = galaxy_sub
        
        print(f"Galaxy {galaxy_id} defense: {len(defense_report['flagged_clients'])} flagged")
    
    # === PHASE 4: GLOBAL AGGREGATION ===
    
    # Verify galaxies
    verified_galaxies, rejected = pipeline.phase4_global_verify_galaxies(
        galaxy_submissions
    )
    print(f"Global: {len(verified_galaxies)} galaxies verified, {len(rejected)} rejected")
    
    # Global defense and aggregation
    global_grads, global_defense = pipeline.phase4_global_defense_and_aggregate(
        verified_galaxies
    )
    
    # Update model
    pipeline.phase4_update_global_model(global_grads, learning_rate=1.0)
    
    # Distribute model
    sync_package = pipeline.phase4_distribute_model()
    print(f"Model distributed with hash: {sync_package['model_hash'][:16]}...")


# ============================================================================
# Example 4: Using REST API for Production
# ============================================================================

def production_deployment():
    """Deploy ProtoGalaxy with REST APIs"""
    
    from src.communication.rest_api import (
        GalaxyAPIServer,
        GlobalAPIServer,
        GalaxyAPIClient
    )
    
    # Start Galaxy API servers
    galaxy_servers = []
    for galaxy_id in range(10):
        server = GalaxyAPIServer(
            galaxy_id=f"galaxy_{galaxy_id}",
            host='0.0.0.0',
            port=5000 + galaxy_id
        )
        server.start()
        galaxy_servers.append(server)
    
    # Start Global API server
    global_server = GlobalAPIServer(host='0.0.0.0', port=6000)
    global_server.set_expected_galaxies(10)
    global_server.start()
    
    # Clients connect via API
    client = GalaxyAPIClient(
        galaxy_host='localhost',
        galaxy_port=5000,
        timeout=30.0
    )
    
    # Submit gradient
    from src.communication.rest_api import GradientSubmission
    submission = GradientSubmission(
        client_id='client_1',
        galaxy_id='galaxy_0',
        round_number=0,
        gradients=[[0.1, 0.2, 0.3]],  # Serialized
        commitment_hash='abc123...'
    )
    response = client.submit_gradient(submission)
    
    # Get proof
    proof = client.get_proof('client_1', round_number=0)
    print(f"Received proof: {proof}")


# ============================================================================
# Utility Functions
# ============================================================================

def evaluate_model(model, test_loader):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Run basic FL training
    pipeline = basic_fl_training()
    
    # Get round statistics
    stats = pipeline.get_round_statistics()
    print(f"\nCompleted {len(stats)} rounds")
    print(f"Final model hash: {stats[-1]['model_hash']}")

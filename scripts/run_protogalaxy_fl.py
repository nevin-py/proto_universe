"""Complete ProtoGalaxy Federated Learning Pipeline.

This script implements the full 4-phase Protogalaxy protocol with:
- MNIST dataset with IID partitioning
- Linear regression model (or MLP/CNN)
- All 5 defense layers
- Merkle tree verification
- Forensic logging
- Configurable clients, galaxies, and rounds

Usage:
    python scripts/run_protogalaxy_fl.py --num-clients 100 --num-galaxies 10 --rounds 20
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mnist import create_mnist_model
from src.client.trainer import Trainer
from src.client.commitment import CommitmentGenerator
from src.data.partition import IIDPartitioner
from src.orchestration.protogalaxy_orchestrator import ProtoGalaxyOrchestrator
from src.orchestration.galaxy_manager import GalaxyManager

# Setup logging
def setup_logging(log_file: str = None, verbose: bool = False):
    """Setup comprehensive logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(name)-30s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers
    )
    
    # Reduce noise from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_mnist_data(data_dir: str = './data'):
    """Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load MNIST data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info(f"Loading MNIST dataset from {data_dir}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Test:  {len(test_dataset)} samples")
    
    return train_dataset, test_dataset


def partition_data(dataset, num_clients: int, partition_type: str = 'iid'):
    """Partition dataset across clients.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        partition_type: 'iid' or 'non-iid'
        
    Returns:
        Dict mapping client_id -> indices
    """
    logger.info(f"Partitioning data for {num_clients} clients ({partition_type})")
    
    if partition_type == 'iid':
        partitioner = IIDPartitioner(seed=42)
        return partitioner.partition(dataset, num_clients)
    else:
        raise NotImplementedError(f"Partition type '{partition_type}' not implemented yet")


def create_clients(
    model_template: nn.Module,
    data_partitions: Dict[int, List[int]],
    full_dataset,
    device: str = 'cpu'
) -> Dict[int, Dict]:
    """Create trainers for all clients.
    
    Args:
        model_template: Model architecture to use
        data_partitions: Dict mapping client_id -> data indices
        full_dataset: Full dataset to sample from
        device: Device to use
        
    Returns:
        Dict mapping client_id -> {'trainer': Trainer, 'data_loader': DataLoader}
    """
    logger.info(f"Creating {len(data_partitions)} client trainers")
    
    clients = {}
    for client_id, indices in data_partitions.items():
        # Create client's local dataset
        from torch.utils.data import Subset, DataLoader
        local_dataset = Subset(full_dataset, indices)
        
        # Create data loader
        data_loader = DataLoader(local_dataset, batch_size=32, shuffle=True)
        
        # Create trainer
        model = create_mnist_model(model_type='linear', num_classes=10).to(device)
        model.load_state_dict(model_template.state_dict())
        
        trainer = Trainer(model=model, device=device)
        
        clients[client_id] = {
            'trainer': trainer,
            'data_loader': data_loader
        }
    
    logger.info(f"  Created {len(clients)} trainers")
    return clients


def evaluate_model(model: nn.Module, test_loader, device: str = 'cpu') -> Dict:
    """Evaluate global model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        
    Returns:
        Dict with accuracy and loss
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_loader.dataset)
    }


def main():
    """Main FL pipeline execution."""
    parser = argparse.ArgumentParser(description='ProtoGalaxy Federated Learning Pipeline')
    
    # FL Configuration
    parser.add_argument('--num-clients', type=int, default=20,
                       help='Number of clients (default: 20)')
    parser.add_argument('--num-galaxies', type=int, default=2,
                       help='Number of galaxies (default: 2)')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of FL rounds (default: 10)')
    parser.add_argument('--local-epochs', type=int, default=1,
                       help='Local training epochs per round (default: 1)')
    
    # Model Configuration
    parser.add_argument('--model-type', type=str, default='linear',
                       choices=['linear', 'mlp', 'cnn'],
                       help='Model architecture (default: linear)')
    
    # Data Configuration
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--partition-type', type=str, default='iid',
                       choices=['iid', 'non-iid'],
                       help='Data partition type (default: iid)')
    
    # Defense Configuration
    parser.add_argument('--byzantine-ratio', type=float, default=0.0,
                       help='Ratio of Byzantine clients (default: 0.0)')
    parser.add_argument('--attack-type', type=str, default='none',
                       choices=['none', 'gradient_poison', 'label_flip'],
                       help='Attack type (default: none)')
    
    # System Configuration
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    parser.add_argument('--output-dir', type=str, default='./fl_results',
                       help='Output directory (default: ./fl_results)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (default: None)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file, verbose=args.verbose)
    
    # Print banner
    logger.info("=" * 100)
    logger.info("PROTOGALAXY FEDERATED LEARNING PIPELINE")
    logger.info("=" * 100)
    logger.info(f"Configuration:")
    logger.info(f"  Clients:      {args.num_clients}")
    logger.info(f"  Galaxies:     {args.num_galaxies}")
    logger.info(f"  Rounds:       {args.rounds}")
    logger.info(f"  Model:        {args.model_type}")
    logger.info(f"  Device:       {args.device}")
    logger.info(f"  Byzantine:    {args.byzantine_ratio:.1%} ({args.attack_type})")
    logger.info("=" * 100 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}\n")
    
    # Load data
    logger.info("STEP 1: DATA LOADING & PARTITIONING")
    logger.info("-" * 100)
    train_dataset, test_dataset = load_mnist_data(args.data_dir)
    data_partitions = partition_data(train_dataset, args.num_clients, args.partition_type)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )
    logger.info("")
    
    # Create global model
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("-" * 100)
    global_model = create_mnist_model(model_type=args.model_type, num_classes=10).to(args.device)
    logger.info(f"Created {args.model_type} model with {sum(p.numel() for p in global_model.parameters())} parameters")
    
    # Initial evaluation
    initial_metrics = evaluate_model(global_model, test_loader, args.device)
    logger.info(f"Initial accuracy: {initial_metrics['accuracy']:.2f}%")
    logger.info("")
    
    # Create clients
    logger.info("STEP 3: CLIENT INITIALIZATION")
    logger.info("-" * 100)
    clients = create_clients(global_model, data_partitions, train_dataset, args.device)
    logger.info("")
    
    # Create orchestrator
    logger.info("STEP 4: ORCHESTRATOR INITIALIZATION")
    logger.info("-" * 100)
    orchestrator = ProtoGalaxyOrchestrator(
        num_clients=args.num_clients,
        num_galaxies=args.num_galaxies,
        model=global_model,
        defense_config={
            'layer3_method': 'multi_krum',
            'layer3_krum_f': int(args.num_clients * args.byzantine_ratio),
            'layer5_norm_threshold': 3.0,
        },
        forensic_dir=str(output_dir / 'forensic_evidence')
    )
    
    # Assign clients to galaxies
    client_ids = list(clients.keys())
    orchestrator.galaxy_manager.assign_clients_round_robin(client_ids)
    orchestrator.galaxy_manager.log_status()
    logger.info("")
    
    # Training loop
    logger.info("STEP 5: FEDERATED TRAINING")
    logger.info("=" * 100 + "\n")
    
    round_metrics = []
    
    for round_num in range(args.rounds):
        round_start_time = time.time()
        
        # Client local training
        logger.info(f"Round {round_num}: Local training ({args.local_epochs} epochs)")
        client_updates = {}
        client_commitments = {}
        client_gradients = {}  # Raw gradients for ZK sum-check proofs
        
        for client_id, client_info in clients.items():
            trainer = client_info['trainer']
            data_loader = client_info['data_loader']
            
            # Train locally
            trainer.train(data_loader, num_epochs=args.local_epochs)
            
            # Get gradient (List[torch.Tensor])
            gradient = trainer.get_gradients()
            client_updates[client_id] = gradient
            client_gradients[client_id] = gradient  # Raw tensors for ZK proofs
            
            # Generate commitment - create GradientCommitment directly
            from src.client.commitment import GradientCommitment
            commitment = GradientCommitment(gradient, client_id=client_id, round_number=round_num)
            commitment.commit()
            client_commitments[client_id] = commitment
        
        logger.info(f"  ✓ {len(client_updates)} clients completed local training\n")
        
        # Run 4-phase protocol (with ZK sum-check proofs)
        round_result = orchestrator.run_round(
            client_updates=client_updates,
            client_commitments=client_commitments,
            client_gradients=client_gradients
        )
        
        # Evaluate global model
        test_metrics = evaluate_model(global_model, test_loader, args.device)
        round_result['test_metrics'] = test_metrics
        
        round_time = time.time() - round_start_time
        
        logger.info(f"\n{'='*100}")
        logger.info(f"ROUND {round_num} RESULTS")
        logger.info(f"{'='*100}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}% ({test_metrics['correct']}/{test_metrics['total']})")
        logger.info(f"Test Loss:     {test_metrics['loss']:.4f}")
        logger.info(f"Round Time:    {round_time:.2f}s")
        logger.info(f"{'='*100}\n\n")
        
        round_metrics.append({
            'round': round_num,
            'accuracy': test_metrics['accuracy'],
            'loss': test_metrics['loss'],
            'time': round_time
        })
        
        # Update client models with global model
        for client_id, client_info in clients.items():
            client_info['trainer'].model.load_state_dict(global_model.state_dict())
    
    # Final summary
    logger.info("\n" + "=" * 100)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 100)
    logger.info(f"Initial Accuracy: {initial_metrics['accuracy']:.2f}%")
    logger.info(f"Final Accuracy:   {round_metrics[-1]['accuracy']:.2f}%")
    logger.info(f"Improvement:      {round_metrics[-1]['accuracy'] - initial_metrics['accuracy']:.2f}%")
    logger.info("=" * 100)
    
    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'initial_metrics': initial_metrics,
            'round_metrics': round_metrics,
            'forensic_stats': orchestrator.defense_coordinator.forensic_logger.get_statistics()
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Forensic evidence: {output_dir / 'forensic_evidence'}\n")


if __name__ == '__main__':
    main()

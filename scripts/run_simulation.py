"""Main simulation runner for ProtoGalaxy.

Runs complete federated learning simulation with Byzantine clients
and defense mechanisms.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.manager import ConfigManager
from src.logging import FLLoggerFactory, LogLevel
from src.models.mnist import create_mnist_model
from src.data.datasets import load_mnist
from src.data.partition import IIDPartitioner, DirichletPartitioner, NonIIDPartitioner
from src.data.loader import create_client_loaders, create_test_loader
from src.simulation.runner import FLSimulation, SimulationConfig
from src.storage.manager import StorageManager


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Run ProtoGalaxy FL simulation')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of rounds (overrides config)')
    parser.add_argument('--clients', type=int, default=None,
                       help='Number of clients (overrides config)')
    parser.add_argument('--byzantine', type=int, default=None,
                       help='Number of Byzantine clients (overrides config)')
    parser.add_argument('--attack', type=str, default=None,
                       choices=['none', 'gradient_poison', 'label_flip', 'noise'],
                       help='Attack type')
    parser.add_argument('--partition', type=str, default='iid',
                       choices=['iid', 'noniid', 'dirichlet'],
                       help='Data partitioning strategy')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_all()
    
    # Setup logging
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    FLLoggerFactory.configure(
        log_dir=f"{args.output_dir}/logs",
        default_level=LogLevel.DEBUG if args.verbose else LogLevel.INFO
    )
    logger = FLLoggerFactory.get_global_logger()
    
    logger.info(f"Starting ProtoGalaxy simulation")
    logger.info(f"Configuration: {args.config}")
    
    # Build simulation config
    sim_config = SimulationConfig(
        num_clients=args.clients or config['fl'].get('num_clients', 10),
        num_galaxies=config['galaxy'].get('num_galaxies', 2),
        num_rounds=args.rounds or config['fl'].get('num_rounds', 10),
        local_epochs=config['fl'].get('local_epochs', 1),
        batch_size=config['fl'].get('local_batch_size', 32),
        learning_rate=config['model'].get('learning_rate', 0.01),
        num_byzantine=args.byzantine if args.byzantine is not None else config['simulation'].get('byzantine_clients', 0),
        attack_type=args.attack or config['simulation'].get('attack_type', 'gradient_poison'),
        enable_defense=config['defense'].get('enabled', True) if 'enabled' in config.get('defense', {}) else True,
        defense_method=config['defense'].get('method', 'trimmed_mean') if 'method' in config.get('defense', {}) else 'trimmed_mean',
        log_dir=f"{args.output_dir}/logs",
        verbose=args.verbose
    )
    
    logger.info(f"Simulation config: {sim_config.num_clients} clients, "
                f"{sim_config.num_byzantine} Byzantine, "
                f"{sim_config.num_rounds} rounds")
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist(data_dir="data/mnist")
    
    # Partition data
    logger.info(f"Partitioning data using {args.partition} strategy...")
    if args.partition == 'iid':
        partitioner = IIDPartitioner(seed=args.seed)
    elif args.partition == 'noniid':
        partitioner = NonIIDPartitioner(num_classes_per_client=2, seed=args.seed)
    else:  # dirichlet
        partitioner = DirichletPartitioner(alpha=0.5, seed=args.seed)
    
    partitions = partitioner.partition(train_dataset, sim_config.num_clients)
    
    # Log partition statistics
    from src.data.partition import get_partition_stats
    stats = get_partition_stats(train_dataset, partitions)
    logger.info(f"Data partition: avg {stats['avg_samples_per_client']:.0f} samples/client, "
                f"avg {stats['avg_classes_per_client']:.1f} classes/client")
    
    # Create data loaders
    train_loaders = create_client_loaders(
        train_dataset, partitions, sim_config.batch_size
    )
    test_loader = create_test_loader(test_dataset, sim_config.batch_size)
    
    # Create model
    logger.info("Creating model...")
    model = create_mnist_model()
    
    # Initialize storage
    storage = StorageManager(args.output_dir)
    
    # Run simulation
    logger.info("Starting FL simulation...")
    simulation = FLSimulation(
        model=model,
        config=sim_config,
        train_loaders=train_loaders,
        test_loader=test_loader,
        logger=logger
    )
    
    results = simulation.run()
    
    # Save results
    logger.info("Saving results...")
    storage.save_model(model, sim_config.num_rounds - 1, metadata={
        'final_accuracy': results['final_accuracy'],
        'final_loss': results['final_loss']
    })
    storage.save_metrics(results['metrics_summary'], 'summary')
    storage.save_metrics({
        'config': {
            'num_clients': sim_config.num_clients,
            'num_byzantine': sim_config.num_byzantine,
            'num_rounds': sim_config.num_rounds,
            'attack_type': sim_config.attack_type,
            'partition': args.partition
        },
        'results': {
            'final_accuracy': results['final_accuracy'],
            'final_loss': results['final_loss'],
            'total_duration': results['total_duration']
        }
    }, 'experiment')
    
    # Print summary
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print("="*50)
    print(f"Total Rounds:      {results['total_rounds']}")
    print(f"Total Duration:    {results['total_duration']:.2f}s")
    print(f"Final Accuracy:    {results['final_accuracy']*100:.2f}%")
    print(f"Final Loss:        {results['final_loss']:.4f}")
    print(f"Byzantine Clients: {sim_config.num_byzantine}")
    print(f"Defense Enabled:   {sim_config.enable_defense}")
    print("="*50)
    
    # Cleanup
    FLLoggerFactory.close_all()
    
    return results


if __name__ == '__main__':
    main()

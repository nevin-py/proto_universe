"""Main simulation runner for ProtoGalaxy"""

import argparse
import yaml
from src.config.manager import ConfigManager
from src.config.logger import LoggerSetup
from src.models.mnist import create_mnist_model
from src.orchestration.coordinator import FLCoordinator
from src.storage.manager import StorageManager
from src.simulation.metrics import MetricsCollector


def main():
    parser = argparse.ArgumentParser(description='Run ProtoGalaxy simulation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_all()
    
    # Setup logging
    logger_setup = LoggerSetup(f"{args.output_dir}/logs")
    logger = logger_setup.get_global_logger()
    logger.info(f"Starting simulation with config: {args.config}")
    
    # Initialize components
    model = create_mnist_model()
    fl_coordinator = FLCoordinator(
        num_clients=config['fl']['num_clients'],
        num_galaxies=config['galaxy']['num_galaxies'],
        model=model,
        config=config
    )
    
    storage_manager = StorageManager(args.output_dir)
    metrics_collector = MetricsCollector()
    
    # Initialize clients and galaxies
    fl_coordinator.initialize_clients(num_byzantine=config['simulation'].get('byzantine_clients', 0))
    fl_coordinator.initialize_galaxies()
    
    logger.info(f"Initialized {config['fl']['num_clients']} clients and {config['galaxy']['num_galaxies']} galaxies")
    
    # Run training rounds
    num_rounds = config['fl']['num_rounds']
    attack_type = config['simulation'].get('attack_type', 'gradient_poison')
    
    for round_num in range(num_rounds):
        logger.info(f"Starting round {round_num + 1}/{num_rounds}")
        
        # Execute round
        round_result = fl_coordinator.execute_round([], byzantine_attack=attack_type)
        
        # Record metrics
        if round_result['defense_results']:
            detections = round_result['defense_results'].get('layer1_detections', [])
            metrics_collector.record_detection(round_num, len(detections))
        
        logger.info(f"Round {round_num + 1} completed")
    
    # Save final model
    storage_manager.save_model(model, num_rounds - 1)
    
    # Save metrics
    summary = metrics_collector.get_summary()
    storage_manager.save_metrics(summary, 'summary')
    
    logger.info("Simulation completed successfully")
    logger.info(f"Summary: {summary}")


if __name__ == '__main__':
    main()

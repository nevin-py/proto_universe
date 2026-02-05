"""Batch experiment runner for multiple configurations"""

import argparse
import os
from src.config.manager import ConfigManager
from src.config.logger import LoggerSetup


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    parser.add_argument('--configs', nargs='+', required=True, help='Configuration files')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs per config')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()
    
    logger_setup = LoggerSetup(f"{args.output_dir}/logs")
    logger = logger_setup.get_global_logger()
    
    logger.info(f"Running batch experiments with {len(args.configs)} configurations")
    logger.info(f"Number of runs per configuration: {args.num_runs}")
    
    results = {}
    for config_file in args.configs:
        logger.info(f"Processing configuration: {config_file}")
        
        config_name = os.path.basename(config_file).replace('.yaml', '')
        results[config_name] = []
        
        for run in range(args.num_runs):
            logger.info(f"  Run {run + 1}/{args.num_runs}")
            # Run simulation with this config
            # Results would be aggregated here
        
        logger.info(f"Completed {config_name}")
    
    logger.info("Batch experiments completed")
    logger.info(f"Results summary: {results}")


if __name__ == '__main__':
    main()

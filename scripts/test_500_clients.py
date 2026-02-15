#!/usr/bin/env python3
"""
Test Script: 500 Clients MNIST IID

Quick test run with 500 clients on MNIST IID to validate scalability
and ProtoGalaxy pipeline performance at large scale.

Usage:
    python test_500_clients.py
    python test_500_clients.py --ablation merkle_only
    python test_500_clients.py --num-rounds 20 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evaluation import (
    run_single_experiment,
    ExperimentConfig,
    print_summary,
    save_resource_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("protogalaxy.test_500")


def main():
    parser = argparse.ArgumentParser(
        description="Test ProtoGalaxy with 500 clients on MNIST IID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
        help="FL rounds (default: 50)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local epochs per round (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="",
        choices=["", "merkle_only", "zk_merkle"],
        help="Ablation mode (default: full pipeline)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_500_results",
        help="Output directory (default: ./test_500_results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create experiment config
    cfg = ExperimentConfig(
        mode="test_500",
        trial_id=0,
        seed=args.seed,
        
        # Scale parameters
        num_clients=500,
        num_galaxies=10,  # 1 galaxy per 50 clients
        
        # Data
        dataset="mnist",
        model_type="linear",
        partition="iid",
        
        # Defense (full ProtoGalaxy pipeline)
        defense="protogalaxy_full",
        aggregation_method="trimmed_mean",
        trim_ratio=0.1,
        
        # Attack (clean run)
        attack="none",
        byzantine_fraction=0.0,
        
        # Training
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=64,
        learning_rate=0.01,
        
        # Ablation
        ablation=args.ablation,
    )
    
    print("\n" + "=" * 80)
    print("  ProtoGalaxy 500-Client Test")
    print("=" * 80)
    print(f"  Clients:      {cfg.num_clients}")
    print(f"  Galaxies:     {cfg.num_galaxies}")
    print(f"  Dataset:      {cfg.dataset} / {cfg.partition}")
    print(f"  Rounds:       {cfg.num_rounds}")
    print(f"  Ablation:     {cfg.ablation or 'full (merkle+zkp)'}")
    print(f"  Seed:         {cfg.seed}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 80)
    print()
    
    # Run experiment
    logger.info("Starting 500-client experiment...")
    
    try:
        result = run_single_experiment(cfg)
        
        # Save result
        from scripts.run_evaluation import _save_result
        _save_result(result, args.output_dir)
        
        # Print summary
        print_summary([result])
        save_resource_report([result], args.output_dir)
        
        # Additional metrics
        print("\n" + "=" * 80)
        print("  500-CLIENT TEST RESULTS")
        print("=" * 80)
        print(f"  Final Accuracy:        {result.final_accuracy:.4f} ({result.final_accuracy:.2%})")
        print(f"  Best Accuracy:         {result.best_accuracy:.4f}")
        print(f"  Total Time:            {result.total_time:.1f}s ({result.total_time/60:.1f} min)")
        print(f"  Avg Round Time:        {result.avg_round_time:.2f}s")
        print(f"  Avg ZK Prove Time:     {result.avg_zk_prove_time:.4f}s")
        print(f"  Avg ZK Verify Time:    {result.avg_zk_verify_time:.4f}s")
        print(f"  Avg Merkle Time:       {result.avg_merkle_time:.4f}s")
        print(f"  Total Communication:   {result.total_bytes:,} bytes ({result.total_bytes/(1024**2):.1f} MB)")
        
        ru = result.resource_usage
        if ru.get('num_samples', 0) > 0:
            print("\n  Resource Usage:")
            if 'cpu_percent' in ru:
                print(f"    CPU:   avg={ru['cpu_percent']['mean']:.1f}%, max={ru['cpu_percent']['max']:.1f}%")
            if 'ram_mb' in ru:
                print(f"    RAM:   avg={ru['ram_mb']['mean']:.0f} MB, peak={ru['ram_mb']['max']:.0f} MB")
            if 'gpu_util_percent' in ru:
                print(f"    GPU:   avg={ru['gpu_util_percent']['mean']:.1f}%, max={ru['gpu_util_percent']['max']:.1f}%")
            if 'gpu_mem_allocated_mb' in ru:
                print(f"    VRAM:  avg={ru['gpu_mem_allocated_mb']['mean']:.0f} MB, peak={ru['gpu_mem_allocated_mb']['max']:.0f} MB")
        
        print("=" * 80)
        print(f"\n✓ Test completed successfully!")
        print(f"  Results saved to: {args.output_dir}/")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scalability Benchmark Script

Evaluates ProtoGalaxy performance across varying client counts (20-500)
with Merkle-only vs Merkle+ZKP configurations.

Usage:
    python bench_scalability.py --output-dir ./scalability_results
    python bench_scalability.py --clients 20 50 100 --datasets mnist --trials 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_evaluation import (
    run_single_experiment,
    ExperimentConfig,
    ExperimentResult,
    print_summary,
    save_resource_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("protogalaxy.bench_scalability")


def generate_scalability_configs(
    client_counts: List[int],
    datasets: List[str],
    partitions: List[str],
    ablations: List[str],
    trials: int,
    base_seed: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    use_amp: bool,
) -> List[ExperimentConfig]:
    """Generate experiment configurations for scalability benchmarking.
    
    Parameters
    ----------
    client_counts : List[int]
        Number of clients to test (e.g., [20, 50, 100, 200, 500])
    datasets : List[str]
        Datasets to use ('mnist', 'cifar10')
    partitions : List[str]
        Data partition methods ('iid', 'noniid')
    ablations : List[str]
        Ablation modes ('merkle_only', 'full')
    trials : int
        Number of independent trials per configuration
    base_seed : int
        Base random seed (trial_i uses base_seed + i)
    num_rounds : int
        FL rounds per experiment
    local_epochs : int
        Local training epochs per round
    
    Returns
    -------
    List[ExperimentConfig]
        Complete experiment matrix
    """
    configs = []
    
    for num_clients in client_counts:
        # Calculate galaxies (roughly 1 galaxy per 10 clients, min 2, max 10)
        num_galaxies = min(10, max(2, num_clients // 10))
        
        for dataset in datasets:
            # Model selection based on dataset
            if dataset == "mnist":
                model_type = "linear"
            elif dataset == "cifar10":
                model_type = "cifar10_cnn"
            else:
                model_type = "mlp"
            
            for partition in partitions:
                for ablation in ablations:
                    for trial in range(trials):
                        cfg = ExperimentConfig(
                            mode="scalability",
                            trial_id=trial,
                            seed=base_seed + trial,
                            
                            # Scale parameters
                            num_clients=num_clients,
                            num_galaxies=num_galaxies,
                            
                            # Data
                            dataset=dataset,
                            model_type=model_type,
                            partition=partition,
                            
                            # Defense (full ProtoGalaxy pipeline)
                            defense="protogalaxy_full",
                            aggregation_method="trimmed_mean",
                            trim_ratio=0.1,
                            
                            # Attack (clean run for scalability)
                            attack="none",
                            byzantine_fraction=0.0,
                            
                            # Training
                            num_rounds=num_rounds,
                            local_epochs=local_epochs,
                            batch_size=batch_size,
                            learning_rate=0.01,
                            
                            # Optimization
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            use_amp=use_amp,
                            
                            # Ablation
                            ablation=ablation if ablation != "full" else "",
                        )
                        configs.append(cfg)
    
    return configs


def save_scalability_summary(
    results: List[ExperimentResult],
    output_dir: str,
):
    """Save scalability-specific summary with timing breakdown.
    
    Parameters
    ----------
    results : List[ExperimentResult]
        All experiment results
    output_dir : str
        Output directory
    """
    summary_path = Path(output_dir) / "scalability_summary.json"
    
    # Group results by configuration
    summary = {}
    
    for r in results:
        key = (
            r.config.num_clients,
            r.config.dataset,
            r.config.partition,
            r.config.ablation or "full",
        )
        
        if key not in summary:
            summary[key] = {
                "num_clients": r.config.num_clients,
                "num_galaxies": r.config.num_galaxies,
                "dataset": r.config.dataset,
                "partition": r.config.partition,
                "ablation": r.config.ablation or "full",
                "trials": [],
            }
        
        summary[key]["trials"].append({
            "trial_id": r.config.trial_id,
            "final_accuracy": r.final_accuracy,
            "total_time": r.total_time,
            "avg_round_time": r.avg_round_time,
            "avg_zk_prove_time": r.avg_zk_prove_time,
            "avg_zk_verify_time": r.avg_zk_verify_time,
            "avg_merkle_time": r.avg_merkle_time,
            "total_bytes": r.total_bytes,
            "resource_usage": r.resource_usage,
        })
    
    # Compute aggregates
    for key, data in summary.items():
        trials = data["trials"]
        n = len(trials)
        
        data["avg_accuracy"] = sum(t["final_accuracy"] for t in trials) / n
        data["avg_total_time"] = sum(t["total_time"] for t in trials) / n
        data["avg_round_time"] = sum(t["avg_round_time"] for t in trials) / n
        data["avg_zk_prove"] = sum(t["avg_zk_prove_time"] for t in trials) / n
        data["avg_zk_verify"] = sum(t["avg_zk_verify_time"] for t in trials) / n
        data["avg_merkle"] = sum(t["avg_merkle_time"] for t in trials) / n
        data["avg_total_bytes"] = sum(t["total_bytes"] for t in trials) / n
    
    # Convert to list and sort by client count
    summary_list = sorted(summary.values(), key=lambda x: (x["num_clients"], x["dataset"], x["partition"], x["ablation"]))
    
    with open(summary_path, "w") as f:
        json.dump(summary_list, f, indent=2)
    
    logger.info(f"Scalability summary saved to {summary_path}")
    
    # Print table
    print("\n" + "=" * 120)
    print("  SCALABILITY BENCHMARK SUMMARY")
    print("=" * 120)
    header = (
        f"{'Clients':>8} {'Galaxies':>8} {'Dataset':>10} {'Partition':>10} "
        f"{'Ablation':>12} {'Acc':>6} {'Round(s)':>9} {'ZKP(s)':>8} {'Merkle(s)':>10} {'Total(s)':>9}"
    )
    print(header)
    print("-" * 120)
    
    for data in summary_list:
        print(
            f"{data['num_clients']:>8} {data['num_galaxies']:>8} "
            f"{data['dataset']:>10} {data['partition']:>10} "
            f"{data['ablation']:>12} "
            f"{data['avg_accuracy']:>5.3f} "
            f"{data['avg_round_time']:>8.2f} "
            f"{data['avg_zk_prove'] + data['avg_zk_verify']:>7.2f} "
            f"{data['avg_merkle']:>9.4f} "
            f"{data['avg_total_time']:>8.1f}"
        )
    
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="ProtoGalaxy Scalability Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Scalability parameters
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200, 500],
        help="Client counts to benchmark (default: 20 50 100 200 500)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mnist", "cifar10"],
        choices=["mnist", "cifar10"],
        help="Datasets to use (default: mnist cifar10)",
    )
    parser.add_argument(
        "--partitions",
        type=str,
        nargs="+",
        default=["iid", "noniid"],
        choices=["iid", "noniid"],
        help="Partition methods (default: iid noniid)",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=["merkle_only", "full"],
        choices=["merkle_only", "full"],
        help="Ablation modes: merkle_only (no ZKP), full (merkle+ZKP) (default: both)",
    )
    
    # Experiment parameters
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per configuration (default: 3)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
        help="FL rounds per experiment (default: 50)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local training epochs per round (default: 1)",
    )
    
    # Optimization parameters (optimized for Colab T4)
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers (default: 2)")
    parser.add_argument("--pin-memory", action="store_true", default=True, help="Pin memory (default: True)")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory", help="Disable pin memory")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor (default: 2)")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use AMP (default: True)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp", help="Disable AMP")
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scalability_results",
        help="Output directory for results (default: ./scalability_results)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose result file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment matrix without running",
    )
    
    args = parser.parse_args()
    
    # Generate experiment matrix
    configs = generate_scalability_configs(
        client_counts=args.clients,
        datasets=args.datasets,
        partitions=args.partitions,
        ablations=args.ablations,
        trials=args.trials,
        base_seed=args.base_seed,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        use_amp=args.use_amp,
    )
    
    print("\n" + "=" * 80)
    print("  ProtoGalaxy Scalability Benchmark")
    print("=" * 80)
    print(f"  Client counts:    {args.clients}")
    print(f"  Datasets:         {args.datasets}")
    print(f"  Partitions:       {args.partitions}")
    print(f"  Ablations:        {args.ablations}")
    print(f"  Trials/config:    {args.trials}")
    print(f"  Rounds:           {args.num_rounds}")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Output:           {args.output_dir}")
    print("=" * 80)
    
    if args.dry_run:
        print("\n  DRY RUN — experiment matrix:\n")
        for i, c in enumerate(configs):
            print(
                f"  {i+1:>4}. clients={c.num_clients:<4} galaxies={c.num_galaxies:<3} "
                f"{c.dataset:<10} {c.partition:<8} {c.ablation or 'full':<12} "
                f"trial={c.trial_id}"
            )
        print(f"\n  Total: {len(configs)} experiments")
        return
    
    # Run experiments
    from scripts.run_evaluation import _is_completed, _save_result
    
    all_results: List[ExperimentResult] = []
    failed: List[tuple] = []
    
    start_time = time.time()
    
    for i, cfg in enumerate(configs):
        print(f"\n{'─' * 80}")
        print(f"  Experiment {i+1}/{len(configs)}: {cfg.experiment_id}")
        print(f"{'─' * 80}")
        
        if args.resume and _is_completed(cfg, args.output_dir):
            logger.info(f"  ✓ Already completed — skipping")
            continue
        
        try:
            result = run_single_experiment(cfg)
            _save_result(result, args.output_dir)
            all_results.append(result)
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            failed.append((cfg.experiment_id, str(e)))
    
    elapsed = time.time() - start_time
    
    # Summaries
    print_summary(all_results)
    
    if all_results:
        save_scalability_summary(all_results, args.output_dir)
        save_resource_report(all_results, args.output_dir)
    
    if failed:
        print(f"\n  ⚠ {len(failed)} experiments failed:")
        for eid, reason in failed:
            print(f"    - {eid}: {reason}")
    
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

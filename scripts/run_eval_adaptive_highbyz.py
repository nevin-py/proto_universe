#!/usr/bin/env python3
"""
Adaptive Attack — High Byzantine Fraction Evaluation
=====================================================

Tests whether ZKP outperforms Multi-Krum when Byzantine fraction exceeds
Krum's theoretical safety threshold (f < n/2 - 1).

Configuration:
  • 10 clients, 2 galaxies (5 clients/galaxy)
  • 10 rounds (5 stealth + 5 poison)
  • Byzantine fractions: 40% (4 clients), 50% (5 clients)
  • Partition: IID only
  • Defenses: vanilla, multi_krum, protogalaxy_full (merkle_only, zk_merkle)
  • 1 trial per config (seed 42)

At 40% Byzantine (4/10):
  Krum f=4, per-galaxy: 5 clients, 4 Byzantine → krum_m = max(1, 5-4-2) = 1
  Krum selects only 1 client per galaxy — thin margin.

At 50% Byzantine (5/10):
  Krum f=5, per-galaxy: 5 clients, 5 Byzantine → krum_m = max(1, 5-5-2) = 1
  Krum cannot guarantee honest selection — may select malicious clients.

Experiment Matrix:
  2 byz_fractions × 4 defenses × 1 trial = 8 experiments
  Estimated: ~15 min

Usage:
    python scripts/run_eval_adaptive_highbyz.py
    python scripts/run_eval_adaptive_highbyz.py --dry-run
    python scripts/run_eval_adaptive_highbyz.py --byz 0.4    # only 40%
    python scripts/run_eval_adaptive_highbyz.py --byz 0.5    # only 50%
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ExperimentConfig:
    defense: str
    attack: str
    partition: str
    ablation: str
    trial_id: int
    seed: int
    num_rounds: int
    num_clients: int
    num_galaxies: int
    byzantine_fraction: float

    def to_cmd(self, output_dir: str) -> List[str]:
        cmd = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--mode", "custom",
            "--defense", self.defense,
            "--attack", self.attack,
            "--num-clients", str(self.num_clients),
            "--num-galaxies", str(self.num_galaxies),
            "--num-rounds", str(self.num_rounds),
            "--byzantine-fraction", str(self.byzantine_fraction),
            "--partition", self.partition,
            "--trials", "1",
            "--base-seed", str(self.seed),
            "--output-dir", output_dir,
            "--verbose",
        ]
        if self.ablation:
            cmd.extend(["--ablation", self.ablation])
        return cmd

    def experiment_id(self) -> str:
        byz_pct = int(self.byzantine_fraction * 100)
        parts = [self.defense, self.attack, self.partition,
                 f"c{self.num_clients}_g{self.num_galaxies}",
                 f"byz{byz_pct}", f"t{self.trial_id}"]
        if self.ablation:
            parts.append(self.ablation)
        return "__".join(parts)

    def estimated_time_seconds(self) -> float:
        per_round = {
            ("vanilla", ""): 4.0,
            ("multi_krum", ""): 4.0,
            ("protogalaxy_full", "merkle_only"): 5.0,
            ("protogalaxy_full", "zk_merkle"): 20.0,
        }
        return per_round.get((self.defense, self.ablation), 5.0) * self.num_rounds


def generate_experiments(
    byz_fractions: List[float],
    num_rounds: int = 10,
    seed: int = 42,
) -> List[ExperimentConfig]:
    experiments = []
    for byz_frac in byz_fractions:
        for defense, ablation in [
            ("vanilla", ""),
            ("multi_krum", ""),
            ("protogalaxy_full", "merkle_only"),
            ("protogalaxy_full", "zk_merkle"),
        ]:
            experiments.append(ExperimentConfig(
                defense=defense,
                attack="adaptive",
                partition="iid",
                ablation=ablation,
                trial_id=0,
                seed=seed,
                num_rounds=num_rounds,
                num_clients=10,
                num_galaxies=2,
                byzantine_fraction=byz_frac,
            ))
    return experiments


def run_experiment(exp: ExperimentConfig, output_dir: str,
                   dry_run: bool = False) -> Optional[Dict[str, Any]]:
    cmd = exp.to_cmd(output_dir)

    if dry_run:
        print(f"  [DRY] {exp.experiment_id()}  (~{exp.estimated_time_seconds():.0f}s)")
        return None

    byz_pct = int(exp.byzantine_fraction * 100)
    print(f"\n{'='*80}")
    print(f"Running: {exp.experiment_id()}")
    print(f"  Defense={exp.defense}  Ablation={exp.ablation or '—'}")
    print(f"  Byzantine={byz_pct}%  Rounds={exp.num_rounds}  Seed={exp.seed}")
    print(f"  Est: {exp.estimated_time_seconds():.0f}s")
    print(f"{'='*80}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    return {
        "experiment_id": exp.experiment_id(),
        "exit_code": result.returncode,
        "elapsed_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive attack — high Byzantine evaluation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--byz", type=float, nargs="*", default=None,
                        help="Byzantine fractions to test (default: 0.4 0.5)")
    parser.add_argument("--output-dir", default="eval_adaptive_highbyz")
    args = parser.parse_args()

    byz_fractions = args.byz if args.byz else [0.4, 0.5]
    output_dir = args.output_dir

    experiments = generate_experiments(
        byz_fractions=byz_fractions,
        num_rounds=args.rounds,
    )

    total_est = sum(e.estimated_time_seconds() for e in experiments)
    print(f"\n{'#'*80}")
    print(f"#  Adaptive Attack — High Byzantine Evaluation")
    print(f"#  Byzantine fractions: {[f'{b:.0%}' for b in byz_fractions]}")
    print(f"#  {len(experiments)} experiments, est. {total_est/60:.1f} min")
    print(f"#  Output: {output_dir}/")
    print(f"{'#'*80}\n")

    if args.dry_run:
        print("DRY RUN — listing experiments:\n")
        for exp in experiments:
            print(f"  [DRY] {exp.experiment_id()}  (~{exp.estimated_time_seconds():.0f}s)")
        print(f"\nTotal estimated: {total_est/60:.1f} min")
        return

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n>>> Experiment {i}/{len(experiments)}")
        r = run_experiment(exp, output_dir)
        if r:
            results.append(r)
            status = "OK" if r["exit_code"] == 0 else f"FAIL (exit {r['exit_code']})"
            print(f"  → {status} in {r['elapsed_time']:.1f}s")

    # Summary
    print(f"\n\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    ok = sum(1 for r in results if r["exit_code"] == 0)
    fail = len(results) - ok
    total_time = sum(r["elapsed_time"] for r in results)
    print(f"  Completed: {ok}/{len(results)} OK, {fail} failed")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    for r in results:
        status = "✓" if r["exit_code"] == 0 else "✗"
        print(f"  {status} {r['experiment_id']}  ({r['elapsed_time']:.0f}s)")

    print(f"\nResults saved to: {output_dir}/custom/")


if __name__ == "__main__":
    main()

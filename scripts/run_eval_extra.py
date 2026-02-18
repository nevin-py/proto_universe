#!/usr/bin/env python3
"""
Small-Scale ProtoGalaxy Evaluation — Gaussian Noise & Targeted Label Flip (IID only)
=====================================================================================

Runs the same evaluation matrix as run_small_eval.py but for:
  • gaussian_noise attack
  • targeted_label_flip attack
  • IID partition only

Configuration:
  • 10 clients, 2 galaxies (5 clients/galaxy)
  • 10 rounds
  • 1 trial (seed 42)
  • Byzantine fraction: 30% (3 malicious clients, randomized)
  • Partition: IID only
  • Defenses: vanilla, multi_krum, protogalaxy_full (merkle_only, zk_merkle)

Estimated Time:
  2 attacks × 4 defense configs × 1 trial = 8 experiments
  = 2 vanilla (~40s) + 2 multi_krum (~40s) + 2 merkle_only (~50s) + 2 zk_merkle (~200s)
  ≈ 12 min total

Usage:
    python scripts/run_eval_extra.py
    python scripts/run_eval_extra.py --dry-run
    python scripts/run_eval_extra.py --attack gaussian_noise
    python scripts/run_eval_extra.py --no-zkp
"""

import argparse
import json
import os
import statistics
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

    def to_cmd(self, base_config: Dict[str, Any]) -> List[str]:
        cmd = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--mode", "custom",
            "--defense", self.defense,
            "--attack", self.attack,
            "--num-clients", str(base_config["num_clients"]),
            "--num-galaxies", str(base_config["num_galaxies"]),
            "--num-rounds", str(base_config["num_rounds"]),
            "--byzantine-fraction", str(base_config["byzantine_fraction"]),
            "--partition", self.partition,
            "--trials", "1",
            "--base-seed", str(self.seed),
            "--output-dir", base_config["output_dir"],
            "--verbose",
        ]
        if self.ablation:
            cmd.extend(["--ablation", self.ablation])
        return cmd

    def experiment_id(self) -> str:
        parts = [self.defense, self.attack, self.partition,
                 f"c{10}_g{2}", f"byz30", f"t{self.trial_id}"]
        if self.ablation:
            parts.append(self.ablation)
        return "__".join(parts)

    def estimated_time_seconds(self) -> float:
        if self.defense == "protogalaxy_full":
            if self.ablation == "zk_merkle":
                return 200
            return 50
        return 40


def generate_experiment_matrix(
    num_trials: int = 1,
    base_seed: int = 42,
    attacks: Optional[List[str]] = None,
    include_zkp: bool = True,
) -> List[ExperimentConfig]:
    if attacks is None:
        attacks = ["gaussian_noise", "targeted_label_flip"]

    experiments = []
    for attack in attacks:
        for trial_id in range(num_trials):
            seed = base_seed + trial_id

            experiments.append(ExperimentConfig(
                defense="vanilla", attack=attack, partition="iid",
                ablation="", trial_id=trial_id, seed=seed,
            ))
            experiments.append(ExperimentConfig(
                defense="multi_krum", attack=attack, partition="iid",
                ablation="", trial_id=trial_id, seed=seed,
            ))
            experiments.append(ExperimentConfig(
                defense="protogalaxy_full", attack=attack, partition="iid",
                ablation="merkle_only", trial_id=trial_id, seed=seed,
            ))
            if include_zkp:
                experiments.append(ExperimentConfig(
                    defense="protogalaxy_full", attack=attack, partition="iid",
                    ablation="zk_merkle", trial_id=trial_id, seed=seed,
                ))
    return experiments


def run_experiment(exp: ExperimentConfig, base_config: Dict[str, Any],
                   dry_run: bool = False) -> Optional[Dict[str, Any]]:
    cmd = exp.to_cmd(base_config)

    if dry_run:
        print(f"  [DRY] {exp.experiment_id()}  (~{exp.estimated_time_seconds():.0f}s)")
        return None

    print(f"\n{'='*80}")
    print(f"Running: {exp.experiment_id()}")
    print(f"  Defense={exp.defense}  Attack={exp.attack}  Ablation={exp.ablation or '—'}")
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


def create_summary_report(output_dir: Path, experiments: List[ExperimentConfig],
                          base_config: Dict[str, Any]):
    """Load result JSONs and write a markdown summary."""
    custom_dir = output_dir / "custom"
    if not custom_dir.exists():
        print("No custom/ results directory found — skipping report.")
        return

    # Load all JSONs
    data_map: Dict[str, Any] = {}
    for jf in sorted(custom_dir.glob("*.json")):
        with open(jf) as f:
            d = json.load(f)
        data_map[jf.stem] = d

    # Filter to our attacks only
    our_attacks = {"gaussian_noise", "targeted_label_flip"}
    filtered = {k: v for k, v in data_map.items()
                if v["config"]["attack"] in our_attacks and v["config"]["partition"] == "iid"}

    if not filtered:
        print("No matching results found for gaussian_noise / targeted_label_flip IID.")
        return

    report_path = output_dir / "EXTRA_EVAL_RESULTS.md"
    with open(report_path, "w") as f:
        f.write("# Evaluation Results — Gaussian Noise & Targeted Label Flip (IID)\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Clients: {base_config['num_clients']}, Galaxies: {base_config['num_galaxies']}\n")
        f.write(f"- Rounds: {base_config['num_rounds']}, Byzantine: {base_config['byzantine_fraction']:.0%}\n")
        f.write(f"- Partition: IID only\n")
        f.write(f"- Attacks: gaussian_noise, targeted_label_flip\n\n")

        # Group by (defense, attack, ablation)
        from collections import defaultdict
        grouped = defaultdict(list)
        for name, d in filtered.items():
            cfg = d["config"]
            key = (cfg["defense"], cfg["attack"], cfg.get("ablation", ""))
            grouped[key].append(d)

        # --- Accuracy Table ---
        f.write("## Final Accuracy\n\n")
        f.write("| Defense | Ablation | gaussian_noise | targeted_label_flip |\n")
        f.write("|---------|----------|:-:|:-:|\n")

        defense_order = [
            ("vanilla", ""),
            ("multi_krum", ""),
            ("protogalaxy_full", "merkle_only"),
            ("protogalaxy_full", "zk_merkle"),
        ]
        for defense, ablation in defense_order:
            gn = grouped.get((defense, "gaussian_noise", ablation), [])
            tlf = grouped.get((defense, "targeted_label_flip", ablation), [])
            gn_acc = f"{statistics.mean([d['final_accuracy'] for d in gn]):.4f}" if gn else "—"
            tlf_acc = f"{statistics.mean([d['final_accuracy'] for d in tlf]):.4f}" if tlf else "—"
            abl_str = ablation if ablation else "—"
            f.write(f"| {defense} | {abl_str} | {gn_acc} | {tlf_acc} |\n")

        # --- Detection Table ---
        f.write("\n## Detection Performance\n\n")
        f.write("| Defense | Ablation | Attack | TPR | FPR | F1 |\n")
        f.write("|---------|----------|--------|:---:|:---:|:--:|\n")

        for (defense, attack, ablation), dlist in sorted(grouped.items()):
            tprs = [d["avg_tpr"] for d in dlist]
            fprs = [d["avg_fpr"] for d in dlist]
            f1s = [d["avg_f1"] for d in dlist]
            abl_str = ablation if ablation else "—"
            f.write(f"| {defense} | {abl_str} | {attack} | "
                    f"{statistics.mean(tprs):.3f} | {statistics.mean(fprs):.3f} | "
                    f"{statistics.mean(f1s):.3f} |\n")

        # --- ZKP Metrics ---
        f.write("\n## ZKP Proof Metrics\n\n")
        f.write("| Attack | ZKP Prove (s) | ZKP Verify (s) | Failed/Rnd | Gen/Rnd |\n")
        f.write("|--------|:---:|:---:|:---:|:---:|\n")

        for (defense, attack, ablation), dlist in sorted(grouped.items()):
            if ablation != "zk_merkle":
                continue
            for d in dlist:
                rounds = d.get("rounds", [])
                avg_fail = statistics.mean([r.get("zk_proofs_failed", 0) for r in rounds]) if rounds else 0
                avg_gen = statistics.mean([r.get("zk_proofs_generated", 0) for r in rounds]) if rounds else 0
                f.write(f"| {attack} | {d['avg_zk_prove_time']:.2f} | "
                        f"{d['avg_zk_verify_time']:.2f} | {avg_fail:.1f} | {avg_gen:.1f} |\n")

        # --- Timing ---
        f.write("\n## Timing\n\n")
        f.write("| Defense | Ablation | Attack | Avg/Rnd (s) | Total (s) |\n")
        f.write("|---------|----------|--------|:-:|:-:|\n")

        for (defense, attack, ablation), dlist in sorted(grouped.items()):
            abl_str = ablation if ablation else "—"
            for d in dlist:
                f.write(f"| {defense} | {abl_str} | {attack} | "
                        f"{d['avg_round_time']:.2f} | {d['total_time']:.1f} |\n")

        # --- Trajectories ---
        f.write("\n## Accuracy Trajectories\n\n")
        f.write("```\n")
        for (defense, attack, ablation), dlist in sorted(grouped.items()):
            abl = f"+{ablation}" if ablation else ""
            for d in dlist:
                rounds = d.get("rounds", [])
                traj = " → ".join(f"{r['accuracy']:.3f}" for r in rounds)
                f.write(f"{defense}{abl} | {attack}: {traj}\n")
        f.write("```\n")

        f.write(f"\n---\n*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\n✅ Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Eval: gaussian_noise & targeted_label_flip (IID only)")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--attack", type=str,
                        choices=["gaussian_noise", "targeted_label_flip"])
    parser.add_argument("--no-zkp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./eval_extra")
    args = parser.parse_args()

    base_config = {
        "num_clients": 10,
        "num_galaxies": 2,
        "num_rounds": 10,
        "byzantine_fraction": 0.3,
        "output_dir": args.output_dir,
    }

    attacks = [args.attack] if args.attack else None
    experiments = generate_experiment_matrix(
        num_trials=args.trials,
        attacks=attacks,
        include_zkp=not args.no_zkp,
    )

    total_est = sum(e.estimated_time_seconds() for e in experiments)
    print(f"\n{'='*80}")
    print(f"EXTRA EVALUATION: gaussian_noise + targeted_label_flip (IID)")
    print(f"{'='*80}")
    print(f"Experiments: {len(experiments)}")
    print(f"Config: {base_config['num_clients']} clients, "
          f"{base_config['num_galaxies']} galaxies, "
          f"{base_config['num_rounds']} rounds, "
          f"{base_config['byzantine_fraction']:.0%} Byzantine")
    print(f"Est. time: {total_est/60:.0f} min")
    print(f"Output: {base_config['output_dir']}")
    print(f"{'='*80}\n")

    if args.dry_run:
        for exp in experiments:
            run_experiment(exp, base_config, dry_run=True)
        print(f"\nTotal: {len(experiments)} experiments")
        return

    output_path = Path(base_config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        result = run_experiment(exp, base_config)
        if result:
            results.append(result)
            elapsed = time.time() - start_time
            remaining = len(experiments) - i
            eta = (elapsed / i) * remaining if i > 0 else 0
            print(f"\n⏱️  Progress: {i}/{len(experiments)} ({i/len(experiments):.0%})")
            print(f"   Elapsed: {elapsed/60:.1f} min, ETA: {eta/60:.1f} min")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"Results: {output_path}")

    create_summary_report(output_path, experiments, base_config)
    print(f"\n✅ Done! Check {output_path}/EXTRA_EVAL_RESULTS.md")


if __name__ == "__main__":
    main()

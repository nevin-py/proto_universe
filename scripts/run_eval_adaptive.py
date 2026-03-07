#!/usr/bin/env python3
"""
Small-Scale FiZK Evaluation — Adaptive Attack
=====================================================

Runs evaluation for the adaptive adversary that observes honest gradient
statistics and crafts evasive poisons staying within norm/cosine bounds.

The adaptive attacker has a 5-round observation window (stealth mode),
then poisons rounds 5-9 (untargeted: reverses honest centroid direction).

With 10 rounds -> 5 poison rounds.  With 20 rounds -> 15 poison rounds.

Configuration:
  • 10 clients, 2 galaxies (5 clients/galaxy)
  • 10 rounds (default) or 20 (--rounds 20)
  • 2 trials (seeds 42, 43)
  • Byzantine fraction: 30% (3 malicious clients, randomized)
  • Partition: IID only
  • Defenses: vanilla, multi_krum, protogalaxy_full (merkle_only, zk_merkle)

Usage:
    python scripts/run_eval_adaptive.py
    python scripts/run_eval_adaptive.py --rounds 20
    python scripts/run_eval_adaptive.py --dry-run
    python scripts/run_eval_adaptive.py --no-zkp
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
    num_rounds: int

    def to_cmd(self, base_config: Dict[str, Any]) -> List[str]:
        cmd = [
            sys.executable,
            "scripts/run_evaluation.py",
            "--mode", "custom",
            "--defense", self.defense,
            "--attack", self.attack,
            "--num-clients", str(base_config["num_clients"]),
            "--num-galaxies", str(base_config["num_galaxies"]),
            "--num-rounds", str(self.num_rounds),
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
                 f"c{10}_g{2}", f"r{self.num_rounds}", f"byz30",
                 f"t{self.trial_id}"]
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


def generate_experiment_matrix(
    num_trials: int = 2,
    base_seed: int = 42,
    include_zkp: bool = True,
    num_rounds: int = 10,
) -> List[ExperimentConfig]:
    experiments = []
    for trial_id in range(num_trials):
        seed = base_seed + trial_id

        experiments.append(ExperimentConfig(
            defense="vanilla", attack="adaptive", partition="iid",
            ablation="", trial_id=trial_id, seed=seed,
            num_rounds=num_rounds,
        ))
        experiments.append(ExperimentConfig(
            defense="multi_krum", attack="adaptive", partition="iid",
            ablation="", trial_id=trial_id, seed=seed,
            num_rounds=num_rounds,
        ))
        experiments.append(ExperimentConfig(
            defense="protogalaxy_full", attack="adaptive", partition="iid",
            ablation="merkle_only", trial_id=trial_id, seed=seed,
            num_rounds=num_rounds,
        ))
        if include_zkp:
            experiments.append(ExperimentConfig(
                defense="protogalaxy_full", attack="adaptive", partition="iid",
                ablation="zk_merkle", trial_id=trial_id, seed=seed,
                num_rounds=num_rounds,
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
    print(f"  Defense={exp.defense}  Ablation={exp.ablation or '—'}")
    print(f"  Rounds={exp.num_rounds}  Trial={exp.trial_id}  Seed={exp.seed}")
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

    filtered = {k: v for k, v in data_map.items()
                if v["config"]["attack"] == "adaptive" and v["config"]["partition"] == "iid"}

    if not filtered:
        print("No matching results found for adaptive attack IID.")
        return

    num_rounds = experiments[0].num_rounds
    obs_window = 5  # observation window of AdaptiveAttacker
    poison_rounds = max(0, num_rounds - obs_window)

    report_path = output_dir / "ADAPTIVE_EVAL_RESULTS.md"
    with open(report_path, "w") as f:
        f.write("# Evaluation Results — Adaptive Attack (IID)\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Clients: {base_config['num_clients']}, Galaxies: {base_config['num_galaxies']}\n")
        f.write(f"- Rounds: {num_rounds}, Byzantine: {base_config['byzantine_fraction']:.0%}\n")
        f.write(f"- Partition: IID only\n")
        f.write(f"- Attack: adaptive (observation_window=5, norm_sigma=2.0, "
                f"cos_threshold=0.3, strength=2.0)\n")
        f.write(f"- **Stealth phase**: rounds 0–{obs_window-1} (honest gradients)\n")
        f.write(f"- **Poison phase**: rounds {obs_window}–{num_rounds-1} "
                f"({poison_rounds} rounds of active poisoning)\n\n")

        from collections import defaultdict
        grouped = defaultdict(list)
        for name, d in filtered.items():
            cfg = d["config"]
            key = (cfg["defense"], cfg.get("ablation", ""))
            grouped[key].append(d)

        f.write("## Final Accuracy\n\n")
        f.write("| Defense | Ablation | Final Acc | Std | Convergence (90%) |\n")
        f.write("|---------|----------|:---------:|:---:|:-----------------:|\n")

        defense_order = [
            ("vanilla", ""),
            ("multi_krum", ""),
            ("protogalaxy_full", "merkle_only"),
            ("protogalaxy_full", "zk_merkle"),
        ]
        for defense, ablation in defense_order:
            dlist = grouped.get((defense, ablation), [])
            if not dlist:
                continue
            accs = [d["final_accuracy"] for d in dlist]
            avg_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            convs = [d["convergence_round_90"] for d in dlist]
            valid_conv = [c for c in convs if c >= 0]
            conv_str = f"{statistics.mean(valid_conv):.0f}" if valid_conv else "x"
            abl_str = ablation if ablation else "—"
            f.write(f"| {defense} | {abl_str} | {avg_acc:.4f} | {std_acc:.4f} | {conv_str} |\n")

        f.write("\n## Detection Performance\n\n")
        f.write("| Defense | Ablation | TPR | FPR | F1 | Flagged/Rnd |\n")
        f.write("|---------|----------|:---:|:---:|:--:|:-----------:|\n")

        for defense, ablation in defense_order:
            dlist = grouped.get((defense, ablation), [])
            if not dlist:
                continue
            avg_tpr = statistics.mean([d["avg_tpr"] for d in dlist])
            avg_fpr = statistics.mean([d["avg_fpr"] for d in dlist])
            avg_f1 = statistics.mean([d["avg_f1"] for d in dlist])
            # Avg flagged per round
            avg_flagged = statistics.mean([
                statistics.mean([len(r.get("flagged_ids", [])) for r in d.get("rounds", [])])
                for d in dlist
            ]) if dlist[0].get("rounds") else 0
            abl_str = ablation if ablation else "—"
            f.write(f"| {defense} | {abl_str} | {avg_tpr:.3f} | {avg_fpr:.3f} | "
                    f"{avg_f1:.3f} | {avg_flagged:.1f} |\n")

        f.write("\n## ZKP Proof Metrics\n\n")
        f.write("| Trial | ZKP Prove (s) | ZKP Verify (s) | Failed/Rnd | Gen/Rnd |\n")
        f.write("|:-----:|:---:|:---:|:---:|:---:|\n")

        zkp_data = grouped.get(("protogalaxy_full", "zk_merkle"), [])
        for i, d in enumerate(zkp_data):
            rounds = d.get("rounds", [])
            avg_fail = statistics.mean([r.get("zk_proofs_failed", 0) for r in rounds]) if rounds else 0
            avg_gen = statistics.mean([r.get("zk_proofs_generated", 0) for r in rounds]) if rounds else 0
            f.write(f"| {i} | {d['avg_zk_prove_time']:.2f} | "
                    f"{d['avg_zk_verify_time']:.2f} | {avg_fail:.1f} | {avg_gen:.1f} |\n")

        f.write("\n## Accuracy: Stealth Phase vs Poison Phase\n\n")
        f.write("Shows accuracy at end of observation window (round 4) vs final round.\n\n")
        f.write("| Defense | Ablation | Acc @ Rnd 4 (stealth) | Final Acc | Delta |\n")
        f.write("|---------|----------|:---------------------:|:---------:|:-----:|\n")

        for defense, ablation in defense_order:
            dlist = grouped.get((defense, ablation), [])
            if not dlist:
                continue
            abl_str = ablation if ablation else "—"
            for d in dlist:
                rounds = d.get("rounds", [])
                if len(rounds) >= obs_window:
                    acc_stealth = rounds[obs_window - 1]["accuracy"]
                    acc_final = rounds[-1]["accuracy"]
                    delta = acc_final - acc_stealth
                    f.write(f"| {defense} | {abl_str} | {acc_stealth:.4f} | "
                            f"{acc_final:.4f} | {delta:+.4f} |\n")

        f.write("\n## Timing\n\n")
        f.write("| Defense | Ablation | Avg/Rnd (s) | Total (s) |\n")
        f.write("|---------|----------|:-----------:|:---------:|\n")

        for defense, ablation in defense_order:
            dlist = grouped.get((defense, ablation), [])
            if not dlist:
                continue
            abl_str = ablation if ablation else "—"
            avg_rnd = statistics.mean([d["avg_round_time"] for d in dlist])
            avg_total = statistics.mean([d["total_time"] for d in dlist])
            f.write(f"| {defense} | {abl_str} | {avg_rnd:.2f} | {avg_total:.1f} |\n")

        f.write("\n## Accuracy Trajectories\n\n")
        f.write(f"Stealth phase (rounds 0–{obs_window-1}) -> "
                f"Poison phase (rounds {obs_window}–{num_rounds-1})\n\n")
        f.write("```\n")
        for defense, ablation in defense_order:
            dlist = grouped.get((defense, ablation), [])
            if not dlist:
                continue
            abl = f"+{ablation}" if ablation else ""
            for d in dlist:
                rounds = d.get("rounds", [])
                traj_parts = []
                for i, r in enumerate(rounds):
                    marker = "•" if i < obs_window else "x"
                    traj_parts.append(f"{r['accuracy']:.3f}{marker}")
                traj = " -> ".join(traj_parts)
                f.write(f"{defense}{abl}: {traj}\n")
        f.write("```\n")
        f.write(f"(• = stealth/honest, x = active poison)\n")

        f.write("\n## Key Findings\n\n")

        vanilla = grouped.get(("vanilla", ""), [])
        krum = grouped.get(("multi_krum", ""), [])
        pg_lite = grouped.get(("protogalaxy_full", "merkle_only"), [])
        pg_full = grouped.get(("protogalaxy_full", "zk_merkle"), [])

        if vanilla:
            acc = statistics.mean([d["final_accuracy"] for d in vanilla])
            f.write(f"1. **Vanilla FL under adaptive attack**: {acc:.2%} final accuracy\n")
        if krum and vanilla:
            krum_acc = statistics.mean([d["final_accuracy"] for d in krum])
            van_acc = statistics.mean([d["final_accuracy"] for d in vanilla])
            f.write(f"2. **Krum vs Vanilla**: {krum_acc:.2%} vs {van_acc:.2%} "
                    f"({krum_acc - van_acc:+.2%})\n")
        if pg_full and krum:
            full_acc = statistics.mean([d["final_accuracy"] for d in pg_full])
            krum_acc = statistics.mean([d["final_accuracy"] for d in krum])
            f.write(f"3. **PG-Full (ZKP) vs Krum**: {full_acc:.2%} vs {krum_acc:.2%} "
                    f"({full_acc - krum_acc:+.2%})\n")
            if abs(full_acc - krum_acc) < 0.005:
                f.write(f"   -> ZKP provides **no accuracy benefit** — "
                        f"adaptive poison norms stay within ZKP bounds\n")
            elif full_acc > krum_acc:
                f.write(f"   -> ZKP **catches adaptive poison** that evades Krum!\n")
            else:
                f.write(f"   -> ZKP **worse** — adaptive attack designed to evade norm checks\n")

        if pg_full:
            avg_fpr = statistics.mean([d["avg_fpr"] for d in pg_full])
            avg_tpr = statistics.mean([d["avg_tpr"] for d in pg_full])
            f.write(f"4. **ZKP Detection**: TPR={avg_tpr:.3f}, FPR={avg_fpr:.3f}\n")

            # Check ZKP rejections
            for d in pg_full:
                rounds = d.get("rounds", [])
                poison_rounds_data = rounds[obs_window:]
                if poison_rounds_data:
                    avg_fail = statistics.mean([
                        r.get("zk_proofs_failed", 0) for r in poison_rounds_data
                    ])
                    f.write(f"   -> ZKP rejected {avg_fail:.1f}/round during poison phase\n")

        f.write(f"\n### Adaptive Attack Evasion Analysis\n\n")
        f.write(f"The adaptive attacker crafts gradients with:\n")
        f.write(f"- Norm within mu +/- 2sigma of honest gradient norms\n")
        f.write(f"- Cosine similarity > 0.3 with honest centroid\n\n")
        f.write(f"**ZKP**: Uses server-side median + 3xMAD bounds.\n")
        f.write(f"- If adaptive norm ∈ [mu-2sigma, mu+2sigma] ⊂ [median-3xMAD, median+3xMAD] -> "
                f"passes ZKP x\n")
        f.write(f"- If adaptive norm exceeds server bounds -> caught by ZKP :)\n\n")
        f.write(f"**Krum**: Uses distance-based selection.\n")
        f.write(f"- If cosine(poison, centroid) > 0.3 -> similar direction -> may survive Krum x\n")
        f.write(f"- If attack_strength is high -> distance from honest cluster -> Krum catches :)\n")

        f.write(f"\n---\n*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\n Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Eval: adaptive attack (IID only)")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of FL rounds (default: 10; try 20 for more poison rounds)")
    parser.add_argument("--no-zkp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./eval_adaptive")
    args = parser.parse_args()

    base_config = {
        "num_clients": 10,
        "num_galaxies": 2,
        "num_rounds": args.rounds,
        "byzantine_fraction": 0.3,
        "output_dir": args.output_dir,
    }

    experiments = generate_experiment_matrix(
        num_trials=args.trials,
        include_zkp=not args.no_zkp,
        num_rounds=args.rounds,
    )

    total_est = sum(e.estimated_time_seconds() for e in experiments)
    obs_window = 5
    poison_rounds = max(0, args.rounds - obs_window)

    print(f"\n{'='*80}")
    print(f"ADAPTIVE ATTACK EVALUATION (IID)")
    print(f"{'='*80}")
    print(f"Experiments: {len(experiments)}")
    print(f"Config: {base_config['num_clients']} clients, "
          f"{base_config['num_galaxies']} galaxies, "
          f"{args.rounds} rounds, "
          f"{base_config['byzantine_fraction']:.0%} Byzantine")
    print(f"Adaptive attacker: 5-round observation -> {poison_rounds} rounds active poison")
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
            print(f"\n  Progress: {i}/{len(experiments)} ({i/len(experiments):.0%})")
            print(f"   Elapsed: {elapsed/60:.1f} min, ETA: {eta/60:.1f} min")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"Results: {output_path}")

    create_summary_report(output_path, experiments, base_config)
    print(f"\n Done! Check {output_path}/ADAPTIVE_EVAL_RESULTS.md")


if __name__ == "__main__":
    main()

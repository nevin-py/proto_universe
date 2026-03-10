#!/usr/bin/env python3
"""
bench_attack_robustness.py
==========================
Measures model accuracy under four attack types across all defenses.

  Model   : MLP (109 K params, 6 gradient tensors per client)
  Clients : 10, galaxies 2
  Rounds  : 10 per trial
  Trials  : 3  (seeds 42 / 43 / 44)
  Byzantine: 30% of clients

  Defenses tested:
    vanilla           - no defense (trimmed mean only)
    multi_krum        - multi-Krum aggregation
    fizk_norm         - Merkle + L2 norm filter  (no ZKP)
    protogalaxy_full  - FiZK-Full (Merkle + ZKP range proof)

  Attacks tested:
    none              - clean baseline
    model_poisoning   - scale gradients 10x
    backdoor          - targeted pattern injection
    label_flip        - global label permutation

  Output: Eval_results/benchmarks/attack_robustness/custom/<id>.json

  Estimated runtime (GPU: RTX 4050):
    vanilla / multi_krum / fizk_norm   ~15 min total each
    protogalaxy_full                   ~3 - 4 h
    Grand total                        ~4 - 5 h

  RESUMABLE: existing JSON files are skipped automatically.
  Run from the workspace root:
      python benchmarks/bench_attack_robustness.py
      python benchmarks/bench_attack_robustness.py --dry-run   # list experiments
      python benchmarks/bench_attack_robustness.py --summary   # print table from done results
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_evaluation import ExperimentConfig, ExperimentResult, run_single_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench.attack_robustness")

OUT_DIR = ROOT / "Eval_results" / "benchmarks" / "attack_robustness" / "custom"

# ── Experiment matrix ──────────────────────────────────────────────────────────
DEFENSES  = ["vanilla", "multi_krum", "fizk_norm", "protogalaxy_full"]
ATTACKS   = ["none", "model_poisoning", "backdoor", "label_flip"]
BYZ_FRAC  = 0.30
TRIALS    = 3
NUM_CLIENTS   = 10
NUM_GALAXIES  = 2
NUM_ROUNDS    = 10
MODEL_TYPE    = "linear"       # 7,850 params, 2 gradient tensors/client — fast enough for ZKP


def build_configs() -> List[ExperimentConfig]:
    configs = []
    for trial in range(TRIALS):
        seed = 42 + trial
        for defense in DEFENSES:
            for attack in ATTACKS:
                byz = BYZ_FRAC if attack != "none" else 0.0
                cfg = ExperimentConfig(
                    mode="custom",
                    trial_id=trial,
                    seed=seed,
                    dataset="mnist",
                    model_type=MODEL_TYPE,
                    partition="iid",
                    num_clients=NUM_CLIENTS,
                    num_galaxies=NUM_GALAXIES,
                    num_rounds=NUM_ROUNDS,
                    local_epochs=1,
                    batch_size=64,
                    learning_rate=0.01,
                    defense=defense,
                    aggregation_method="multi_krum",
                    attack=attack,
                    byzantine_fraction=byz,
                    attack_scale=10.0,
                )
                configs.append(cfg)
    return configs


def is_done(cfg: ExperimentConfig) -> bool:
    return (OUT_DIR / f"{cfg.experiment_id}.json").exists()


def save_result(result: ExperimentResult) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{result.config.experiment_id}.json"
    with open(p, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return p


def print_summary():
    """Print a Markdown table from all completed results in OUT_DIR."""
    import os
    files = sorted(OUT_DIR.glob("*.json")) if OUT_DIR.exists() else []
    if not files:
        print("No results found.")
        return

    rows = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
            c = d["config"]
            rows.append({
                "defense": c["defense"],
                "attack": c["attack"],
                "trial": c["trial_id"],
                "byz": c["byzantine_fraction"],
                "acc": d["final_accuracy"],
                "f1": d["avg_f1"],
                "round_t": d["avg_round_time"],
            })
        except Exception:
            continue

    # Aggregate mean over trials
    from collections import defaultdict
    import numpy as np
    groups = defaultdict(list)
    for r in rows:
        groups[(r["defense"], r["attack"])].append(r)

    header = f"{'Defense':<20} {'Attack':<18} {'Byz':>5} {'Acc':>8} {'F1':>8} {'RT(s)':>8}"
    print("\n" + "=" * 72)
    print("Attack Robustness — mean over trials")
    print("=" * 72)
    print(header)
    print("-" * 72)
    for defense in DEFENSES:
        for attack in ATTACKS:
            key = (defense, attack)
            if key not in groups:
                continue
            g = groups[key]
            acc = float(np.mean([r["acc"] for r in g]))
            f1  = float(np.mean([r["f1"]  for r in g]))
            rt  = float(np.mean([r["round_t"] for r in g]))
            byz = g[0]["byz"]
            n   = len(g)
            print(f"{defense:<20} {attack:<18} {byz:>5.0%} {acc:>8.4f} {f1:>8.4f} {rt:>8.1f}  (n={n})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Attack robustness benchmark")
    parser.add_argument("--dry-run",  action="store_true", help="List experiments without running")
    parser.add_argument("--summary",  action="store_true", help="Print results table and exit")
    parser.add_argument("--defense",  nargs="+", default=None, choices=DEFENSES,
                        help="Only run these defenses")
    parser.add_argument("--attack",   nargs="+", default=None, choices=ATTACKS,
                        help="Only run these attacks")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    configs = build_configs()
    if args.defense:
        configs = [c for c in configs if c.defense in args.defense]
    if args.attack:
        configs = [c for c in configs if c.attack in args.attack]

    pending = [c for c in configs if not is_done(c)]
    done    = len(configs) - len(pending)

    log.info(f"Attack robustness benchmark: {len(configs)} total, {done} already done, {len(pending)} pending")

    if args.dry_run:
        print(f"\n{'#':<4} {'Experiment ID':<65} {'Status'}")
        print("-" * 80)
        for i, c in enumerate(configs):
            status = "DONE" if is_done(c) else "pending"
            print(f"{i+1:<4} {c.experiment_id:<65} {status}")
        return

    if not pending:
        log.info("All experiments done. Run --summary to see results.")
        print_summary()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR.parent / "run.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    t_start   = time.time()
    completed = 0
    failed    = 0

    for i, cfg in enumerate(pending):
        idx = done + i + 1
        total = len(configs)
        log.info(f"[{idx}/{total}] {cfg.experiment_id}")
        t0 = time.time()
        try:
            result = run_single_experiment(cfg)
            p = save_result(result)
            elapsed = time.time() - t0
            completed += 1
            log.info(
                f"  [OK] acc={result.final_accuracy:.4f}  f1={result.avg_f1:.4f}"
                f"  rt={result.avg_round_time:.1f}s/round  elapsed={elapsed:.0f}s  -> {p.name}"
            )
        except Exception as e:
            failed += 1
            elapsed = time.time() - t0
            log.error(f"  [FAIL] {cfg.experiment_id}: {e}  ({elapsed:.0f}s)")

        # Running ETA
        if completed > 0:
            spent = time.time() - t_start
            avg   = spent / (completed + failed)
            remaining = len(pending) - (i + 1)
            eta_s = avg * remaining
            log.info(f"  ETA: {eta_s/60:.0f} min ({remaining} experiments left)")

    wall = time.time() - t_start
    log.info(f"\nDone: {completed} OK, {failed} failed — {wall/60:.1f} min total")
    print_summary()


if __name__ == "__main__":
    main()

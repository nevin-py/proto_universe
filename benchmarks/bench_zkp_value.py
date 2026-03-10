#!/usr/bin/env python3
"""
bench_zkp_value.py
==================
Demonstrates the cryptographic security properties that ZKP adds beyond
a plain norm filter (fizk_norm).

Two ZKP-specific attack scenarios:

  gradient_substitution   Byzantine client commits honest gradient hashes
                          in Phase 1, substitutes poisoned gradients in Phase 2.
                          fizk_norm: undetectable (only checks final norms).
                          protogalaxy_full: commitment hash mismatch → rejected.

  compromised_aggregator  Server silently re-admits one norm-violating
                          Byzantine client after the filter check.
                          fizk_norm: no evidence — client is silently included.
                          protogalaxy_full: no valid IVC proof for that client
                                           → zk_proofs_failed count is auditable.

Plus a clean-run overhead comparison (no attack) to measure the ZKP cost.

  Model   : MLP (109 K params)
  Clients : 10, galaxies 2
  Rounds  : 8 per trial
  Trials  : 3  (seeds 42 / 43 / 44)

  Defenses:
    vanilla           - baseline (no filter, no ZKP)
    fizk_norm         - Merkle + L2 norm filter only
    protogalaxy_full  - FiZK-Full  (norm filter + ZKP range proof)

  Output: Eval_results/benchmarks/zkp_value/custom/<id>.json

  Estimated runtime:
    vanilla                        ~5 min
    fizk_norm                      ~12 min
    protogalaxy_full               ~2.5 - 3.5 h
    Grand total                    ~3 h

  RESUMABLE: existing JSON files are skipped automatically.
  Run from workspace root:
      python benchmarks/bench_zkp_value.py
      python benchmarks/bench_zkp_value.py --dry-run
      python benchmarks/bench_zkp_value.py --summary
"""

import argparse
import json
import logging
import sys
import time
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
log = logging.getLogger("bench.zkp_value")

OUT_DIR = ROOT / "Eval_results" / "benchmarks" / "zkp_value" / "custom"

DEFENSES = ["vanilla", "fizk_norm", "protogalaxy_full"]
SCENARIOS = [
    # (attack_name,               byz_frac,  description)
    ("none",                      0.0,   "Clean baseline — no attack"),
    ("model_poisoning",           0.30,  "Standard attack — ZKP and filter both catch"),
    ("gradient_substitution",     0.30,  "Bypass via commit-then-substitute (30% byz)"),
    ("gradient_substitution",     0.50,  "High-rate substitution — robustness under pressure"),
    ("gradient_substitution",     0.70,  "Extreme-rate substitution — system survivability"),
    ("compromised_aggregator",    0.20,  "Server bypasses norm filter for 1 client"),
]
TRIALS       = 3
NUM_CLIENTS  = 10
NUM_GALAXIES = 2
NUM_ROUNDS   = 8
MODEL_TYPE   = "linear"   # 7,850 params — fast enough for ZKP overhead comparison


def build_configs() -> List[ExperimentConfig]:
    configs = []
    for trial in range(TRIALS):
        seed = 42 + trial
        for defense in DEFENSES:
            for attack, byz, _ in SCENARIOS:
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
    import numpy as np
    from collections import defaultdict

    files = sorted(OUT_DIR.glob("*.json")) if OUT_DIR.exists() else []
    if not files:
        print("No results yet.")
        return

    rows = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
            c = d["config"]
            # Sum zk_proofs_failed across rounds
            zk_failed = sum(r.get("zk_proofs_failed", 0) for r in d.get("rounds", []))
            rows.append({
                "defense": c["defense"],
                "attack":  c["attack"],
                "byz":     c["byzantine_fraction"],
                "trial":   c["trial_id"],
                "acc":     d["final_accuracy"],
                "f1":      d["avg_f1"],
                "rt":      d["avg_round_time"],
                "zk_fail": zk_failed,
            })
        except Exception:
            continue

    groups = defaultdict(list)
    for r in rows:
        groups[(r["defense"], r["attack"], r["byz"])].append(r)

    attack_order = [(a, b) for a, b, _ in SCENARIOS]

    print("\n" + "=" * 90)
    print("ZKP Security Value — mean over trials")
    print("  zk_fail = total ZKP proof failures across all rounds (>0 = cryptographic rejection)")
    print("=" * 90)
    print(f"{'Defense':<22} {'Attack':<28} {'Byz':>5} {'Acc':>8} {'F1':>8} {'RT(s)':>8} {'ZKFail':>8}  n")
    print("-" * 90)
    for defense in DEFENSES:
        for attack, byz in attack_order:
            key = (defense, attack, byz)
            if key not in groups:
                continue
            g   = groups[key]
            acc = float(np.mean([r["acc"]     for r in g]))
            f1  = float(np.mean([r["f1"]      for r in g]))
            rt  = float(np.mean([r["rt"]      for r in g]))
            zkf = float(np.mean([r["zk_fail"] for r in g]))
            n   = len(g)
            print(f"{defense:<22} {attack:<28} {byz:>5.0%} {acc:>8.4f} {f1:>8.4f} {rt:>8.1f} {zkf:>8.1f}  {n}")
        print()

    print()
    print("Key comparisons (ZKP survivability at high Byzantine rates):")
    seen = set()
    for attack, byz, _ in SCENARIOS:
        if attack not in ("gradient_substitution", "compromised_aggregator"):
            continue
        if (attack, byz) in seen:
            continue
        seen.add((attack, byz))
        fn_key = ("fizk_norm",        attack, byz)
        pg_key = ("protogalaxy_full", attack, byz)
        if fn_key not in groups and pg_key not in groups:
            continue
        print(f"  {attack} @ {byz:.0%} byz:")
        for label, key in [("fizk_norm", fn_key), ("protogalaxy_full", pg_key)]:
            if key in groups:
                g = groups[key]
                acc = float(np.mean([r["acc"] for r in g]))
                zkf = float(np.mean([r["zk_fail"] for r in g]))
                print(f"    {label:<20} acc={acc:.4f}  zk_fail={zkf:.1f}")
        if fn_key in groups and pg_key in groups:
            diff = float(np.mean([r["zk_fail"] for r in groups[pg_key]])) \
                 - float(np.mean([r["zk_fail"] for r in groups[fn_key]]))
            if diff > 0:
                print(f"    -> ZKP adds {diff:.0f} cryptographic rejections per run")
    print()


def main():
    parser = argparse.ArgumentParser(description="ZKP cryptographic value benchmark")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--defense",  nargs="+", default=None, choices=DEFENSES)
    parser.add_argument("--attack",   nargs="+", default=None)
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

    log.info(f"ZKP value benchmark: {len(configs)} total, {done} done, {len(pending)} pending")

    if args.dry_run:
        print(f"\n{'#':<4} {'Experiment ID':<65} {'Status'}")
        print("-" * 80)
        for i, c in enumerate(configs):
            print(f"{i+1:<4} {c.experiment_id:<65} {'DONE' if is_done(c) else 'pending'}")
        return

    if not pending:
        log.info("All done.")
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
        log.info(f"[{idx}/{len(configs)}] {cfg.experiment_id}")
        t0 = time.time()
        try:
            result = run_single_experiment(cfg)
            # Count ZKP failures across rounds for quick display
            zk_fail = sum(r.zk_proofs_failed for r in result.rounds)
            p = save_result(result)
            elapsed = time.time() - t0
            completed += 1
            log.info(
                f"  [OK] acc={result.final_accuracy:.4f}  f1={result.avg_f1:.4f}"
                f"  zk_fail={zk_fail}  rt={result.avg_round_time:.1f}s/round  ({elapsed:.0f}s)"
            )
        except Exception as e:
            failed += 1
            log.error(f"  [FAIL] {cfg.experiment_id}: {e}  ({time.time()-t0:.0f}s)")

        if completed > 0:
            spent = time.time() - t_start
            eta_s = (spent / (completed + failed)) * (len(pending) - i - 1)
            log.info(f"  ETA: {eta_s/60:.0f} min")

    log.info(f"Done: {completed} OK, {failed} failed — {(time.time()-t_start)/60:.1f} min")
    print_summary()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bench_overhead.py
=================
Detailed timing breakdown of where each round's time is spent — comparing
training only, Merkle + norm filter, and full ZKP.

This is the fastest benchmark (~45 minutes).  Run it first to confirm
the setup works before starting the longer benchmarks.

  Model   : MLP (109 K params)
  Clients : 10, galaxies 2
  Rounds  : 5 per trial  (clean run — no Byzantine)
  Trials  : 3  (seeds 42 / 43 / 44)
  Attack  : none

  Defenses:
    vanilla           - SGD + trimmed mean only
    fizk_norm         - + Merkle commitments + L2 norm filter
    protogalaxy_full  - + ProtoGalaxy IVC ZKP range proof

  Metrics captured per round:
    round_time        total wall-clock time
    phase1_time       client training + gradient submission
    phase2_time       server aggregation
    phase3_time       defense check (norm filter / ZKP verify)
    zk_prove_time     ProtoGalaxy proof generation (per client, summed)
    zk_verify_time    ProtoGalaxy IVC verification
    merkle_build_time Merkle tree construction
    bytes_sent        gradient communication bytes

  Output: Eval_results/benchmarks/overhead/custom/<id>.json

  Estimated runtime: ~45 min total.

  RESUMABLE: existing JSON files are skipped automatically.
  Run from workspace root:
      python benchmarks/bench_overhead.py
      python benchmarks/bench_overhead.py --dry-run
      python benchmarks/bench_overhead.py --summary
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
log = logging.getLogger("bench.overhead")

OUT_DIR = ROOT / "Eval_results" / "benchmarks" / "overhead" / "custom"

DEFENSES     = ["vanilla", "fizk_norm", "protogalaxy_full"]
TRIALS       = 3
NUM_CLIENTS  = 10
NUM_GALAXIES = 2
NUM_ROUNDS   = 5
MODEL_TYPE   = "linear"   # 7,850 params — fast enough for ZKP overhead comparison


def build_configs() -> List[ExperimentConfig]:
    configs = []
    for trial in range(TRIALS):
        seed = 42 + trial
        for defense in DEFENSES:
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
                attack="none",
                byzantine_fraction=0.0,
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
            rounds_data = d.get("rounds", [])
            # Average timing fields across rounds (all are in seconds)
            def avg_field(field):
                vals = [r.get(field, 0.0) for r in rounds_data if r.get(field) is not None]
                return float(np.mean(vals)) if vals else 0.0

            rows.append({
                "defense":     c["defense"],
                "trial":       c["trial_id"],
                "round_t":     avg_field("round_time"),
                "phase1_t":    avg_field("phase1_time"),
                "phase2_t":    avg_field("phase2_time"),
                "phase3_t":    avg_field("phase3_time"),
                "zk_prove_t":  avg_field("zk_prove_time"),
                "zk_verify_t": avg_field("zk_verify_time"),
                "merkle_t":    avg_field("merkle_build_time"),
                "bytes":       d.get("total_bytes", 0),
                "acc":         d.get("final_accuracy", 0.0),
            })
        except Exception:
            continue

    groups = defaultdict(list)
    for r in rows:
        groups[r["defense"]].append(r)

    print("\n" + "=" * 90)
    print("Overhead Breakdown — avg per round (seconds), mean over trials")
    print("=" * 90)
    print(f"{'Defense':<22} {'Total':>7} {'Phase1':>8} {'Phase2':>8} {'Phase3':>8}"
          f" {'ZKProve':>9} {'ZKVerif':>9} {'Merkle':>8} {'Acc':>7}")
    print("-" * 90)

    baseline_rt = None
    for defense in DEFENSES:
        if defense not in groups:
            continue
        g   = groups[defense]
        rt  = float(np.mean([r["round_t"]     for r in g]))
        p1  = float(np.mean([r["phase1_t"]    for r in g]))
        p2  = float(np.mean([r["phase2_t"]    for r in g]))
        p3  = float(np.mean([r["phase3_t"]    for r in g]))
        zkp = float(np.mean([r["zk_prove_t"]  for r in g]))
        zkv = float(np.mean([r["zk_verify_t"] for r in g]))
        mkl = float(np.mean([r["merkle_t"]    for r in g]))
        acc = float(np.mean([r["acc"]         for r in g]))
        n   = len(g)

        if defense == "vanilla":
            baseline_rt = rt

        ratio = f"({rt/baseline_rt:.1f}x)" if baseline_rt and baseline_rt > 0 else ""
        print(f"{defense:<22} {rt:>7.2f} {p1:>8.2f} {p2:>8.2f} {p3:>8.2f}"
              f" {zkp:>9.2f} {zkv:>9.2f} {mkl:>8.2f} {acc:>7.4f}  {ratio}  n={n}")

    print()
    print("Phase legend:")
    print("  Phase1 = client training + gradient upload")
    print("  Phase2 = server aggregation")
    print("  Phase3 = defense check (norm filter / ZKP verification)")
    print("  ZKProve = total ZKP proof generation time (all clients summed per round)")
    print("  ZKVerif = total ZKP IVC verification time per round")
    print("  Merkle  = Merkle tree build + verification time")
    print()


def main():
    parser = argparse.ArgumentParser(description="Overhead breakdown benchmark")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--defense",  nargs="+", default=None, choices=DEFENSES)
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    configs = build_configs()
    if args.defense:
        configs = [c for c in configs if c.defense in args.defense]

    pending = [c for c in configs if not is_done(c)]
    done    = len(configs) - len(pending)

    log.info(f"Overhead benchmark: {len(configs)} total, {done} done, {len(pending)} pending")

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
            p = save_result(result)
            elapsed = time.time() - t0
            completed += 1
            zkp_t = result.avg_zk_prove_time
            log.info(
                f"  [OK] acc={result.final_accuracy:.4f}  rt={result.avg_round_time:.2f}s"
                f"  zk_prove={zkp_t:.2f}s  elapsed={elapsed:.0f}s"
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

#!/usr/bin/env python3
"""
bench_scalability.py
====================
Measures how round time and ZKP overhead scale with client count.

We keep the model small (linear, 7 850 params) so the bottleneck is
visible — ZKP proof generation grows linearly with client count while
vanilla aggregation barely changes.

  Model   : linear  (2 gradient tensors/client — isolates ZKP scaling)
  Clients : 10, 20, 30, 50
  Galaxies: auto-scaled (clients // 5, min 2)
  Rounds  : 5 per trial  (short — we care about per-round cost)
  Trials  : 3  (seeds 42 / 43 / 44)
  Attack  : none  (clean — isolates scaling overhead)

  Defenses:
    vanilla           - baseline (no overhead)
    fizk_norm         - Merkle + norm filter
    protogalaxy_full  - FiZK-Full  (most expensive, shows scaling slope)

  Output: Eval_results/benchmarks/scalability/custom/<id>.json

  Estimated runtime:
    vanilla  (all counts)          ~5 min
    fizk_norm (all counts)         ~10 min
    protogalaxy_full 10 clients    ~15 min
    protogalaxy_full 20 clients    ~30 min
    protogalaxy_full 30 clients    ~45 min
    protogalaxy_full 50 clients    ~75 min
    Grand total                    ~3 h

  The plot_scalability() function saves a PNG of round-time vs client count.

  RESUMABLE: existing JSON files are skipped automatically.
  Run from workspace root:
      python benchmarks/bench_scalability.py
      python benchmarks/bench_scalability.py --dry-run
      python benchmarks/bench_scalability.py --summary
      python benchmarks/bench_scalability.py --plot        # requires matplotlib
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
log = logging.getLogger("bench.scalability")

OUT_DIR = ROOT / "Eval_results" / "benchmarks" / "scalability" / "custom"

CLIENT_COUNTS = [10, 20, 30, 50]
DEFENSES      = ["vanilla", "fizk_norm", "protogalaxy_full"]
TRIALS        = 3
NUM_ROUNDS    = 5
MODEL_TYPE    = "linear"   # fast — bottleneck is ZKP, not model


def _num_galaxies(clients: int) -> int:
    return max(2, clients // 5)


def build_configs() -> List[ExperimentConfig]:
    configs = []
    for trial in range(TRIALS):
        seed = 42 + trial
        for clients in CLIENT_COUNTS:
            galaxies = _num_galaxies(clients)
            for defense in DEFENSES:
                cfg = ExperimentConfig(
                    mode="custom",
                    trial_id=trial,
                    seed=seed,
                    dataset="mnist",
                    model_type=MODEL_TYPE,
                    partition="iid",
                    num_clients=clients,
                    num_galaxies=galaxies,
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
            rows.append({
                "defense":  c["defense"],
                "clients":  c["num_clients"],
                "rt":       d["avg_round_time"],
                "zk_prove": d["avg_zk_prove_time"],
                "bytes":    d["total_bytes"],
            })
        except Exception:
            continue

    groups = defaultdict(list)
    for r in rows:
        groups[(r["defense"], r["clients"])].append(r)

    print("\n" + "=" * 72)
    print("Scalability — avg round time (s) by defense × client count")
    print("=" * 72)

    # Header: client counts as columns
    col_w = 10
    header = f"{'Defense':<22}" + "".join(f"{c:>{col_w}}" for c in CLIENT_COUNTS)
    print(header)
    print("-" * (22 + col_w * len(CLIENT_COUNTS)))
    for defense in DEFENSES:
        row = f"{defense:<22}"
        for clients in CLIENT_COUNTS:
            key = (defense, clients)
            if key in groups:
                rt = float(np.mean([r["rt"] for r in groups[key]]))
                row += f"{rt:>{col_w}.1f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    print()
    # ZKP overhead ratio vs vanilla
    print("ZKP overhead ratio vs vanilla (protogalaxy_full / vanilla):")
    for clients in CLIENT_COUNTS:
        vk = ("vanilla",        clients)
        pk = ("protogalaxy_full", clients)
        if vk in groups and pk in groups:
            vt = float(np.mean([r["rt"] for r in groups[vk]]))
            pt = float(np.mean([r["rt"] for r in groups[pk]]))
            ratio = pt / vt if vt > 0 else float("nan")
            print(f"  {clients:>3} clients: {pt:.1f}s / {vt:.1f}s = {ratio:.1f}x")
    print()


def plot_scalability():
    """Save a PNG chart of round-time vs client count per defense."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    files = sorted(OUT_DIR.glob("*.json")) if OUT_DIR.exists() else []
    if not files:
        print("No results yet.")
        return

    rows = []
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
            rows.append({
                "defense": d["config"]["defense"],
                "clients": d["config"]["num_clients"],
                "rt":      d["avg_round_time"],
            })
        except Exception:
            continue

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["defense"], r["clients"])].append(r["rt"])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"vanilla": "#2196F3", "fizk_norm": "#FF9800", "protogalaxy_full": "#4CAF50"}
    for defense in DEFENSES:
        xs, ys, yerr = [], [], []
        for clients in CLIENT_COUNTS:
            key = (defense, clients)
            if key in groups:
                vals = groups[key]
                xs.append(clients)
                ys.append(float(np.mean(vals)))
                yerr.append(float(np.std(vals)))
        if xs:
            ax.errorbar(xs, ys, yerr=yerr, label=defense,
                        color=colors.get(defense), marker="o", linewidth=2,
                        capsize=4)

    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Avg round time (s)")
    ax.set_title("Scalability: round time vs client count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CLIENT_COUNTS)

    out_png = OUT_DIR.parent / "scalability.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Scalability benchmark")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--summary",  action="store_true")
    parser.add_argument("--plot",     action="store_true", help="Generate PNG chart")
    parser.add_argument("--clients",  nargs="+", type=int, default=None,
                        help=f"Subset of client counts to run (default: {CLIENT_COUNTS})")
    parser.add_argument("--defense",  nargs="+", default=None, choices=DEFENSES)
    args = parser.parse_args()

    if args.summary or args.plot:
        print_summary()
        if args.plot:
            plot_scalability()
        return

    configs = build_configs()
    if args.clients:
        configs = [c for c in configs if c.num_clients in args.clients]
    if args.defense:
        configs = [c for c in configs if c.defense in args.defense]

    pending = [c for c in configs if not is_done(c)]
    done    = len(configs) - len(pending)

    log.info(f"Scalability benchmark: {len(configs)} total, {done} done, {len(pending)} pending")

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
        log.info(f"[{idx}/{len(configs)}] {cfg.experiment_id}  ({cfg.num_clients} clients)")
        t0 = time.time()
        try:
            result = run_single_experiment(cfg)
            p = save_result(result)
            elapsed = time.time() - t0
            completed += 1
            log.info(
                f"  [OK] rt={result.avg_round_time:.1f}s/round  "
                f"total={result.total_time:.0f}s  elapsed={elapsed:.0f}s  -> {p.name}"
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

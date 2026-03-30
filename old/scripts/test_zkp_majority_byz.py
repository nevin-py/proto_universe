#!/usr/bin/env python3
"""
Test: Can ZKP+FedAvg handle majority Byzantine clients (60%, 70%, 80%)?

Hypothesis: ZKP circuit rejects all out-of-bound Byzantine gradients
cryptographically. After rejection, only honest gradients remain.
FedAvg over the honest subset should converge regardless of how many
Byzantines there were, as long as ≥1 honest client exists.

Control: protogalaxy_full + multi_krum (the current default) — known to
collapse at 60% due to multi_krum's 2f+3 requirement.

Experiment: protogalaxy_full + trimmed_mean(trim=0) = FedAvg post-ZKP.
"""
import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_evaluation import ExperimentConfig, run_single_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test.zkp_majority_byz")

OUT_DIR = ROOT / "Eval_results" / "test_zkp_majority_byz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment matrix ──────────────────────────────────────────────────────
# (defense, aggregation_method, trim_ratio, label)
CONFIGS = [
    # Controls — current setup, known to fail at 60%+
    ("protogalaxy_full", "multi_krum",    0.1, "ZKP+MultiKrum  (current)"),
    # Hypothesis — ZKP filter first, then plain FedAvg over honest subset
    ("protogalaxy_full", "trimmed_mean",  0.0, "ZKP+FedAvg     (hypothesis)"),
    # Sanity — no ZKP, plain FedAvg — should always collapse
    ("vanilla",          "trimmed_mean",  0.0, "Vanilla+FedAvg (sanity)"),
]

BYZ_FRACTIONS = [0.50, 0.60, 0.70, 0.80]
ATTACK        = "model_poisoning"   # large-norm injection — clearly violates ZKP bound
ATTACK_SCALE  = 10.0
TRIALS        = 2
NUM_CLIENTS   = 10
NUM_ROUNDS    = 10


def experiment_id(defense, agg, byz, trial):
    byz_tag = f"byz{int(byz*100)}"
    agg_tag = agg.replace("_", "")
    return f"{defense}__{agg_tag}__{ATTACK}__{byz_tag}__t{trial}"


def run():
    results = []

    for defense, agg, trim, label in CONFIGS:
        for byz in BYZ_FRACTIONS:
            for trial in range(TRIALS):
                exp_id = experiment_id(defense, agg, byz, trial)
                out_path = OUT_DIR / f"{exp_id}.json"

                if out_path.exists():
                    log.info(f"[SKIP] {exp_id} — already done")
                    with open(out_path) as f:
                        d = json.load(f)
                    results.append((label, byz, trial, d["final_accuracy"],
                                    sum(r.get("zk_proofs_failed", 0) for r in d.get("rounds", []))))
                    continue

                log.info(f"[RUN ] {exp_id}  ({label} @ {byz:.0%} byz)")

                cfg = ExperimentConfig(
                    mode="custom",
                    trial_id=trial,
                    seed=42 + trial,
                    dataset="mnist",
                    model_type="linear",
                    partition="iid",
                    num_clients=NUM_CLIENTS,
                    num_galaxies=2,
                    num_rounds=NUM_ROUNDS,
                    local_epochs=1,
                    batch_size=64,
                    learning_rate=0.01,
                    defense=defense,
                    aggregation_method=agg,
                    trim_ratio=trim,
                    attack=ATTACK,
                    byzantine_fraction=byz,
                    attack_scale=ATTACK_SCALE,
                )

                try:
                    result = run_single_experiment(cfg)
                    zk_fail = sum(r.zk_proofs_failed for r in result.rounds)
                    with open(out_path, "w") as f:
                        json.dump(result.to_dict(), f, indent=2, default=str)
                    log.info(f"  → acc={result.final_accuracy:.4f}  zk_fail={zk_fail}")
                    results.append((label, byz, trial, result.final_accuracy, zk_fail))
                except Exception as e:
                    log.error(f"  FAILED: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    import numpy as np
    from collections import defaultdict

    print("\n" + "=" * 80)
    print("RESULT: ZKP Majority-Byzantine Test")
    print("Attack: model_poisoning, scale=10x, 10 clients, 10 rounds, linear model")
    print("=" * 80)
    print(f"\n  {'Config':<30} {'Byz':>5}  {'Acc (mean)':>12}  {'ZKFail (mean)':>14}  n")
    print("  " + "-"*70)

    g = defaultdict(list)
    for label, byz, trial, acc, zkf in results:
        g[(label, byz)].append((acc, zkf))

    for defense, agg, trim, label in CONFIGS:
        for byz in BYZ_FRACTIONS:
            key = (label, byz)
            if key not in g:
                continue
            accs = [r[0] for r in g[key]]
            zkfs = [r[1] for r in g[key]]
            print(f"  {label:<30} {byz:>5.0%}  {np.mean(accs):>12.4f}  {np.mean(zkfs):>14.1f}  {len(accs)}")
        print()

    print("\nInterpretation:")
    print("  ZKP+FedAvg should hold acc ~0.92 at 60-80% byz (ZKP rejects all Byzantine,")
    print("  FedAvg over residual honest clients converges normally)")
    print("  ZKP+MultiKrum should collapse at 60%+ (2f+3 > n constraint violated)")
    print("  Vanilla+FedAvg should collapse at all fractions (no filter)")


if __name__ == "__main__":
    run()

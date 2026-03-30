#!/usr/bin/env python3
"""Quick focused test: ZKP+FedAvg vs ZKP+MultiKrum at 60% Byzantine."""
import sys, json, logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_evaluation import ExperimentConfig, run_single_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

OUT_DIR = ROOT / "Eval_results" / "test_zkp_majority_byz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (defense, aggregation_method, trim_ratio, label)
CONFIGS = [
    # ("protogalaxy_full", "multi_krum",   0.1, "ZKP+MultiKrum  (control)"),
    ("protogalaxy_full", "trimmed_mean", 0.0, "ZKP+FedAvg     (hypothesis)"),
    ("vanilla",          "trimmed_mean", 0.0, "Vanilla+FedAvg (sanity)"),
]
BYZ     = 0.60
TRIALS  = 2
ATTACK  = "model_poisoning"
SCALE   = 10.0
CLIENTS = 10
ROUNDS  = 10

results = []
total   = len(CONFIGS) * TRIALS
done    = 0

for defense, agg, trim, label in CONFIGS:
    for trial in range(TRIALS):
        agg_tag  = agg.replace("_", "")
        exp_id   = f"{defense}__{agg_tag}__{ATTACK}__byz60__t{trial}"
        out_file = OUT_DIR / f"{exp_id}.json"

        if out_file.exists():
            d = json.loads(out_file.read_text())
            acc = d.get("final_accuracy", 0)
            rzf = sum(r.get("zk_proofs_failed", 0) for r in d.get("rounds", []))
            print(f"[CACHED] {label:30s} trial={trial}  acc={acc:.4f}  zk_fail={rzf}")
            results.append((label, trial, acc, rzf))
            done += 1
            continue

        done += 1
        print(f"\n[{done}/{total}] {label} | trial={trial}")

        cfg = ExperimentConfig(
            experiment_id    = exp_id,
            defense          = defense,
            aggregation_method = agg,
            trim_ratio       = trim,
            byzantine_fraction = BYZ,
            attack           = ATTACK,
            attack_scale     = SCALE,
            num_clients      = CLIENTS,
            num_rounds       = ROUNDS,
            num_galaxies     = 2,
            trial_id         = trial,
            seed             = 42 + trial,
        )

        result = run_single_experiment(cfg)
        if result is None:
            print(f"  ERROR: experiment returned None")
            continue

        d = result.to_dict()
        out_file.write_text(json.dumps(d, indent=2, default=str))

        acc = result.final_accuracy
        rzf = sum(r.zk_proofs_failed for r in result.rounds)
        rbv = sum(getattr(r, 'zk_bound_violations', 0) for r in result.rounds)
        print(f"  acc={acc:.4f}  zk_fail={rzf}  bnd_viol={rbv}")
        results.append((label, trial, acc, rzf))

print("\n" + "="*60)
print("SUMMARY — 60% Byzantine, model_poisoning, scale=10")
print("="*60)
print(f"{'Config':<30} {'t0 acc':>8} {'t1 acc':>8} {'avg zk_fail':>12}")
print("-"*60)

by_label: dict = {}
for label, trial, acc, zkf in results:
    by_label.setdefault(label, []).append((acc, zkf))

for (defense, agg, trim, label) in CONFIGS:
    rows = by_label.get(label, [])
    accs = [r[0] for r in rows]
    zkfs = [r[1] for r in rows]
    acc_str = "  ".join(f"{a:.4f}" for a in accs)
    zkf_avg = sum(zkfs) / max(len(zkfs), 1)
    verdict = "✓ HOLDS" if (accs and min(accs) > 0.80) else "✗ COLLAPSES"
    print(f"  {label:<28} {acc_str:>18}   zk_fail={zkf_avg:.1f}  {verdict}")

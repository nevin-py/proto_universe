#!/usr/bin/env python3
"""
ZKP Byzantine-Majority Robustness Test
=======================================
Demonstrate that the ProtoGalaxy ZKP circuit (25th-pct + pure-relative-margin
bound) still rejects Byzantine clients at 40 / 50 / 60 / 70 % fractions.

3 configs × 4 byz fracs × 2 trials = 24 experiments.

Output layout (mirrors run_evaluation.py)
-----------------------------------------
Eval_results/test_byz_majority_proof/
    custom/                        ← per-experiment JSON files
        protogalaxy_full__...json
        ...
    logs/                          ← timestamped log file
        eval_byz_majority_<ts>.log
    resource_usage_report.txt
    resource_usage_report.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import helpers from run_evaluation so behaviour is identical
from scripts.run_evaluation import (
    ExperimentConfig,
    ExperimentResult,
    RoundMetrics,
    _is_completed,
    _save_result,
    run_single_experiment,
    save_resource_report,
)

# ---------------------------------------------------------------------------
# Output directory  (Eval_results/test_byz_majority_proof/)
# ---------------------------------------------------------------------------
OUT_DIR = ROOT / "Eval_results" / "test_byz_majority_proof"
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — dual console + file  (same format as run_evaluation.py)
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = LOG_DIR / f"eval_byz_majority_{_ts}.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
for _h in root_logger.handlers[:]:          # clear any handlers from imports
    root_logger.removeHandler(_h)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
root_logger.addHandler(_ch)

_fh = logging.FileHandler(str(_log_file), mode="w")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(
    logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
root_logger.addHandler(_fh)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {_log_file}")

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
CONFIGS = [
    # (defense, aggregation, trim_ratio, label)
    ("protogalaxy_full", "trimmed_mean", 0.0, "ZKP+FedAvg  (fixed bound)"),
    ("protogalaxy_full", "multi_krum",   0.1, "ZKP+MultiKrum (baseline)"),
    ("vanilla",          "trimmed_mean", 0.0, "Vanilla+FedAvg (sanity)"),
]
BYZ_FRACTIONS = [0.40, 0.50, 0.60, 0.70]
ATTACK        = "model_poisoning"
SCALE         = 10.0
TRIALS        = 2
CLIENTS       = 10
ROUNDS        = 10

total = len(CONFIGS) * len(BYZ_FRACTIONS) * TRIALS

logger.info("=" * 72)
logger.info("  ZKP BYZANTINE-MAJORITY ROBUSTNESS TEST")
logger.info("=" * 72)
logger.info(f"  Attack:       {ATTACK}  scale={SCALE}x")
logger.info(f"  Clients:      {CLIENTS}  |  Rounds: {ROUNDS}  |  Trials: {TRIALS}")
logger.info(f"  Byz frac:     {[f'{b:.0%}' for b in BYZ_FRACTIONS]}")
logger.info(f"  Experiments:  {total}")
logger.info(f"  Output dir:   {OUT_DIR}")
logger.info("=" * 72)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_round_fields = {f.name for f in dataclasses.fields(RoundMetrics)}

all_results: list[ExperimentResult] = []
failed:      list[tuple[str, str]]  = []
done = 0

for defense, agg, trim, label in CONFIGS:
    for byz in BYZ_FRACTIONS:
        for trial in range(TRIALS):
            done += 1
            agg_tag = agg.replace("_", "")
            byz_tag = f"byz{int(byz * 100)}"
            exp_id  = f"{defense}__{agg_tag}__{ATTACK}__{byz_tag}__t{trial}"

            cfg = ExperimentConfig(
                experiment_id      = exp_id,
                mode               = "custom",
                defense            = defense,
                aggregation_method = agg,
                trim_ratio         = trim,
                byzantine_fraction = byz,
                attack             = ATTACK,
                attack_scale       = SCALE,
                num_clients        = CLIENTS,
                num_rounds         = ROUNDS,
                num_galaxies       = 2,
                trial_id           = trial,
                seed               = 42 + trial,
            )

            print(f"\n{'─' * 72}")
            print(f"  [{done:2d}/{total}] {label}  |  byz={byz:.0%}  |  trial={trial}")
            print(f"{'─' * 72}")

            if _is_completed(cfg, str(OUT_DIR)):
                logger.info("  Already completed — loading from cache")
                cached = OUT_DIR / "custom" / f"{exp_id}.json"
                if cached.exists():
                    d = json.loads(cached.read_text())
                    r = ExperimentResult.__new__(ExperimentResult)
                    r.config         = cfg
                    r.final_accuracy = d.get("final_accuracy", 0.0)
                    r.avg_tpr        = d.get("avg_tpr", 0.0)
                    r.resource_usage = d.get("resource_usage", {})
                    r.rounds = [
                        dataclasses.replace(
                            RoundMetrics(),
                            **{k: v for k, v in rd.items() if k in _round_fields},
                        )
                        for rd in d.get("rounds", [])
                    ]
                    zkf = sum(rnd.zk_proofs_failed for rnd in r.rounds)
                    logger.info(
                        f"  CACHED  acc={r.final_accuracy:.4f}  "
                        f"zk_fail={zkf}  tpr={r.avg_tpr:.3f}"
                    )
                    all_results.append(r)
                continue

            try:
                result = run_single_experiment(cfg)
                result.compute_aggregates()
                _save_result(result, str(OUT_DIR))
                all_results.append(result)

                zkf = sum(rnd.zk_proofs_failed for rnd in result.rounds)
                logger.info(
                    f"  → acc={result.final_accuracy:.4f}  "
                    f"zk_fail={zkf}  tpr={result.avg_tpr:.3f}"
                )
            except Exception as exc:
                logger.error(f"  FAILED: {exc}")
                logger.debug(traceback.format_exc())
                failed.append((exp_id, str(exc)))

# ---------------------------------------------------------------------------
# Resource usage report  (mirrors run_evaluation.py)
# ---------------------------------------------------------------------------
if all_results:
    save_resource_report(all_results, str(OUT_DIR))
    logger.info(f"Resource report saved to {OUT_DIR}/resource_usage_report.txt")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("RESULT SUMMARY — ZKP Byzantine Majority Robustness")
print(f"Attack: {ATTACK}, scale={SCALE}x, {CLIENTS} clients, {ROUNDS} rounds, IID MNIST")
print("=" * 78)
print(
    f"\n  {'Config':<26}  {'Byz':>5}  {'Acc mean±std':>14}  "
    f"{'ZKFail/rnd':>11}  {'TPR':>6}  Verdict"
)
print("  " + "-" * 74)

g: dict = defaultdict(list)
for r in all_results:
    zkf = sum(rnd.zk_proofs_failed for rnd in r.rounds)
    g[(r.config.defense, r.config.aggregation_method, r.config.byzantine_fraction)].append(
        (r.final_accuracy, zkf, r.avg_tpr)
    )

for defense, agg, trim, label in CONFIGS:
    for byz in BYZ_FRACTIONS:
        key = (defense, agg, byz)
        if key not in g:
            continue
        rows      = g[key]
        accs      = [row[0] for row in rows]
        zkfs      = [row[1] for row in rows]
        tprs      = [row[2] for row in rows]
        mean_acc      = float(np.mean(accs))
        std_acc       = float(np.std(accs))
        zkf_per_round = float(np.mean(zkfs)) / ROUNDS
        mean_tpr      = float(np.mean(tprs))
        verdict       = "HOLDS" if mean_acc > 0.80 else "COLLAPSES"
        print(
            f"  {label:<26}  {byz:>5.0%}  {mean_acc:.4f}±{std_acc:.4f}"
            f"  {zkf_per_round:>11.1f}  {mean_tpr:>6.3f}  {verdict}"
        )
    print()

print(
    "\nKey:"
    "\n  ZKFail/rnd = avg ZK proof failures per round (Byzantine clients caught by circuit)"
    "\n  TPR        = True Positive Rate (Byzantine detection rate)"
    "\n  HOLDS/COLLAPSES = final accuracy above/below 80%"
)

if failed:
    print(f"\n  {len(failed)} experiment(s) failed:")
    for eid, reason in failed:
        print(f"    - {eid}: {reason}")

print(f"\n  Results saved to: {OUT_DIR}/")
logger.info("Done.")

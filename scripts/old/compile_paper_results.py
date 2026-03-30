#!/usr/bin/env python3
"""
Compile FiZK experiment results into paper-ready LaTeX tables and analysis.

Reads JSON results from eval_results/custom/ and produces:
  1. Table 3: Adaptive attack accuracy (IID)
  2. Table 4: Model poisoning + backdoor (IID)
  3. Table 5: ZKP proof failure rate
  4. Table 6: Other attacks (label_flip, gaussian, targeted)
  5. Table 7: Detection performance (TPR/FPR/F1)
  6. Table 8: Computational overhead
  7. Table NI: Non-IID results (Dirichlet)
  8. Clean baselines

Usage:
    python scripts/compile_paper_results.py [--latex]
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# Default; overridden by --results-dir at runtime
RESULTS_DIR = Path("eval_results/custom")

# Defense display names
DEFENSE_NAMES = {
    "vanilla": "Vanilla",
    "multi_krum": "Multi-Krum",
    "protogalaxy_full__merkle_only": "FiZK-Lite",
    "protogalaxy_full": "FiZK-Full",
}

DEFENSE_ORDER = ["vanilla", "multi_krum", "protogalaxy_full__merkle_only", "protogalaxy_full"]


def load_all_results():
    """Load all JSON result files, keyed by experiment ID."""
    results = {}
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            # Key: strip trial suffix for grouping
            key = f.stem  # e.g. "protogalaxy_full__adaptive__iid__c10_g2__byz50__t0__merkle_only"
            results[key] = data
        except Exception as e:
            print(f"  WARN: Failed to load {f.name}: {e}", file=sys.stderr)
    return results


def make_key(defense, attack, partition, byz_pct, trial=0, ablation=""):
    """Build the result JSON filename stem."""
    byz_int = int(round(byz_pct * 100))
    key = f"{defense}__{attack}__{partition}__c10_g2__byz{byz_int}__t{trial}"
    if ablation:
        key += f"__{ablation}"
    return key


def get_result(results, defense, attack, partition, byz_pct, ablation=""):
    """Look up a single experiment result. Returns None if missing."""
    key = make_key(defense, attack, partition, byz_pct, 0, ablation)
    return results.get(key)


def fmt_acc(val, std=None):
    """Format accuracy as percentage string, optionally with ±std."""
    if val is None:
        return "—"
    s = f"{val * 100:.1f}"
    if std is not None and std > 0.0005:
        s += f"±{std * 100:.1f}"
    return s


def fmt_pct(val):
    """Format a 0-1 value as percentage."""
    if val is None:
        return "—"
    return f"{val * 100:.1f}"


def fmt_time(val):
    """Format time in seconds."""
    if val is None:
        return "—"
    return f"{val:.1f}"


def get_defense_result(results, defense, attack, partition, byz_pct):
    """Get result for a defense, averaging over all available trials."""
    if defense == "protogalaxy_full__merkle_only":
        return get_result_avg(results, "protogalaxy_full", attack, partition, byz_pct, "merkle_only")
    return get_result_avg(results, defense, attack, partition, byz_pct)


def get_result_avg(results, defense, attack, partition, byz_pct, ablation=""):
    """Collect all trials for a config and return an averaged pseudo-result."""
    byz_int = int(round(byz_pct * 100))
    trials_found = []
    for trial in range(10):  # check up to 10 trials
        key = make_key(defense, attack, partition, byz_pct, trial, ablation)
        r = results.get(key)
        if r is not None:
            trials_found.append(r)
    if not trials_found:
        return None
    # Average scalar metrics across trials
    averaged = dict(trials_found[0])  # copy structure
    scalar_keys = [
        "final_accuracy", "best_accuracy", "avg_tpr", "avg_fpr",
        "avg_precision", "avg_f1", "avg_round_time", "avg_zk_prove_time",
        "avg_zk_verify_time", "avg_merkle_time", "total_time",
    ]
    for k in scalar_keys:
        vals = [r[k] for r in trials_found if k in r and r[k] is not None]
        if vals:
            averaged[k] = sum(vals) / len(vals)
            averaged[f"{k}_std"] = (sum((v - averaged[k])**2 for v in vals) / max(len(vals)-1, 1)) ** 0.5
    averaged["_num_trials"] = len(trials_found)
    return averaged


def extract_metric(r, metric):
    """Extract a metric from a result dict."""
    if r is None:
        return None
    return r.get(metric)


def extract_metric_with_std(r, metric):
    """Return (mean, std) tuple from an averaged result."""
    if r is None:
        return None, None
    return r.get(metric), r.get(f"{metric}_std")


def print_separator(title, char="═"):
    width = 80
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def compile_table3(results):
    """Table 3: Adaptive Attack — IID, varying Byzantine fraction."""
    print_separator("TABLE 3: Test Accuracy Under Adaptive Attack (IID)")
    print(f"\n{'Byz %':<8}", end="")
    for d in DEFENSE_ORDER:
        print(f"{DEFENSE_NAMES[d]:<14}", end="")
    print()
    print("-" * 64)

    for byz in [0.20, 0.30, 0.40, 0.50]:
        print(f"{int(byz*100)}%{'':<5}", end="")
        for defense in DEFENSE_ORDER:
            r = get_defense_result(results, defense, "adaptive", "iid", byz)
            acc, std = extract_metric_with_std(r, "final_accuracy")
            n = r.get("_num_trials", 1) if r else 1
            cell = fmt_acc(acc, std if n > 1 else None)
            print(f"{cell:<16}", end="")
        print()


def compile_table4(results):
    """Table 4: Model Poisoning + Backdoor — IID."""
    print_separator("TABLE 4: Model Poisoning & Backdoor (IID)")

    for attack in ["model_poisoning", "backdoor"]:
        print(f"\n  Attack: {attack}")
        print(f"  {'Byz %':<8}", end="")
        for d in DEFENSE_ORDER:
            print(f"{DEFENSE_NAMES[d]:<14}", end="")
        print()
        print("  " + "-" * 62)

        for byz in [0.30, 0.50]:
            print(f"  {int(byz*100)}%{'':<5}", end="")
            for defense in DEFENSE_ORDER:
                r = get_defense_result(results, defense, attack, "iid", byz)
                acc, std = extract_metric_with_std(r, "final_accuracy")
                n = r.get("_num_trials", 1) if r else 1
                cell = fmt_acc(acc, std if n > 1 else None)
                print(f"{cell:<16}", end="")
            print()


def compile_detection_table(results):
    """Table 7: Detection Performance — TPR, FPR, Precision, F1."""
    print_separator("TABLE 7: Detection Performance (IID)")

    attacks = ["adaptive", "model_poisoning", "backdoor"]
    byz_levels = [0.30, 0.50]
    defenses_det = ["multi_krum", "protogalaxy_full__merkle_only", "protogalaxy_full"]

    print(f"\n{'Attack':<18}{'Byz%':<6}{'Defense':<14}{'TPR':<8}{'FPR':<8}{'Prec':<8}{'F1':<8}")
    print("-" * 70)

    for attack in attacks:
        for byz in byz_levels:
            for defense in defenses_det:
                r = get_defense_result(results, defense, attack, "iid", byz)
                if r is None:
                    continue
                tpr = extract_metric(r, "avg_tpr")
                fpr = extract_metric(r, "avg_fpr")
                prec = extract_metric(r, "avg_precision")
                f1 = extract_metric(r, "avg_f1")
                print(
                    f"{attack:<18}{int(byz*100)}%{'':<3}"
                    f"{DEFENSE_NAMES[defense]:<14}"
                    f"{fmt_pct(tpr):<8}{fmt_pct(fpr):<8}"
                    f"{fmt_pct(prec):<8}{fmt_pct(f1):<8}"
                )


def compile_zkp_table(results):
    """Table 5: ZKP Proof Failure Rate."""
    print_separator("TABLE 5: ZKP Proof Failure Rate (FiZK-Full, IID)")

    attacks = ["adaptive", "model_poisoning", "backdoor"]
    byz_levels = [0.20, 0.30, 0.40, 0.50]

    print(f"\n{'Attack':<20}{'Byz%':<6}{'Trials':<8}{'Proofs Gen':<12}{'Rejected':<10}{'Reject%':<10}")
    print("-" * 66)

    for attack in attacks:
        for byz in byz_levels:
            # Aggregate across all available trials
            total_gen = total_fail = total_bound = total_expected = 0
            n_trials = 0
            for trial in range(10):
                key = make_key("protogalaxy_full", attack, "iid", byz, trial)
                r = results.get(key)
                if r is None:
                    continue
                n_trials += 1
                rounds = r.get("rounds", [])
                total_gen += sum(rd.get("zk_proofs_generated", 0) for rd in rounds)
                total_fail += sum(rd.get("zk_proofs_failed", 0) for rd in rounds)
                expected = len(rounds) * 10  # 10 clients per round
                total_expected += expected
                bv = expected - total_gen - total_fail
                total_bound += max(0, bv)
            if n_trials == 0:
                continue
            total_rejected = total_fail + total_bound
            reject_rate = (total_rejected / max(total_expected, 1)) * 100

            print(
                f"{attack:<20}{int(byz*100)}%{'':<3}"
                f"{n_trials:<8}{total_gen:<12}{total_rejected:<10}"
                f"{reject_rate:.1f}%"
            )


def compile_overhead_table(results):
    """Table 8: Computational Overhead."""
    print_separator("TABLE 8: Computational Overhead (IID, adaptive, 30% Byz)")

    defenses_overhead = DEFENSE_ORDER

    print(f"\n{'Defense':<16}{'Avg Round(s)':<14}{'ZK Prove(s)':<13}{'ZK Verify(s)':<14}{'Merkle(ms)':<12}{'Total(min)'}")
    print("-" * 80)

    for defense in defenses_overhead:
        r = get_defense_result(results, defense, "adaptive", "iid", 0.30)
        if r is None:
            continue
        avg_round = extract_metric(r, "avg_round_time")
        avg_prove = extract_metric(r, "avg_zk_prove_time") or 0
        avg_verify = extract_metric(r, "avg_zk_verify_time") or 0
        avg_merkle = extract_metric(r, "avg_merkle_time") or 0
        total = extract_metric(r, "total_time") or 0

        print(
            f"{DEFENSE_NAMES[defense]:<16}"
            f"{fmt_time(avg_round):<14}"
            f"{fmt_time(avg_prove):<13}"
            f"{fmt_time(avg_verify):<14}"
            f"{avg_merkle*1000:.2f}{'':<6}"
            f"{total/60:.1f}"
        )


def compile_other_attacks(results):
    """Table 6: Other Attacks — label_flip, gaussian_noise, targeted_label_flip."""
    print_separator("TABLE 6: Other Attacks (IID)")

    attacks = ["label_flip", "gaussian_noise", "targeted_label_flip"]
    defenses_other = ["vanilla", "multi_krum", "protogalaxy_full"]

    for attack in attacks:
        print(f"\n  Attack: {attack}")
        print(f"  {'Byz %':<8}", end="")
        for d in defenses_other:
            print(f"{DEFENSE_NAMES[d]:<14}", end="")
        print()
        print("  " + "-" * 50)

        for byz in [0.30, 0.50]:
            print(f"  {int(byz*100)}%{'':<5}", end="")
            for defense in defenses_other:
                r = get_defense_result(results, defense, attack, "iid", byz)
                acc = extract_metric(r, "final_accuracy")
                print(f"{fmt_acc(acc):<14}", end="")
            print()


def compile_noniid(results):
    """Non-IID Results (Dirichlet α=0.5)."""
    print_separator("TABLE NI: Non-IID Results (Dirichlet α=0.5)")

    attacks = ["adaptive", "model_poisoning", "label_flip"]
    defenses_ni = ["vanilla", "multi_krum", "protogalaxy_full"]

    for attack in attacks:
        print(f"\n  Attack: {attack}")
        print(f"  {'Byz %':<8}", end="")
        for d in defenses_ni:
            print(f"{DEFENSE_NAMES[d]:<14}", end="")
        print()
        print("  " + "-" * 50)

        for byz in [0.30, 0.50]:
            print(f"  {int(byz*100)}%{'':<5}", end="")
            for defense in defenses_ni:
                r = get_defense_result(results, defense, attack, "dirichlet", byz)
                acc = extract_metric(r, "final_accuracy")
                print(f"{fmt_acc(acc):<14}", end="")
            print()


def compile_baselines(results):
    """Clean baselines (no attack)."""
    print_separator("BASELINES: Clean (No Attack)")

    print(f"\n{'Partition':<14}{'Vanilla':<14}{'FiZK-Full':<14}")
    print("-" * 42)

    for partition in ["iid", "dirichlet"]:
        print(f"{partition:<14}", end="")
        for defense in ["vanilla", "protogalaxy_full"]:
            r = get_result(results, defense, "none", partition, 0.0)
            acc = extract_metric(r, "final_accuracy")
            print(f"{fmt_acc(acc):<14}", end="")
        print()


def compile_ablation(results):
    """Ablation: FiZK-Lite vs FiZK-Full."""
    print_separator("TABLE 9: Ablation — FiZK-Lite vs FiZK-Full (IID)")

    attacks = ["adaptive", "model_poisoning", "backdoor"]

    print(f"\n{'Attack':<20}{'Byz%':<6}{'FiZK-Lite':<14}{'FiZK-Full':<14}{'Δ Acc':<10}")
    print("-" * 64)

    for attack in attacks:
        for byz in [0.30, 0.50]:
            r_lite = get_defense_result(results, "protogalaxy_full__merkle_only", attack, "iid", byz)
            r_full = get_defense_result(results, "protogalaxy_full", attack, "iid", byz)
            acc_lite = extract_metric(r_lite, "final_accuracy")
            acc_full = extract_metric(r_full, "final_accuracy")
            delta = ""
            if acc_lite is not None and acc_full is not None:
                d = (acc_full - acc_lite) * 100
                delta = f"{d:+.1f}pp"
            print(
                f"{attack:<20}{int(byz*100)}%{'':<3}"
                f"{fmt_acc(acc_lite):<14}{fmt_acc(acc_full):<14}{delta}"
            )


def compile_latex_table3(results):
    """Generate LaTeX for Table 3."""
    print("\n% LaTeX: Table 3 — Adaptive Attack Accuracy (IID)")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Test accuracy (\\%) under adaptive attack (IID, MNIST).}")
    print("\\label{tab:adaptive}")
    print("\\begin{tabular}{l" + "c" * len(DEFENSE_ORDER) + "}")
    print("\\toprule")
    header = " & ".join([DEFENSE_NAMES[d] for d in DEFENSE_ORDER])
    print(f"$\\alpha$ & {header} \\\\")
    print("\\midrule")

    for byz in [0.20, 0.30, 0.40, 0.50]:
        row = [f"{int(byz*100)}\\%"]
        best = -1
        vals = []
        for defense in DEFENSE_ORDER:
            r = get_defense_result(results, defense, "adaptive", "iid", byz)
            acc = extract_metric(r, "final_accuracy")
            vals.append(acc)
            if acc is not None and acc > best:
                best = acc
        for acc in vals:
            if acc is not None and abs(acc - best) < 0.001:
                row.append(f"\\textbf{{{fmt_acc(acc)}}}")
            else:
                row.append(fmt_acc(acc))
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def summary_stats(results):
    """Print summary statistics."""
    print_separator("SUMMARY")

    # Count results
    total = len(results)
    attacks = set()
    defenses = set()
    for key in results:
        parts = key.split("__")
        if len(parts) >= 2:
            defenses.add(parts[0])
            attacks.add(parts[1])

    print(f"\n  Total experiments:  {total}")
    print(f"  Unique defenses:    {sorted(defenses)}")
    print(f"  Unique attacks:     {sorted(attacks)}")

    # Find best FiZK-Full results
    print("\n  Key Highlights:")
    for attack in ["adaptive", "model_poisoning"]:
        for byz in [0.30, 0.50]:
            r_v = get_result(results, "vanilla", attack, "iid", byz)
            r_f = get_result(results, "protogalaxy_full", attack, "iid", byz)
            if r_v and r_f:
                acc_v = r_v.get("final_accuracy", 0)
                acc_f = r_f.get("final_accuracy", 0)
                delta = (acc_f - acc_v) * 100
                print(
                    f"    {attack} @ {int(byz*100)}% Byz: "
                    f"Vanilla={fmt_acc(acc_v)}% → FiZK-Full={fmt_acc(acc_f)}% "
                    f"({delta:+.1f}pp)"
                )


def compile_scale_sensitivity(results):
    """Table S: Attack scale sensitivity (FiZK-Full vs Multi-Krum, model_poisoning)."""
    print_separator("TABLE S: Attack Scale Sensitivity (model_poisoning, IID)")

    scales = [1.0, 2.0, 5.0, 10.0, 20.0]
    defenses = ["vanilla", "multi_krum", "protogalaxy_full"]
    col_w = 22

    for byz in [0.3, 0.5]:
        byz_int = int(byz * 100)
        print(f"\n  Byzantine fraction: {byz_int}%")
        print(f"  {'Scale':<8}", end="")
        for d in defenses:
            print(f"{DEFENSE_NAMES[d]:>{col_w}}", end="")
        print()
        print("  " + "-" * (8 + col_w * len(defenses)))

        for scale in scales:
            suffix = f"__scale{scale:g}" if scale != 10.0 else ""
            row = f"  {scale:<8.1f}"
            found_any = False
            for defense in defenses:
                key0 = f"{defense}__model_poisoning__iid__c10_g2__byz{byz_int}__t0{suffix}"
                r = results.get(key0)
                if r is None:
                    row += f"{'—':>{col_w}}"
                else:
                    found_any = True
                    acc = r.get("final_accuracy")
                    tpr = r.get("avg_tpr")
                    cell = f"{fmt_acc(acc)}% tpr={fmt_pct(tpr)}%"
                    row += f"{cell:>{col_w}}"
            if found_any:
                print(row)


def main():
    parser = argparse.ArgumentParser(description="Compile FiZK paper results")
    parser.add_argument("--results-dir", default=None,
                        help="Path to custom results dir (default: eval_results/custom)")
    parser.add_argument("--latex", action="store_true", help="Also emit LaTeX tables")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir is not None:
        RESULTS_DIR = Path(args.results_dir)
    latex_mode = args.latex

    print("=" * 80)
    print("  FiZK Paper Results Compilation")
    print(f"  Source: {RESULTS_DIR}/")
    print("=" * 80)

    results = load_all_results()
    print(f"\n  Loaded {len(results)} experiment results.")

    compile_baselines(results)
    compile_table3(results)
    compile_table4(results)
    compile_zkp_table(results)
    compile_other_attacks(results)
    compile_detection_table(results)
    compile_overhead_table(results)
    compile_ablation(results)
    compile_noniid(results)
    compile_scale_sensitivity(results)
    summary_stats(results)

    if latex_mode:
        print("\n")
        compile_latex_table3(results)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

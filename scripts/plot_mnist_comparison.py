#!/usr/bin/env python3
"""
Generate two 2-subplot comparison figures from friend's MNIST eval data:
  Figure 1: MNIST IID        — left: byz30, right: byz50
  Figure 2: MNIST non-IID    — left: byz30, right: byz50

Each subplot: 3 grouped bars (Vanilla, Multi-Krum, FiZK) across 3 attacks.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = "/tmp/mnist_friend/Eval_results/benchmarks/attack_robustness/custom"
OUT_DIR  = "/run/media/vane/Data/Project/proto_universe/Eval_results"
os.makedirs(OUT_DIR, exist_ok=True)

DEFENSES = {
    "vanilla":          "Vanilla FedAvg",
    "multi_krum":       "Multi-Krum",
    "fizk_norm":        "FiZK (ProtoGalaxy)",
}
ATTACKS  = ["backdoor", "label_flip", "model_poisoning"]
ATTACK_LABELS = ["Backdoor", "Label Flip", "Model Poisoning"]
BYZ_LEVELS = ["byz30", "byz50"]
BYZ_LABELS = [r"$\alpha=0.30$", r"$\alpha=0.50$"]

# Colours matching existing plot_attack_comparison.py style
COLORS = {
    "vanilla":   "#4878d0",
    "multi_krum":"#ee854a",
    "fizk_norm": "#6acc65",
}
HATCH = {
    "vanilla":   "",
    "multi_krum":"//",
    "fizk_norm": "xx",
}

# ── Data loading helper ──────────────────────────────────────────────────────
def load_accuracy(defense, attack, split_prefix, byz):
    """Average final_accuracy over all available trials for a given key."""
    pattern = os.path.join(
        DATA_DIR,
        f"{defense}__{attack}__{split_prefix}__{byz}__t*.json"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    accs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        acc = d.get("final_accuracy")
        if acc is not None:
            accs.append(acc * 100.0)
    return float(np.mean(accs)) if accs else None

# ── Plotting function ─────────────────────────────────────────────────────────
def make_figure(split_prefix, fig_title, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.01)

    n_attacks  = len(ATTACKS)
    n_defenses = len(DEFENSES)
    x          = np.arange(n_attacks)
    total_w    = 0.65
    bar_w      = total_w / n_defenses
    offsets    = np.linspace(-(total_w - bar_w) / 2,
                             (total_w - bar_w) / 2,
                             n_defenses)

    for ax_idx, (byz, byz_label) in enumerate(zip(BYZ_LEVELS, BYZ_LABELS)):
        ax = axes[ax_idx]

        for d_idx, (def_key, def_label) in enumerate(DEFENSES.items()):
            values = []
            for attack in ATTACKS:
                acc = load_accuracy(def_key, attack, split_prefix, byz)
                values.append(acc if acc is not None else 0.0)

            bars = ax.bar(
                x + offsets[d_idx],
                values,
                width=bar_w,
                color=COLORS[def_key],
                hatch=HATCH[def_key],
                edgecolor="white",
                linewidth=0.8,
                label=def_label,
            )
            # Annotate bars with value
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#333333"
                    )

        # 50% chance line
        ax.axhline(50, color="grey", linestyle="--", linewidth=0.9, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(ATTACK_LABELS, fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_title(f"Byzantine fraction {byz_label}", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend below both subplots
    handles = [
        mpatches.Patch(facecolor=COLORS[k], hatch=HATCH[k],
                       edgecolor="white", label=v)
        for k, v in DEFENSES.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{out_stem}.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_figure(
        split_prefix="iid__c10_g2",
        fig_title="MNIST — IID Setting: Attack Robustness Comparison",
        out_stem="mnist_iid_comparison",
    )
    make_figure(
        split_prefix="dirichlet__c10_g2",
        fig_title=r"MNIST — Non-IID (Dirichlet $\beta=0.5$) Setting: Attack Robustness Comparison",
        out_stem="mnist_noniid_comparison",
    )
    print("All figures generated.")

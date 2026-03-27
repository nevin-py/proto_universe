#!/usr/bin/env python3
"""
Generates two publication-quality figures, each with 2×2 subplots:

  Figure 1 — MNIST (from friend/Eval_results/benchmarks/attack_lite):
      [IID  α=0.30]  [IID  α=0.50]
      [nIID α=0.30]  [nIID α=0.50]

  Figure 2 — CNN (from friend/Eval_results/benchmarks/eval_cnn):
      [IID  α=0.30]  [IID  α=0.50]
      [nIID α=0.30]  [nIID α=0.50]

Each subplot: 3 grouped bars (Vanilla / Multi-Krum / FiZK) × 3 attacks.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
FRIEND_BASE = "/run/media/vane/Data/Project/proto_universe/friend/Eval_results/benchmarks"
MNIST_DIR   = os.path.join(FRIEND_BASE, "attack_lite", "custom")
CNN_DIR     = os.path.join(FRIEND_BASE, "eval_cnn",    "custom")
OUT_DIR     = "/run/media/vane/Data/Project/proto_universe/Eval_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Legend labels / keys ─────────────────────────────────────────────────────
# Both benchmarks use "protogalaxy_full" as the FiZK key in the friend folder
DATASETS = {
    "mnist": {
        "data_dir": MNIST_DIR,
        "defenses": [
            ("vanilla",          "Vanilla FedAvg"),
            ("multi_krum",       "Multi-Krum"),
            ("protogalaxy_full", "FiZK (ProtoGalaxy)"),
        ],
        "title": "MNIST — Attack Robustness Comparison",
        "out_stem": "mnist_full_comparison",
    },
    "cnn": {
        "data_dir": CNN_DIR,
        "defenses": [
            ("vanilla",          "Vanilla FedAvg"),
            ("multi_krum",       "Multi-Krum"),
            ("protogalaxy_full", "FiZK (ProtoGalaxy)"),
        ],
        "title": "CNN — Attack Robustness Comparison",
        "out_stem": "cnn_full_comparison",
    },
}

ATTACKS       = ["backdoor",  "label_flip",      "model_poisoning"]
ATTACK_LABELS = ["Backdoor",  "Label Flip",       "Model Poisoning"]

SUBPLOTS = [
    ("iid__c10_g2",       "byz30", "IID,  " + r"$\alpha=0.30$"),
    ("iid__c10_g2",       "byz50", "IID,  " + r"$\alpha=0.50$"),
    ("dirichlet__c10_g2", "byz30", r"Non-IID,  $\alpha=0.30$"),
    ("dirichlet__c10_g2", "byz50", r"Non-IID,  $\alpha=0.50$"),
]

# Colours + hatch per defense position (shared across both datasets)
PALETTE = ["#4878d0", "#ee854a", "#6acc65"]
HATCHES = ["",        "//",       "xx"]

# ── Helper ────────────────────────────────────────────────────────────────────
def load_acc(data_dir, defense, attack, split_prefix, byz):
    pattern = os.path.join(data_dir, f"{defense}__{attack}__{split_prefix}__{byz}__t*.json")
    files   = glob.glob(pattern)
    if not files:
        return None
    accs = []
    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        v = d.get("final_accuracy")
        if v is not None:
            accs.append(v * 100.0)
    return float(np.mean(accs)) if accs else None

# ── Main figure builder ───────────────────────────────────────────────────────
def make_figure(dataset_key):
    cfg      = DATASETS[dataset_key]
    data_dir = cfg["data_dir"]
    defenses = cfg["defenses"]   # list of (key, label)
    n_def    = len(defenses)
    n_att    = len(ATTACKS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle(cfg["title"], fontsize=20, fontweight="bold", y=1.01)

    x       = np.arange(n_att)
    total_w = 0.65
    bar_w   = total_w / n_def
    offsets = np.linspace(-(total_w - bar_w) / 2,
                           (total_w - bar_w) / 2, n_def)

    for sp_idx, (split, byz, sp_label) in enumerate(SUBPLOTS):
        row, col = divmod(sp_idx, 2)
        ax = axes[row][col]

        for d_idx, (def_key, _) in enumerate(defenses):
            raw_values = [
                load_acc(data_dir, def_key, attack, split, byz)
                for attack in ATTACKS
            ]

            for a_idx, val in enumerate(raw_values):
                bx = x[a_idx] + offsets[d_idx]
                if val is None:
                    # Training failed (gradient norms → ∞) — stub bar + "N/A" label
                    ax.bar(bx, 3, width=bar_w,
                           color="#cccccc",
                           edgecolor="white", linewidth=0.8)
                    ax.text(bx, 4.5, "N/A",
                            ha="center", va="bottom",
                            fontsize=14, color="#555555",
                            style="italic")
                    continue
                ax.bar(bx, val, width=bar_w,
                       color=PALETTE[d_idx],
                       hatch=HATCHES[d_idx],
                       edgecolor="white", linewidth=0.8)
                if val > 1.0:
                    ax.text(bx, val + 0.8, f"{val:.1f}",
                            ha="center", va="bottom",
                            fontsize=14, color="#333333")

        ax.axhline(50, color="grey", linestyle="--", linewidth=0.9, alpha=0.55, label="_nolegend_")
        ax.set_xticks(x)
        ax.set_xticklabels(ATTACK_LABELS, fontsize=14)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Test Accuracy (%)", fontsize=14)
        ax.set_title(sp_label, fontsize=14, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend below the figure
    handles = [
        mpatches.Patch(facecolor=PALETTE[i], hatch=HATCHES[i],
                       edgecolor="white", label=label)
        for i, (_, label) in enumerate(defenses)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=14, frameon=True,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("pdf", "png"):
        out_path = os.path.join(OUT_DIR, f"{cfg['out_stem']}.{ext}")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure("mnist")
    make_figure("cnn")
    print("Done.")

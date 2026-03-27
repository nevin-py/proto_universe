"""
4-subplot attack accuracy comparison figure.
  Top-left : MNIST Linear, IID, α=0.30
  Top-right: MNIST Linear, IID, α=0.50
  Bot-left : CNN,          IID, α=0.30
  Bot-right: CNN,          Dirichlet (non-IID), α=0.30
"""

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────────
BASE_MNIST = "/run/media/vane/Data/Project/proto_universe/Eval_results/benchmarks/attack_robustness/custom/"
BASE_CNN   = "/run/media/vane/Data/Project/proto_universe/Eval_results/benchmarks/eval_cnn/custom/"
OUT        = "/run/media/vane/Data/Project/proto_universe/Eval_results/attack_comparison.pdf"

# ── helpers ──────────────────────────────────────────────────────────────────
def load_acc(base, defense, attack, split, byz):
    pattern = f"{base}{defense}__{attack}__{split}__c10_g2__{byz}__t*.json"
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    accs = [json.load(open(f)).get("final_accuracy", 0) * 100 for f in files]
    return np.mean(accs), np.std(accs)

# ── data spec ────────────────────────────────────────────────────────────────
ATTACKS   = ["backdoor", "label_flip", "model_poisoning"]
ATK_LABEL = ["Backdoor", "Label Flip", "Model Poisoning"]

DEFENSES = [
    ("vanilla",          "Vanilla FedAvg"),
    ("multi_krum",       "Multi-Krum"),
    ("fizk_norm",        "FiZK (ours)"),       # MNIST key
    ("protogalaxy_full", "FiZK (ours)"),       # CNN  key
]

SUBPLOTS = [
    # (title, base,       def_list_idx,   split,       byz)
    ("MNIST Linear — IID, α=0.30",       BASE_MNIST, [0,1,2],    "iid",       "byz30"),
    ("MNIST Linear — IID, α=0.50",       BASE_MNIST, [0,1,2],    "iid",       "byz50"),
    ("CNN — IID, α=0.30",                BASE_CNN,   [0,1,3],    "iid",       "byz30"),
    ("CNN — Dirichlet (non-IID), α=0.30",BASE_CNN,   [0,1,3],    "dirichlet", "byz30"),
]

# ── colours & style ──────────────────────────────────────────────────────────
COLORS = {
    "Vanilla FedAvg": "#4878d0",
    "Multi-Krum":     "#ee854a",
    "FiZK (ours)":    "#6acc65",   # bold green — our system
}
HATCH = {
    "Vanilla FedAvg": "",
    "Multi-Krum":     "//",
    "FiZK (ours)":    "",
}
EDGE = {
    "Vanilla FedAvg": "#2a4fa8",
    "Multi-Krum":     "#c05a18",
    "FiZK (ours)":    "#1a8c14",
}

n_attacks  = len(ATTACKS)
n_defenses = 3
width      = 0.22
x          = np.arange(n_attacks)
offsets    = np.linspace(-(n_defenses-1)/2, (n_defenses-1)/2, n_defenses) * width

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
axes = axes.flatten()

for ax_idx, (title, base, def_idxs, split, byz) in enumerate(SUBPLOTS):
    ax = axes[ax_idx]
    def_entries = [DEFENSES[i] for i in def_idxs]   # 3 entries

    for d_pos, (def_key, def_label) in enumerate(def_entries):
        means, errs = [], []
        for atk in ATTACKS:
            result = load_acc(base, def_key, atk, split, byz)
            if result is None:
                means.append(np.nan)
                errs.append(0)
            else:
                means.append(result[0])
                errs.append(result[1])

        bars = ax.bar(
            x + offsets[d_pos],
            means,
            width      = width,
            label      = def_label,
            color      = COLORS[def_label],
            edgecolor  = EDGE[def_label],
            linewidth  = 0.8,
            hatch      = HATCH[def_label],
            zorder     = 3,
            yerr       = [e if not np.isnan(m) else 0 for e, m in zip(errs, means)],
            capsize    = 3,
            error_kw   = dict(elinewidth=1.0, ecolor="black", capthick=1.0),
        )

        # annotate FiZK bars for model poisoning (the key result)
        for bar, mean in zip(bars, means):
            if np.isnan(mean):
                continue
            # label only stands-out bars (< 20 or FiZK on model poisoning)
            if mean < 20 or (def_label == "FiZK (ours)" and mean > 70):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{mean:.0f}%",
                    ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                    color=EDGE[def_label],
                )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(ATK_LABEL, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=9)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.axhline(50, color="gray", linewidth=0.7, linestyle="--", zorder=1, alpha=0.6)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=COLORS[l], edgecolor=EDGE[l],
                   hatch=HATCH[l], linewidth=0.8, label=l)
    for l in ["Vanilla FedAvg", "Multi-Krum", "FiZK (ours)"]
]
fig.legend(
    handles     = legend_handles,
    loc         = "lower center",
    ncol        = 3,
    fontsize    = 10,
    frameon     = True,
    framealpha  = 0.9,
    edgecolor   = "#aaaaaa",
    bbox_to_anchor = (0.5, -0.02),
)

fig.suptitle(
    "Byzantine Robustness: MNIST vs CNN under Three Attacks",
    fontsize=12, fontweight="bold", y=1.01,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
# also save as PNG for quick preview
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Saved: {OUT.replace('.pdf', '.png')}")

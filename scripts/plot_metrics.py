#!/usr/bin/env python
"""
scripts/plot_metrics.py
~~~~~~~~~~~~~~~~~~~~~~~
Reads eval JSON files from data/vis/ (single_concept, two_concepts,
three_concepts) and plots how aggregate metrics change as alpha varies
(beta = 1 - alpha, gamma ignored).

Saves a 3x3 grid of subplots to plots/metrics_vs_alpha.png.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

VIS_DIR   = Path("data/vis")
PLOTS_DIR = Path("data/combined_vis")

PREFIXES = {
    "single_concept":  "Single Concept",
    "two_concepts":    "Two Concepts",
    "three_concepts":  "Three Concepts",
}

FILENAME_RE = re.compile(
    r"^(single_concept|two_concepts|three_concepts)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)\.json$"
)

# (subplot title, metric key, y-label, lower-is-better)
SUBPLOTS = [
    ("Mean IoU @ 1",            "mean_iou_at_1",          "IoU",        False),
    ("Mean IoU @ K",            "mean_iou_at_k",          "IoU",        False),
    ("Mean Displacement @ K",   "mean_disp_at_k",         "Seconds",    True),
    ("Recall@K  IoU≥0.3",       "recall_at_k_iou03",      "Recall",     False),
    ("Recall@K  IoU≥0.5",       "recall_at_k_iou05",      "Recall",     False),
    ("Recall@K  IoU≥0.7",       "recall_at_k_iou07",      "Recall",     False),
    ("Precision@K  IoU≥0.3",    "precision_at_k_iou03",   "Precision",  False),
    ("Precision@K  IoU≥0.5",    "precision_at_k_iou05",   "Precision",  False),
    ("Precision@K  IoU≥0.7",    "precision_at_k_iou07",   "Precision",  False),
]

COLORS = {
    "single_concept": "#1f77b4",
    "two_concepts":   "#ff7f0e",
    "three_concepts": "#2ca02c",
}
MARKERS = {
    "single_concept": "o",
    "two_concepts":   "s",
    "three_concepts": "^",
}


def load_data() -> dict[str, dict[float, dict]]:
    """Return {prefix: {alpha: aggregate_dict}}."""
    data: dict[str, dict[float, dict]] = defaultdict(dict)
    for path in sorted(VIS_DIR.glob("*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        prefix, alpha_s = m.group(1), m.group(2)
        alpha = float(alpha_s)
        file_data = json.loads(path.read_text())
        data[prefix][alpha] = file_data["aggregate"]
    return data


def main() -> None:
    data = load_data()
    if not data:
        raise SystemExit(f"No matching files found in {VIS_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Metrics vs Alpha  (beta = 1 − alpha)", fontsize=14, fontweight="bold")

    for ax, (title, key, ylabel, lower_better) in zip(axes.flat, SUBPLOTS):
        for prefix, label in PREFIXES.items():
            if prefix not in data:
                continue
            sorted_items = sorted(data[prefix].items())  # sort by alpha
            alphas = [a for a, _ in sorted_items]
            values = [agg[key] for _, agg in sorted_items]
            ax.plot(
                alphas, values,
                label=label,
                color=COLORS[prefix],
                marker=MARKERS[prefix],
                linewidth=1.8,
                markersize=5,
            )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Alpha", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([i / 10 for i in range(11)])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

        if lower_better:
            ax.annotate("↓ lower is better", xy=(0.98, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=7, color="gray")

    # Shared legend below the figure
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = PLOTS_DIR / "metrics_vs_alpha.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

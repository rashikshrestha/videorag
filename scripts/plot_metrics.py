#!/usr/bin/env python
"""
scripts/plot_metrics.py
~~~~~~~~~~~~~~~~~~~~~~~
Produces two sets of plots saved to data/combined_vis/:

  metrics_vs_alpha.png   — line plots of how each metric changes with alpha
  aggregate_metrics.png  — bar charts comparing average vs best across concept types
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

VIS_DIR      = Path("data/vis")
COMBINED_DIR = Path("data/combined_vis")

PREFIX_LABELS = {
    "single_concept":  "Single Concept",
    "two_concepts":    "Two Concepts",
    "three_concepts":  "Three Concepts",
}
PREFIX_ORDER = ["single_concept", "two_concepts", "three_concepts"]

FILENAME_RE = re.compile(
    r"^(single_concept|two_concepts|three_concepts)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)\.json$"
)

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

# (title, metric key in per-alpha aggregate, y-label, lower-is-better)
LINE_SUBPLOTS = [
    ("Mean IoU @ 1",           "mean_iou_at_1",         "IoU",       False),
    ("Mean IoU @ K",           "mean_iou_at_k",         "IoU",       False),
    ("Mean Displacement @ K",  "mean_disp_at_k",        "Seconds",   True),
    ("Recall@K  IoU≥0.3",      "recall_at_k_iou03",     "Recall",    False),
    ("Recall@K  IoU≥0.5",      "recall_at_k_iou05",     "Recall",    False),
    ("Recall@K  IoU≥0.7",      "recall_at_k_iou07",     "Recall",    False),
    ("Precision@K  IoU≥0.3",   "precision_at_k_iou03",  "Precision", False),
    ("Precision@K  IoU≥0.5",   "precision_at_k_iou05",  "Precision", False),
    ("Precision@K  IoU≥0.7",   "precision_at_k_iou07",  "Precision", False),
]

# (title, metric key in combined aggregate, y-label, lower-is-better)
BAR_METRICS = [
    ("Mean IoU @ 1",          "mean_iou_at_1",           "IoU",       False),
    ("Mean IoU @ K",          "mean_iou_at_k",           "IoU",       False),
    ("Mean Displacement @ K", "mean_disp_at_k",          "Seconds",   True),
    ("Precision@K  IoU≥0.3",  "mean_prec_at_k_iou03",    "Precision", False),
    ("Precision@K  IoU≥0.5",  "mean_prec_at_k_iou05",    "Precision", False),
    ("Precision@K  IoU≥0.7",  "mean_prec_at_k_iou07",    "Precision", False),
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_alpha_data() -> dict[str, dict[float, dict]]:
    """Return {prefix: {alpha: aggregate_dict}} from per-run vis files."""
    data: dict[str, dict[float, dict]] = defaultdict(dict)
    for path in sorted(VIS_DIR.glob("*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        prefix, alpha_s = m.group(1), m.group(2)
        file_data = json.loads(path.read_text())
        data[prefix][float(alpha_s)] = file_data["aggregate"]
    return data


def load_combined_data() -> dict[str, dict]:
    """Return {prefix: {average: {...}, best: {...}}} from combined JSON files."""
    data = {}
    for prefix in PREFIX_ORDER:
        path = COMBINED_DIR / f"{prefix}_combined.json"
        if path.exists():
            d = json.loads(path.read_text())
            data[prefix] = d["aggregate"]
    return data


# ---------------------------------------------------------------------------
# Plot 1: metrics vs alpha (line plots)
# ---------------------------------------------------------------------------

def plot_metrics_vs_alpha(data: dict[str, dict[float, dict]]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Metrics vs Alpha  (beta = 1 − alpha)", fontsize=14, fontweight="bold")

    for ax, (title, key, ylabel, lower_better) in zip(axes.flat, LINE_SUBPLOTS):
        for prefix, label in PREFIX_LABELS.items():
            if prefix not in data:
                continue
            sorted_items = sorted(data[prefix].items())
            alphas = [a for a, _ in sorted_items]
            values = [agg[key] for _, agg in sorted_items]
            ax.plot(alphas, values, label=label, color=COLORS[prefix],
                    marker=MARKERS[prefix], linewidth=1.8, markersize=5)

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

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out = COMBINED_DIR / "metrics_vs_alpha.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: aggregate bar charts (average vs best per concept type)
# ---------------------------------------------------------------------------

def plot_aggregate_metrics(combined: dict[str, dict]) -> None:
    n_metrics = len(BAR_METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    fig.suptitle("Aggregate Metrics: Average vs Best across Alpha Configs",
                 fontsize=14, fontweight="bold")

    x      = np.arange(len(PREFIX_ORDER))
    width  = 0.35
    labels = [PREFIX_LABELS[p] for p in PREFIX_ORDER]

    for idx, (title, key, ylabel, lower_better) in enumerate(BAR_METRICS):
        ax = axes.flat[idx]

        avg_vals  = [combined[p]["average"][key] for p in PREFIX_ORDER if p in combined]
        best_vals = [combined[p]["best"][key]    for p in PREFIX_ORDER if p in combined]

        bars_avg  = ax.bar(x - width / 2, avg_vals,  width, label="Average",
                           color=[COLORS[p] for p in PREFIX_ORDER], alpha=0.5)
        bars_best = ax.bar(x + width / 2, best_vals, width, label="Best",
                           color=[COLORS[p] for p in PREFIX_ORDER], alpha=1.0)

        # Value labels on bars
        for bar in bars_avg:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars_best:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(0, max(max(avg_vals + best_vals) * 1.30, 0.01))

        if lower_better:
            ax.annotate("↓ lower is better", xy=(0.98, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=7, color="gray")

    # Hide unused subplots
    for idx in range(n_metrics, nrows * ncols):
        axes.flat[idx].set_visible(False)

    # Shared legend for Average / Best
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.5, label="Average (mean over all alpha configs)"),
        Patch(facecolor="gray", alpha=1.0, label="Best (best alpha config per query)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out = COMBINED_DIR / "aggregate_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: improvement gap (best - average) per concept type
# ---------------------------------------------------------------------------

def plot_improvement_gap(combined: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Improvement Gap: Best − Average  (how much tuning alpha helps)",
                 fontsize=13, fontweight="bold")

    for ax, prefix in zip(axes, PREFIX_ORDER):
        if prefix not in combined:
            ax.set_visible(False)
            continue

        agg    = combined[prefix]
        titles = [t for t, _, _, _ in BAR_METRICS]
        keys   = [k for _, k, _, lb in BAR_METRICS]
        lower  = [lb for _, _, _, lb in BAR_METRICS]

        gaps = []
        for key, lb in zip(keys, lower):
            avg_v  = agg["average"][key]
            best_v = agg["best"][key]
            # Positive gap = improvement
            gap = (avg_v - best_v) if lb else (best_v - avg_v)
            gaps.append(gap)

        colors = ["#d62728" if g < 0 else "#2ca02c" for g in gaps]
        y_pos  = np.arange(len(titles))
        ax.barh(y_pos, gaps, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(PREFIX_LABELS[prefix], fontsize=10)
        ax.set_xlabel("Gap (Best − Average)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        for i, (g, bar_y) in enumerate(zip(gaps, y_pos)):
            ax.text(g + (0.001 if g >= 0 else -0.001), bar_y,
                    f"{g:+.3f}", va="center",
                    ha="left" if g >= 0 else "right", fontsize=7)

    plt.tight_layout()

    out = COMBINED_DIR / "improvement_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    alpha_data   = load_alpha_data()
    combined     = load_combined_data()

    if not alpha_data:
        raise SystemExit(f"No per-run files found in {VIS_DIR}")
    if not combined:
        raise SystemExit(f"No combined files found in {COMBINED_DIR} — run combine_results.py first")

    plot_metrics_vs_alpha(alpha_data)
    plot_aggregate_metrics(combined)
    plot_improvement_gap(combined)


if __name__ == "__main__":
    main()

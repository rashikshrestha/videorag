#!/usr/bin/env python
"""
scripts/eval_results.py
~~~~~~~~~~~~~~~~~~~~~~~
Compute temporal IoU metrics from a query-multi output JSON.

For each query the script matches every retrieved candidate against
every ground-truth segment (same episode required) and keeps the
highest IoU.  Two flavours are reported:

  R@1  — IoU of the top-scored candidate only
  R@K  — best IoU among the top-K candidates  (set with --top-k)

Aggregate metrics
-----------------
  Mean IoU
  Recall@K / IoU>=0.3 | 0.5 | 0.7   (fraction of queries that hit the threshold)
  Precision@K / IoU>=0.3 | 0.5 | 0.7  (fraction of top-K candidates that hit the threshold)
  Mean Center Displacement  (seconds from predicted center to nearest GT center)

Usage
-----
python scripts/eval_results.py --results data/out/results.json
python scripts/eval_results.py --results data/out/results.json --top-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def temporal_iou(pred_start: float, pred_end: float,
                 gt_start: float,   gt_end: float) -> float:
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    if inter == 0.0:
        return 0.0
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


def best_iou_for_candidate(candidate: dict, ground_truth: list[dict]) -> float:
    """Highest IoU between one candidate and all GT segments on the same episode."""
    best = 0.0
    for gt in ground_truth:
        if candidate["episode"] != gt["episode"]:
            continue
        iou = temporal_iou(
            candidate["global_start"], candidate["global_end"],
            gt["gt_start"],            gt["gt_end"],
        )
        best = max(best, iou)
    return best


def center_displacement_for_candidate(candidate: dict, ground_truth: list[dict]) -> float:
    """Minimum distance between candidate center and any GT center on the same episode."""
    pred_center = (candidate["global_start"] + candidate["global_end"]) / 2
    best = float("inf")
    for gt in ground_truth:
        if candidate["episode"] != gt["episode"]:
            continue
        gt_center = (gt["gt_start"] + gt["gt_end"]) / 2
        best = min(best, abs(pred_center - gt_center))
    return best if best != float("inf") else 0.0


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(entry: dict, top_k: int) -> dict:
    gt        = entry["ground_truth"]
    cands     = entry["retrieved_candidates"]
    top_k_cands = cands[:top_k]

    # R@1: top-scored candidate (index 0, already sorted by score descending)
    iou_at_1  = best_iou_for_candidate(cands[0], gt) if cands else 0.0
    disp_at_1 = center_displacement_for_candidate(cands[0], gt) if cands else 0.0

    # R@K: best IoU / min displacement among top-K candidates
    iou_at_k  = max((best_iou_for_candidate(c, gt) for c in top_k_cands), default=0.0)
    disp_at_k = min((center_displacement_for_candidate(c, gt) for c in top_k_cands), default=0.0)

    # Precision@K per threshold: fraction of top-K candidates with IoU >= threshold
    thresholds = [0.3, 0.5, 0.7]
    prec_at_k = {}
    for thr in thresholds:
        hits = sum(1 for c in top_k_cands if best_iou_for_candidate(c, gt) >= thr)
        prec_at_k[thr] = round(hits / len(top_k_cands), 4) if top_k_cands else 0.0

    return {
        "query":      entry["query"],
        "iou_at_1":   round(iou_at_1, 4),
        "iou_at_k":   round(iou_at_k, 4),
        "disp_at_1":  round(disp_at_1, 2),
        "disp_at_k":  round(disp_at_k, 2),
        "prec_at_k":  prec_at_k,
        "n_gt":       len(gt),
        "n_cands":    len(cands),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict], top_k: int) -> dict:
    thresholds = [0.3, 0.5, 0.7]
    n = len(rows)

    mean_iou_1  = sum(r["iou_at_1"]  for r in rows) / n
    mean_iou_k  = sum(r["iou_at_k"]  for r in rows) / n
    mean_disp_1 = sum(r["disp_at_1"] for r in rows) / n
    mean_disp_k = sum(r["disp_at_k"] for r in rows) / n

    metrics: dict = {
        "n_queries":       n,
        "top_k":           top_k,
        "mean_iou_at_1":   round(mean_iou_1,  4),
        "mean_iou_at_k":   round(mean_iou_k,  4),
        "mean_disp_at_1":  round(mean_disp_1, 2),
        "mean_disp_at_k":  round(mean_disp_k, 2),
    }

    for thr in thresholds:
        recall_1 = sum(r["iou_at_1"] >= thr for r in rows) / n
        recall_k = sum(r["iou_at_k"] >= thr for r in rows) / n
        mean_prec = sum(r["prec_at_k"][thr] for r in rows) / n
        key = str(thr).replace(".", "")
        metrics[f"recall_at_1_iou{key}"] = round(recall_1,  4)
        metrics[f"recall_at_k_iou{key}"] = round(recall_k,  4)
        metrics[f"precision_at_k_iou{key}"] = round(mean_prec, 4)

    print(f"\n{'='*60}")
    print(f"  Evaluation  —  {n} queries,  top-K = {top_k}")
    print(f"{'='*60}")
    print(f"  {'Metric':<35} {'R@1':>8}  {'R@'+str(top_k):>8}")
    print(f"  {'-'*51}")
    print(f"  {'Mean IoU':<35} {mean_iou_1:>8.4f}  {mean_iou_k:>8.4f}")
    for thr in thresholds:
        key = str(thr).replace(".", "")
        print(f"  {'Recall  IoU>='+str(thr):<35} {metrics['recall_at_1_iou'+key]:>8.4f}  {metrics['recall_at_k_iou'+key]:>8.4f}")

    print(f"  {'-'*51}")
    print(f"  {'Mean Center Displacement (s)':<35} {mean_disp_1:>8.2f}  {mean_disp_k:>8.2f}")

    print(f"  {'-'*51}")
    for thr in thresholds:
        key = str(thr).replace(".", "")
        print(f"  {'Precision@'+str(top_k)+'  IoU>='+str(thr):<35} {'N/A':>8}  {metrics['precision_at_k_iou'+key]:>8.4f}")

    print(f"{'='*60}\n")
    return metrics


# ---------------------------------------------------------------------------
# Per-query table
# ---------------------------------------------------------------------------

def print_per_query(rows: list[dict], top_k: int) -> None:
    col_w = max(len(r["query"]) for r in rows)
    header = (
        f"  {'Query':<{col_w}}  {'IoU@1':>7}  {'IoU@'+str(top_k):>7}"
        f"  {'Disp@1':>7}  {'Disp@'+str(top_k):>7}  GT  Cands"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        flag = "  <-- no match" if r["iou_at_k"] == 0.0 else ""
        print(
            f"  {r['query']:<{col_w}}  {r['iou_at_1']:>7.4f}  {r['iou_at_k']:>7.4f}"
            f"  {r['disp_at_1']:>7.2f}  {r['disp_at_k']:>7.2f}"
            f"  {r['n_gt']:>2}  {r['n_cands']:>5}{flag}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal IoU from query-multi output JSON."
    )
    parser.add_argument(
        "--results",
        required=True,
        metavar="PATH",
        help="Path to the JSON produced by query-multi",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of top candidates to consider for R@K  (default: 5)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"ERROR: file not found: {results_path}")

    with results_path.open() as f:
        data = json.load(f)

    rows = [evaluate_query(entry, args.top_k) for entry in data]

    print_per_query(rows, args.top_k)
    metrics = aggregate(rows, args.top_k)

    save_path = Path("data/vis") / results_path.name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"aggregate": metrics, "queries": rows}
    with save_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()

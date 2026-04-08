"""
videorag.evaluation
~~~~~~~~~~~~~~~~~~~
Temporal IoU metrics and evaluation harness against annotated gold queries.

Public API
----------
GOLD_QUERIES        : pd.DataFrame — 17 manually annotated ground-truth spans
iou(ps, pe, gs, ge) -> float
evaluate_grounding(gold_df, ctx, top_k, merge_gap) -> pd.DataFrame
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from videorag.pipeline.pipeline import VideoRAGContext, ground
from videorag.utils.time import fmt_time


# ---------------------------------------------------------------------------
# Ground-truth annotations
# ---------------------------------------------------------------------------

GOLD_QUERIES = pd.DataFrame(
    [
        {
            "query": "friends discussing kissing in coffee shop",
            "video": "friends_s01e02.mkv",
            "gt_start": 2.0,
            "gt_end": 25.0,
            "type": "dialogue",
        },
        {
            "query": "Monica brings her date Alan to the apartment",
            "video": "friends_s01e03.mkv",
            "gt_start": 528.0,
            "gt_end": 550.0,
            "type": "mixed",
        },
        {
            "query": "Joey talks to Angela in the coffee shop",
            "video": "friends_s01e05.mkv",
            "gt_start": 245.0,
            "gt_end": 263.0,
            "type": "dialogue",
        },
        {
            "query": "Chandler breaks up with Janice at Central Perk",
            "video": "friends_s01e05.mkv",
            "gt_start": 939.0,
            "gt_end": 965.0,
            "type": "dialogue",
        },
        {
            "query": "Chandler starts smoking",
            "video": "friends_s01e03.mkv",
            "gt_start": 862.0,
            "gt_end": 875.0,
            "type": "action",
        },
        {
            "query": "Chandler, Ross and Joey watching ice hockey",
            "video": "friends_s01e04.mkv",
            "gt_start": 736.0,
            "gt_end": 760.0,
            "type": "action",
        },
        # ── Episode 2 ──────────────────────────────────────────────────────
        {
            "query": "Ross tells friends Carol is pregnant and mentions Susan sadly",
            "video": "friends_s01e02.mkv",
            "gt_start": 415.0,
            "gt_end": 481.0,
            "type": "dialogue",
        },
        {
            "query": "Monica cooking for her parents and discussing restaurant work",
            "video": "friends_s01e02.mkv",
            "gt_start": 504.0,
            "gt_end": 550.0,
            "type": "mixed",
        },
        {
            "query": "Ross giving advice to Rachel in apartment",
            "video": "friends_s01e02.mkv",
            "gt_start": 774.0,
            "gt_end": 819.0,
            "type": "dialogue",
        },
        {
            "query": "Ross visits Carol in hospital with Susan present",
            "video": "friends_s01e02.mkv",
            "gt_start": 875.0,
            "gt_end": 1042.0,
            "type": "mixed",
        },
        # ── Episode 3 ──────────────────────────────────────────────────────
        {
            "query": "Chandler helps Joey rehearse dialogue including smoking",
            "video": "friends_s01e03.mkv",
            "gt_start": 136.0,
            "gt_end": 198.0,
            "type": "dialogue",
        },
        {
            "query": "Chandler asks cigarette from Joey and enjoys smoking",
            "video": "friends_s01e03.mkv",
            "gt_start": 199.0,
            "gt_end": 220.0,
            "type": "action",
        },
        {
            "query": "Phoebe walking in traffic on the road",
            "video": "friends_s01e03.mkv",
            "gt_start": 713.0,
            "gt_end": 718.0,
            "type": "action",
        },
        {
            "query": "Chandler smoking at office and trying to hide smell",
            "video": "friends_s01e03.mkv",
            "gt_start": 770.0,
            "gt_end": 820.0,
            "type": "mixed",
        },
        {
            "query": "Monica tells friends she is breaking up with Alan",
            "video": "friends_s01e03.mkv",
            "gt_start": 1185.0,
            "gt_end": 1260.0,
            "type": "dialogue",
        },
        # ── Episode 4 ──────────────────────────────────────────────────────
        {
            "query": "Rachel meets old friends at coffee shop and they scream",
            "video": "friends_s01e04.mkv",
            "gt_start": 328.0,
            "gt_end": 375.0,
            "type": "action",
        },
        {
            "query": "Ross wearing nose mask after hospital accident",
            "video": "friends_s01e04.mkv",
            "gt_start": 1218.0,
            "gt_end": 1304.0,
            "type": "mixed",
        },
    ]
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou(ps: float, pe: float, gs: float, ge: float) -> float:
    """
    Temporal intersection-over-union between ``[ps, pe]`` and ``[gs, ge]``.

    Returns a value in ``[0, 1]``.  A minimum union of 1e-8 prevents
    division-by-zero for degenerate zero-length spans.
    """
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(1e-8, max(pe, ge) - min(ps, gs))
    return inter / union


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def evaluate_grounding(
    gold_df: pd.DataFrame,
    ctx: VideoRAGContext,
    top_k: int = 10,
    merge_gap: float = 20.0,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the grounding pipeline on every row in *gold_df* and produce
    Top-1 / Top-K temporal IoU metrics.

    Metrics reported
    ----------------
    * **Top-1 Video Accuracy**   — correct video at rank 1
    * **Top-K Video Recall**     — correct video in top-K
    * **Mean IoU (Top-1)**       — average temporal IoU at rank 1
    * **R@1, IoU≥0.3 / ≥0.5**  — recall at IoU thresholds (rank 1)
    * **Mean Best IoU (Top-K)** — best IoU across top-K candidates
    * **Per-type breakdown**     — dialogue / action / mixed

    Args:
        gold_df:   DataFrame with columns: query, video, gt_start, gt_end, type.
        ctx:       Runtime :class:`~videorag.pipeline.pipeline.VideoRAGContext`.
        top_k:     Retrieval candidates per query.
        merge_gap: Span merging gap (seconds).
        save_dir:  If given, save ``evaluation_final.csv`` to this directory.

    Returns:
        Per-query evaluation DataFrame.
    """
    eval_rows = []
    t0 = time.time()

    for _, g in gold_df.iterrows():
        out  = ground(g["query"], ctx, top_k=top_k, merge_gap=merge_gap)
        top1 = out.iloc[0]

        t1_ok  = int(top1["video"] == g["video"])
        t1_iou = (
            iou(
                top1["refined_start"], top1["refined_end"],
                g["gt_start"], g["gt_end"],
            )
            if t1_ok
            else 0.0
        )

        # Best in Top-K
        tk_hit      = 0
        best_iou_k  = 0.0
        best_pred_k = ""
        for _, c in out.iterrows():
            if c["video"] == g["video"]:
                tk_hit = 1
                c_iou  = iou(
                    c["refined_start"], c["refined_end"],
                    g["gt_start"], g["gt_end"],
                )
                if c_iou > best_iou_k:
                    best_iou_k  = c_iou
                    best_pred_k = fmt_time(c["refined_start"]) + " -> " + fmt_time(c["refined_end"])

        eval_rows.append(
            {
                "query":        g["query"][:55],
                "type":         g.get("type", top1["query_type"]),
                "gt_video":     g["video"],
                "gt":           fmt_time(g["gt_start"]) + " -> " + fmt_time(g["gt_end"]),
                "gt_span":      g["gt_end"] - g["gt_start"],
                "pred":         fmt_time(top1["refined_start"]) + " -> " + fmt_time(top1["refined_end"]),
                "pred_span":    top1["span_seconds"],
                "top1_video":   top1["video"],
                "top1_correct": t1_ok,
                "top1_iou":     round(t1_iou, 4),
                "topk_hit":     tk_hit,
                "best_iou_k":   round(best_iou_k, 4),
                "best_pred_k":  best_pred_k or "N/A",
                "confidence":   top1["confidence"],
            }
        )

    elapsed = time.time() - t0
    edf = pd.DataFrame(eval_rows)
    n   = len(edf)

    print("\n" + "=" * 90)
    print("EVALUATION RESULTS")
    print("=" * 90)
    print(f"  Queries:                   {n}")
    print(f"  Time:                      {elapsed:.1f}s ({elapsed/n:.1f}s/query)")
    print()
    print(f"  Top-1 Video Accuracy:      {edf['top1_correct'].mean()*100:.0f}%")
    print(f"  Top-{top_k} Video Recall:      {edf['topk_hit'].mean()*100:.0f}%")
    print()

    miou1 = edf["top1_iou"].mean()
    r1_03 = (edf["top1_iou"] >= 0.3).sum()
    r1_05 = (edf["top1_iou"] >= 0.5).sum()
    print("  -- Top-1 Metrics --")
    print(f"  Mean IoU:                  {miou1:.4f}")
    print(f"  R@1, IoU>=0.3:             {r1_03/n*100:.0f}%  ({r1_03}/{n})")
    print(f"  R@1, IoU>=0.5:             {r1_05/n*100:.0f}%  ({r1_05}/{n})")
    print()

    miouk = edf["best_iou_k"].mean()
    rk_03 = (edf["best_iou_k"] >= 0.3).sum()
    rk_05 = (edf["best_iou_k"] >= 0.5).sum()
    print(f"  -- Best-in-Top-{top_k} Metrics --")
    print(f"  Mean Best IoU:             {miouk:.4f}")
    print(f"  R@{top_k}, IoU>=0.3:            {rk_03/n*100:.0f}%  ({rk_03}/{n})")
    print(f"  R@{top_k}, IoU>=0.5:            {rk_05/n*100:.0f}%  ({rk_05}/{n})")

    print("\n" + "-" * 90)
    print("PER-QUERY DETAILS")
    print("-" * 90)
    for _, r in edf.iterrows():
        status = "  OK " if r["top1_iou"] >= 0.3 else ("  ~~ " if r["top1_iou"] > 0 else "  XX ")
        vid_ok = "Y" if r["top1_correct"] else f"N (got {r['top1_video']})"
        print(f"{status} IoU={r['top1_iou']:.3f}  video={vid_ok}")
        print(f"     Q:    {r['query']}")
        print(f"     GT:   {r['gt']}    PRED: {r['pred']}")
        if r["best_iou_k"] > r["top1_iou"] + 0.01:
            print(f"     ^ Better in top-{top_k}: IoU={r['best_iou_k']:.3f} at {r['best_pred_k']}")
        print()

    print("-" * 90)
    print("BY QUERY TYPE")
    header = f"{'Type':<12} {'N':>3} {'mIoU':>8} {'R@1,0.3':>8} {'R@1,0.5':>8} {'BestmIoU':>10}"
    print(header)
    print("-" * 55)
    for qtype in ["dialogue", "action", "mixed"]:
        sub = edf[edf["type"] == qtype]
        if len(sub):
            print(
                f"{qtype:<12} {len(sub):>3} "
                f"{sub['top1_iou'].mean():>8.4f} "
                f"{(sub['top1_iou']>=0.3).mean()*100:>7.0f}% "
                f"{(sub['top1_iou']>=0.5).mean()*100:>7.0f}% "
                f"{sub['best_iou_k'].mean():>10.4f}"
            )
    print(
        f"{'ALL':<12} {n:>3} "
        f"{edf['top1_iou'].mean():>8.4f} "
        f"{(edf['top1_iou']>=0.3).mean()*100:>7.0f}% "
        f"{(edf['top1_iou']>=0.5).mean()*100:>7.0f}% "
        f"{edf['best_iou_k'].mean():>10.4f}"
    )

    if save_dir:
        from pathlib import Path
        save_path = Path(save_dir) / "evaluation_final.csv"
        edf.to_csv(save_path, index=False)
        print(f"\n✅ Saved → {save_path}")

    return edf

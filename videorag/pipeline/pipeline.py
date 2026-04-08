"""
videorag.pipeline
~~~~~~~~~~~~~~~~~
End-to-end grounding pipeline: retrieval → refinement → merge → calibration.

Public API
----------
fmt(sec)                        -> str  ("MM:SS.ss")
calibrate(raw, floor, ceiling)  -> float
merge_spans(results_df, gap)    -> pd.DataFrame
ground(query, ctx, top_k, merge_gap) -> pd.DataFrame
run(query, ctx, top_k, show_top, merge_gap) -> pd.DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from videorag.config import Settings
from videorag.models.embeddings import ModelBundle
from videorag.retrieval.query import _kw_overlap
from videorag.retrieval.refinement import refine
from videorag.retrieval.search import hybrid_search


# ---------------------------------------------------------------------------
# Context dataclass — bundles all shared state
# ---------------------------------------------------------------------------

@dataclass
class VideoRAGContext:
    """
    Containers for the shared runtime objects.

    Create via :func:`videorag.api.build_context` rather than
    constructing directly.

    Attributes:
        settings:         Project :class:`~videorag.config.Settings`.
        bundle:           Loaded :class:`~videorag.embeddings.ModelBundle`.
        segments_df:      Full segments DataFrame (from preprocessing).
        text_index:       FAISS index over text embeddings.
        image_index:      FAISS index over image embeddings.
        text_embeddings:  Raw text embedding matrix  (N, 384).
        image_embeddings: Raw image embedding matrix (N, 512).
    """
    settings: Settings
    bundle: ModelBundle
    segments_df: pd.DataFrame
    text_index: faiss.IndexFlatIP
    image_index: faiss.IndexFlatIP
    text_embeddings: np.ndarray
    image_embeddings: np.ndarray


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def fmt(sec: float) -> str:
    """Format *sec* seconds as ``"MM:SS.ss"``."""
    m, s = divmod(float(sec), 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def calibrate(raw: float, floor: float, ceiling: float) -> float:
    """
    Map a raw combined score to a ``[0, 1]`` confidence value.

    BUG FIX #5: the original notebook had no calibration, so combined
    scores in the range ~0.4-0.8 all mapped to 100 %.  Now scores
    below *floor* → 0.0, above *ceiling* → 1.0, otherwise linearly
    interpolated.

    Args:
        raw:     Combined pipeline score (typically 0.0–1.0).
        floor:   Score that maps to 0.0.
        ceiling: Score that maps to 1.0.

    Returns:
        Calibrated confidence in ``[0, 1]``.
    """
    return float(np.clip((raw - floor) / (ceiling - floor), 0.0, 1.0))


def _character_match_score(
    query: str,
    subtitle_text: str,
    characters: list,
) -> float:
    """
    Fraction of query-mentioned characters that appear in *subtitle_text*.

    Returns 0.0 when the query mentions no known character names.
    """
    q_lower = query.lower()
    s_lower = str(subtitle_text).lower()
    chars_in_query = [c for c in characters if c in q_lower]
    if not chars_in_query:
        return 0.0
    hits = sum(1 for c in chars_in_query if c in s_lower)
    return hits / len(chars_in_query)


# ---------------------------------------------------------------------------
# Span merging (BUG FIX #4)
# ---------------------------------------------------------------------------

def merge_spans(
    results_df: pd.DataFrame,
    merge_gap: float = 20.0,
) -> pd.DataFrame:
    """
    Merge adjacent/overlapping refined spans from the same video.

    BUG FIX #4: without merging, a query like *'friends discussing
    kissing'* that spans 1–100 s returns fragments ``1-20``, ``20-40``,
    …  ``merge_spans`` groups same-video spans whose gap is
    ≤ ``merge_gap`` seconds and collapses them into the union
    ``[min_start, max_end]``, keeping the highest-scoring row's metadata.

    A merge gap of 20 s is chosen because two scenes about the same
    topic rarely have more than 20 s of unrelated content between them.

    Args:
        results_df: DataFrame with columns ``video``, ``refined_start``,
                    ``refined_end``, ``raw_score``.
        merge_gap:  Maximum gap (seconds) to merge across.

    Returns:
        New DataFrame with merged spans, still sorted by ``raw_score``
        descending.
    """
    if len(results_df) == 0:
        return results_df

    merged_rows = []
    for video, group in results_df.groupby("video", sort=False):
        group = group.sort_values("refined_start").reset_index(drop=True)

        cur_start = float(group.loc[0, "refined_start"])
        cur_end   = float(group.loc[0, "refined_end"])
        best_row  = group.loc[0].copy()

        for idx in range(1, len(group)):
            row = group.loc[idx]
            r_s  = float(row["refined_start"])
            r_e  = float(row["refined_end"])
            gap  = r_s - cur_end  # negative if overlapping

            if gap <= merge_gap:
                cur_end = max(cur_end, r_e)
                if float(row["raw_score"]) > float(best_row["raw_score"]):
                    best_row = row.copy()
            else:
                best_row = best_row.copy()
                best_row["refined_start"] = cur_start
                best_row["refined_end"]   = cur_end
                best_row["span_seconds"]  = cur_end - cur_start
                merged_rows.append(best_row)
                cur_start = r_s
                cur_end   = r_e
                best_row  = row.copy()

        # Emit last cluster
        best_row = best_row.copy()
        best_row["refined_start"] = cur_start
        best_row["refined_end"]   = cur_end
        best_row["span_seconds"]  = cur_end - cur_start
        merged_rows.append(best_row)

    out = pd.DataFrame(merged_rows)
    return out.sort_values("raw_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full grounding pipeline
# ---------------------------------------------------------------------------

def ground(
    query: str,
    ctx: VideoRAGContext,
    top_k: Optional[int] = None,
    merge_gap: Optional[float] = None,
) -> pd.DataFrame:
    """
    Retrieve candidate scenes, refine each to a fine-grained span, fuse
    scores, calibrate confidence and optionally merge adjacent spans.

    Score fusion formula::

        raw = 0.50 × hybrid_score + 0.40 × grounding_conf + 0.10 × keyword_bonus

    Args:
        query:     Natural-language search query.
        ctx:       Runtime :class:`VideoRAGContext`.
        top_k:     Number of retrieval candidates (default: ``settings.retrieval.top_k``).
        merge_gap: Span merging gap (seconds).  ``0`` disables merging.
                   Default: ``settings.retrieval.merge_gap``.

    Returns:
        DataFrame sorted by ``raw_score`` descending.  Columns include
        ``video``, ``scene_id``, ``query_type``, ``refined_start``,
        ``refined_end``, ``span_seconds``, ``confidence``, ``subtitle``.
    """
    settings = ctx.settings
    if top_k is None:
        top_k = settings.retrieval.top_k
    if merge_gap is None:
        merge_gap = settings.retrieval.merge_gap

    retrieved = hybrid_search(
        query,
        ctx.segments_df,
        ctx.text_index,
        ctx.image_index,
        ctx.bundle,
        settings,
        top_k=top_k,
    )

    pl = settings.pipeline
    results = []
    for _, row in retrieved.iterrows():
        r_start, r_end, g_conf = refine(
            row,
            query,
            ctx.bundle,
            settings,
            settings.paths.video_root,
            settings.paths.subtitle_root,
        )

        kw_hits, kw_r = _kw_overlap(query, row["subtitle"])
        kw_bonus = min(0.05, 0.03 * kw_r + 0.005 * kw_hits)

        raw = (
            pl.hybrid_weight    * float(row["hybrid_score"])
            + pl.grounding_weight * float(g_conf)
            + pl.keyword_weight   * float(kw_bonus)
        )

        results.append(
            {
                "video":           row["video"],
                "scene_id":        int(row["scene_id"]),
                "query_type":      row["query_type"],
                "scene_start":     float(row["start"]),
                "scene_end":       float(row["end"]),
                "refined_start":   float(r_start),
                "refined_end":     float(r_end),
                "span_seconds":    float(r_end - r_start),
                "retrieval_score": float(row["hybrid_score"]),
                "grounding_conf":  float(g_conf),
                "keyword_bonus":   float(kw_bonus),
                "raw_score":       float(raw),
                "confidence":      calibrate(raw, pl.calibrate_floor, pl.calibrate_ceiling),
                "subtitle":        row["subtitle"],
            }
        )

    out = pd.DataFrame(results).sort_values(
        ["raw_score", "grounding_conf", "retrieval_score"],
        ascending=False,
    ).reset_index(drop=True)

    if merge_gap > 0:
        out = merge_spans(out, merge_gap=merge_gap)

    return out


# ---------------------------------------------------------------------------
# Human-readable wrapper
# ---------------------------------------------------------------------------

def run(
    query: str,
    ctx: VideoRAGContext,
    top_k: int = 5,
    show_top: int = 3,
    merge_gap: float = 20.0,
) -> pd.DataFrame:
    """
    Pretty-print the top grounding results and return the full DataFrame.

    Args:
        query:     Natural-language search query.
        ctx:       Runtime :class:`VideoRAGContext`.
        top_k:     Number of retrieval candidates.
        show_top:  Number of results to print.
        merge_gap: Span merging gap (seconds).

    Returns:
        Full grounding DataFrame (same as :func:`ground`).
    """
    out = ground(query, ctx, top_k=top_k, merge_gap=merge_gap)
    _AGG = {"action": "max-pool", "dialogue": "mean-pool", "mixed": "mean-pool"}

    print("=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)
    for i, (_, top) in enumerate(out.head(show_top).iterrows(), 1):
        qt  = top["query_type"]
        agg = _AGG.get(qt, "")
        print(
            f"[{i}] {top['video']}  scene={int(top['scene_id'])}  "
            f"type={qt} ({agg})"
        )
        print(
            f"    SCENE:    {fmt(top['scene_start'])} → {fmt(top['scene_end'])}"
        )
        print(
            f"    GROUNDED: {fmt(top['refined_start'])} → {fmt(top['refined_end'])}  "
            f"(span={top['span_seconds']:.1f}s)"
        )
        print(
            f"    CONF:     {top['confidence']:.2%}  "
            f"(ret={top['retrieval_score']:.3f} "
            f"gnd={top['grounding_conf']:.3f} "
            f"kw={top['keyword_bonus']:.3f})"
        )
        print(f"    SUB:      {str(top['subtitle'])[:200]}")
        print("-" * 100)
    return out

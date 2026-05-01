"""
videorag.retrieval.search
~~~~~~~~~~~~~~~~~~
Dual-index FAISS retrieval with adaptive fusion and character boosting.

Public API
----------
hybrid_search(query, segments_df, text_index, image_index, bundle,
              settings, top_k=10) -> pd.DataFrame
"""
from __future__ import annotations

from typing import Optional

import faiss
import numpy as np
import pandas as pd

from videorag.config import Settings
from videorag.models.embeddings import ModelBundle
from videorag.retrieval.query import (
    _kw_overlap,
    _safe_minmax,
    classify_query,
    embed_query_audio,
    embed_query_clip,
    embed_query_text,
)


def hybrid_search(
    query: str,
    segments_df: pd.DataFrame,
    text_index: Optional[faiss.IndexFlatIP],
    image_index: Optional[faiss.IndexFlatIP],
    audio_index: Optional[faiss.IndexFlatIP],
    bundle: ModelBundle,
    settings: Settings,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Retrieve the top-*k* candidate scenes for *query* using dual-index
    score fusion with character-name boosting.

    Pipeline
    --------
    1. Classify query → (label, α, β)
    2. Encode query with both text and CLIP encoders
    3. Search both FAISS indices over the full corpus
    4. Min-max normalise per-index scores → same scale
    5. Fuse: ``score = α * text_norm + β * image_norm``
    6. Apply character boost: ``+0.3 * (char_hits / query_chars)``
       for each segment whose subtitle mentions query-referenced
       characters (applied *before* ranking so signal reaches top-k)
    7. Return top-*k* rows as a ranked DataFrame

    BUG FIX #8: α/β are query-adaptive (action → high β; dialogue → high α)
    rather than the original fixed 0.7/0.3.

    Args:
        query:        Natural-language search string.
        segments_df:  Full segments DataFrame (output of preprocessing).
        text_index:   FAISS index over text embeddings.
        image_index:  FAISS index over image embeddings.
        bundle:       Loaded :class:`~videorag.models.embeddings.ModelBundle`.
        settings:     Project settings.
        top_k:        Number of candidates to return.

    Returns:
        DataFrame with columns: rank, video, scene_id, start, end, subtitle,
        frames, hybrid_score, text_score, image_score, query_type.
        Sorted by hybrid_score descending.
    """
    n = len(segments_df)
    qtype, alpha, beta, gamma = classify_query(query, settings)

    q_text  = embed_query_text(query, bundle)  if text_index  is not None else None
    q_clip  = embed_query_clip(query, bundle)  if image_index is not None else None
    q_audio = embed_query_audio(query, bundle) if audio_index is not None else None

    # ── Search each active index over the full corpus ──
    raw_t = np.zeros(n, dtype=np.float32)
    if text_index is not None and q_text is not None:
        ts, ti = text_index.search(q_text, n)
        ts_map = dict(zip(ti[0].tolist(), ts[0].tolist()))
        raw_t = np.array([max(0.0, ts_map.get(i, 0.0)) for i in range(n)], dtype=np.float32)

    raw_i = np.zeros(n, dtype=np.float32)
    if image_index is not None and q_clip is not None:
        is_, ii = image_index.search(q_clip, n)
        is_map = dict(zip(ii[0].tolist(), is_[0].tolist()))
        raw_i = np.array([max(0.0, is_map.get(i, 0.0)) for i in range(n)], dtype=np.float32)

    raw_a = np.zeros(n, dtype=np.float32)
    if audio_index is not None and q_audio is not None:
        as_, ai = audio_index.search(q_audio, n)
        as_map = dict(zip(ai[0].tolist(), as_[0].tolist()))
        raw_a = np.array([max(0.0, as_map.get(i, 0.0)) for i in range(n)], dtype=np.float32)

    norm_t = _safe_minmax(raw_t)
    norm_i = _safe_minmax(raw_i)
    norm_a = _safe_minmax(raw_a)

    # Renormalize weights over active modalities only.
    w_t = alpha if text_index  is not None else 0.0
    w_i = beta  if image_index is not None else 0.0
    w_a = gamma if (audio_index is not None and q_audio is not None) else 0.0
    total = w_t + w_i + w_a
    if total < 1e-8:
        total = 1.0
    fused = (w_t / total) * norm_t + (w_i / total) * norm_i + (w_a / total) * norm_a

    # ── Character boost ──
    q_lower = query.lower()
    characters = settings.characters
    query_chars = [c for c in characters if c in q_lower]

    if query_chars:
        boost = settings.retrieval.character_boost
        for i in range(n):
            sub = str(segments_df.iloc[i]["subtitle"]).lower()
            hits = sum(1 for c in query_chars if c in sub)
            if hits > 0:
                fused[i] += boost * (hits / len(query_chars))

    order = np.argsort(fused)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(order, 1):
        r = segments_df.iloc[idx]
        rows.append(
            {
                "rank":         rank,
                "video":        r["video"],
                "scene_id":     int(r["scene_id"]),
                "start":        float(r["start"]),
                "end":          float(r["end"]),
                "subtitle":     r["subtitle"],
                "frames":       r["frames"],
                "hybrid_score": float(fused[idx]),
                "text_score":   float(norm_t[idx]),
                "image_score":  float(norm_i[idx]),
                "audio_score":  float(norm_a[idx]),
                "query_type":   qtype,
            }
        )

    return pd.DataFrame(rows)

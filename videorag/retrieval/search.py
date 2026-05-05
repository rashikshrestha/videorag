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
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
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
    qtype, cls_alpha, cls_beta, cls_gamma = classify_query(query, settings)
    alpha = alpha if alpha is not None else cls_alpha
    beta  = beta  if beta  is not None else cls_beta
    gamma = gamma if gamma is not None else cls_gamma
    print(f"Query classified as '{qtype}' with weights text(α)={alpha:.2f}, Image(β)={beta:.2f}, Audio(γ)={gamma:.2f}")

    #! Encode query with active encoders
    q_text  = embed_query_text(query, bundle)  if text_index  is not None else None
    q_clip  = embed_query_clip(query, bundle)  if image_index is not None else None
    q_audio = embed_query_audio(query, bundle) if audio_index is not None else None

    #! Query Text Index
    raw_t = np.zeros(n, dtype=np.float32)
    if text_index is not None and q_text is not None:
        ts, ti = text_index.search(q_text, n)
        # shape of ts, ti is (1, n) since we query with a single vector
        ts_map = dict(zip(ti[0].tolist(), ts[0].tolist()))
        raw_t = np.array([max(0.0, ts_map.get(i, 0.0)) for i in range(n)], dtype=np.float32)
        # shape raw_t is (n,) and contains the text similarity scores for each segment

    #! Query Image Index
    raw_i = np.zeros(n, dtype=np.float32)
    if image_index is not None and q_clip is not None:
        is_, ii = image_index.search(q_clip, n)
        is_map = dict(zip(ii[0].tolist(), is_[0].tolist()))
        raw_i = np.array([max(0.0, is_map.get(i, 0.0)) for i in range(n)], dtype=np.float32)

    #! Query Audio Index
    raw_a = np.zeros(n, dtype=np.float32)
    if audio_index is not None and q_audio is not None:
        as_, ai = audio_index.search(q_audio, n)
        as_map = dict(zip(ai[0].tolist(), as_[0].tolist()))
        raw_a = np.array([max(0.0, as_map.get(i, 0.0)) for i in range(n)], dtype=np.float32)

    #! Normalize betn 0-1
    norm_t = _safe_minmax(raw_t)
    norm_i = _safe_minmax(raw_i)
    norm_a = _safe_minmax(raw_a)

    #! Blending modalities with weights α, β, γ
    w_t = alpha if text_index  is not None else 0.0
    w_i = beta  if image_index is not None else 0.0
    w_a = gamma if (audio_index is not None and q_audio is not None) else 0.0
    total = w_t + w_i + w_a
    if total < 1e-8:
        total = 1.0
    fused = (w_t / total) * norm_t + (w_i / total) * norm_i + (w_a / total) * norm_a

    #! Character boost
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

    #! Get top-k results
    order = np.argsort(fused)[::-1][:top_k]

    #! Format results as DataFrame
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

    return_data = pd.DataFrame(rows)
    print(f"Hybrid search completed. Returning top-{len(return_data)} results.")
    return return_data

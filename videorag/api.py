"""
videorag.api
~~~~~~~~~~~~
Public entry points: context construction and single-call grounding API.

Public API
----------
build_context(config_path)  -> VideoRAGContext
run_video_grounding(query, ctx, config_path, top_k, merge_gap) -> dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from videorag.config import Settings, load_settings
from videorag.models.embeddings import load_models
from videorag.store.index import build_or_load_indices
from videorag.pipeline.pipeline import VideoRAGContext, ground


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(config_path: str | Path = "config.yaml") -> VideoRAGContext:
    """
    Load settings, models and FAISS indices; return a ready-to-use
    :class:`~videorag.pipeline.pipeline.VideoRAGContext`.

    This function performs the following steps:

    1. Parse ``config.yaml`` via :func:`~videorag.config.load_settings`.
    2. Load text and CLIP models via :func:`~videorag.models.embeddings.load_models`.
    3. Read ``<output_root>/segments.csv`` (must exist; run preprocessing
       first if needed).
     4. Build or load FAISS indices via
         :func:`~videorag.store.index.build_or_load_indices`.

    Args:
        config_path: Path to the project ``config.yaml``.

    Returns:
        Populated :class:`~videorag.pipeline.VideoRAGContext`.

    Raises:
        FileNotFoundError: when ``config.yaml`` or ``segments.csv`` is absent.
    """
    settings: Settings = load_settings(config_path)

    # ── Load segments CSV ──
    seg_path = settings.paths.output_root / "segments.csv"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"segments.csv not found at {seg_path}.\n"
            "Run preprocessing first:  "
            "python scripts/run_pipeline.py preprocess --config config.yaml"
        )

    segments_df = pd.read_csv(seg_path)
    segments_df["subtitle"] = segments_df["subtitle"].fillna("").astype(str)
    segments_df["frames"]   = segments_df["frames"].fillna("[]").astype(str)
    if "audio_path" not in segments_df.columns:
        segments_df["audio_path"] = ""
    if "audio_events" not in segments_df.columns:
        segments_df["audio_events"] = "[]"
    if "audio_event_text" not in segments_df.columns:
        segments_df["audio_event_text"] = ""
    segments_df["audio_path"] = segments_df["audio_path"].fillna("").astype(str)
    segments_df["audio_events"] = segments_df["audio_events"].fillna("[]").astype(str)
    segments_df["audio_event_text"] = segments_df["audio_event_text"].fillna("").astype(str)

    # BUG FIX #2: embed pure subtitle text (no episode/scene prefix noise)
    segments_df["embed_text"] = segments_df["subtitle"].apply(
        lambda x: x.strip() if x.strip() else "silent scene no dialogue"
    )
    print(f"Loaded {len(segments_df)} segments from {seg_path}")

    # ── Load models ──
    bundle = load_models(settings)

    # ── Build or load FAISS indices ──
    text_index, image_index, audio_index, text_emb, image_emb, audio_emb, segments_df = build_or_load_indices(
        segments_df, settings, bundle
    )

    return VideoRAGContext(
        settings=settings,
        bundle=bundle,
        segments_df=segments_df,
        text_index=text_index,
        image_index=image_index,
        audio_index=audio_index,
        text_embeddings=text_emb,
        image_embeddings=image_emb,
        audio_embeddings=audio_emb,
    )


# ---------------------------------------------------------------------------
# Single-call public API
# ---------------------------------------------------------------------------

def run_video_grounding(
    query: str,
    ctx: Optional[VideoRAGContext] = None,
    config_path: str | Path = "config.yaml",
    top_k: int = 5,
    merge_gap: float = 20.0,
) -> dict:
    """
    Single public entry point for video temporal grounding.

    Accepts an optional pre-built *ctx* for efficiency when performing
    multiple queries in a session (avoids re-loading models each call).
    When *ctx* is ``None``, one is built automatically from *config_path*.

    Args:
        query:       Natural-language description of the video moment.
        ctx:         Pre-built :class:`~videorag.pipeline.pipeline.VideoRAGContext`.
                     If ``None``, built from *config_path*.
        config_path: Path to ``config.yaml`` (used only when ``ctx`` is None).
        top_k:       Number of retrieval candidates.
        merge_gap:   Span merging gap in seconds.

    Returns:
        Dictionary with keys:

        * ``query``             — original query string
        * ``video``             — predicted video filename
        * ``scene_id``          — predicted scene index
        * ``predicted_start``   — grounded span start (seconds)
        * ``predicted_end``     — grounded span end (seconds)
        * ``span_seconds``      — grounded span duration
        * ``query_type``        — ``'action'`` / ``'dialogue'`` / ``'mixed'``
        * ``frame_aggregation`` — ``'max-pool (action)'`` or ``'mean-pool'``
        * ``confidence``        — calibrated confidence in ``[0, 1]``
        * ``evidence``          — up to 500 chars of subtitle evidence

    Example::

        from videorag.api import build_context, run_video_grounding

        ctx = build_context("config.yaml")
        print(run_video_grounding("Chandler Ross Joey watching ice hockey", ctx=ctx))
    """
    if ctx is None:
        ctx = build_context(config_path)

    out = ground(query, ctx, top_k=top_k, merge_gap=merge_gap)
    top = out.iloc[0]
    agg = "max-pool (action)" if top["query_type"] == "action" else "mean-pool"

    return {
        "query":             query,
        "video":             top["video"],
        "scene_id":          int(top["scene_id"]),
        "predicted_start":   float(top["refined_start"]),
        "predicted_end":     float(top["refined_end"]),
        "span_seconds":      float(top["span_seconds"]),
        "query_type":        top["query_type"],
        "frame_aggregation": agg,
        "confidence":        float(top["confidence"]),
        "evidence":          str(top["subtitle"])[:500],
    }

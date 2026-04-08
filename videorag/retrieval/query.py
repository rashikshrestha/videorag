"""
videorag.retrieval.query
~~~~~~~~~~~~~~
Query classification, embedding and scoring utilities.

Public API
----------
classify_query(query, settings)   -> (label, text_weight, image_weight)
embed_query_text(query, bundle)   -> np.ndarray  (1, 384) float32
embed_query_clip(query, bundle)   -> np.ndarray  (1, 512) float32
_safe_minmax(x)                   -> np.ndarray
_kw_overlap(query, subtitle_text) -> (int, float)
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import torch

from videorag.config import Settings
from videorag.models.embeddings import ModelBundle, embed_audio_query_text


# ---------------------------------------------------------------------------
# Fusion scoring helpers
# ---------------------------------------------------------------------------

def _safe_minmax(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalise *x* to ``[0, 1]``.

    Returns a zero array when the range is numerically zero (all scores
    identical), which prevents division-by-zero silently inflating ranks.
    """
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    return (x - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(x)


def _kw_overlap(query: str, subtitle_text: str) -> Tuple[int, float]:
    """
    Count and fraction of query words (len > 3) that appear in *subtitle_text*.

    Returns ``(hit_count, hit_fraction)``; both zero when the query has no
    qualifying words.
    """
    q_words = [w.lower() for w in query.split() if len(w) > 3]
    if not q_words:
        return 0, 0.0
    sub = str(subtitle_text).lower()
    hits = sum(1 for w in q_words if w in sub)
    return hits, hits / len(q_words)


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_ACTION_VOCAB = frozenset({
    "open", "opening", "close", "walk", "run", "sit", "stand", "hug", "kiss",
    "eat", "eating", "drink", "grab", "leave", "enter", "look", "hold", "pick",
    "fridge", "door", "couch", "dancing", "cooking", "reading", "writing",
    "playing", "watching", "smoking", "crying", "serving", "fighting",
    "sleeping", "laughing", "hugging",
})

_DIALOGUE_VOCAB = frozenset({
    "say", "tell", "ask", "talk", "talking", "conversation", "discuss", "joke",
    "argue", "mention", "explain", "about", "says", "told", "asks", "speaks",
    "speaking", "chat", "discussing", "admits", "confesses", "reveals",
})


def classify_query(
    query: str,
    settings: Settings,
) -> Tuple[str, float, float, float]:
    """
    Classify *query* into one of ``'action'``, ``'dialogue'`` or ``'mixed'``
    and return the corresponding fusion weights.

    The classification also determines the frame aggregation strategy used
    in :func:`~videorag.refinement.refine`:

    * ``action``   → **max-pool**  (one peak frame captures the key action)
    * ``dialogue`` → **mean-pool** (stable talking-head appearance)
    * ``mixed``    → **mean-pool** (conservative default)

    BUG FIX #8: original code used fixed α=0.7, β=0.3 regardless of query
    type, so visual evidence was ignored for dialogue queries and text was
    under-weighted for action queries.

    Args:
        query:    Natural-language search string.
        settings: Project settings (used to look up weights per query type).

    Returns:
        ``(label, text_weight, image_weight)`` where the weights sum to 1.
    """
    words = set(query.lower().split())
    a_score = len(words & _ACTION_VOCAB)
    d_score = len(words & _DIALOGUE_VOCAB)

    if a_score > d_score:
        label = "action"
    elif d_score > a_score:
        label = "dialogue"
    else:
        label = "mixed"

    ws = settings.retrieval.weights[label]
    return label, ws.text, ws.image, ws.audio


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query_text(query: str, bundle: ModelBundle) -> np.ndarray:
    """
    Encode *query* via SentenceTransformer into a (1, 384) unit-norm vector.
    """
    return bundle.text_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def embed_query_clip(query: str, bundle: ModelBundle) -> np.ndarray:
    """
    Encode *query* via the CLIP text encoder into a (1, 512) unit-norm vector.

    BUG FIX #1 — THE MOST CRITICAL FIX:
    The original code obtained CLIP text features from
    ``text_model().pooler_output`` *without* applying ``text_projection``.
    Image embeddings live in the CLIP shared space (after
    ``visual_projection``); the raw text pooler output is in a different
    BERT-like space.  Cosine similarity between the two was meaningless —
    every visual query returned garbage.  This is why "ice hockey" always
    retrieved wrong timestamps regardless of confidence level.

    Fix: apply ``text_projection`` so text and image vectors are both in
    the 512-dim CLIP shared embedding space.
    """
    inp = bundle.clip_processor(
        text=[query], return_tensors="pt", padding=True
    )
    with torch.no_grad():
        out = bundle.clip_model.text_model(
            input_ids=inp["input_ids"].to(bundle.device),
            attention_mask=inp["attention_mask"].to(bundle.device),
        )
        # ✅ Apply text_projection → CLIP shared space
        feat = bundle.clip_model.text_projection(out.pooler_output)
        feat = feat / torch.norm(feat, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def embed_query_audio(query: str, bundle: ModelBundle) -> np.ndarray | None:
    """Encode query text into CLAP text space for audio retrieval."""
    return embed_audio_query_text(query, bundle)

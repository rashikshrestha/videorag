"""
videorag.refinement
~~~~~~~~~~~~~~~~~~~
Fine-grained temporal grounding within a retrieved scene's neighbourhood.

Public API
----------
load_subs(video_name, video_root, subtitle_root)     -> pysubs2.SSAFile | None
get_duration(video_name, video_root)                  -> float
subs_in_range(subs, start, end)                       -> str
snap_to_subtitle_boundaries(subs, s, e, tolerance)    -> (float, float)
smooth_scores(x, window)                              -> np.ndarray
kw_ratio(query, text)                                 -> float
expand_using_subtitles(subs, query, s, e, max_extra)  -> (float, float)
refine(row, query, bundle, settings, video_root, subtitle_root)
    -> (span_start, span_end, confidence)
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pysubs2
import torch
from PIL import Image

from videorag.config import Settings
from videorag.models.embeddings import ModelBundle, _clip_single_frame
from videorag.retrieval.query import _safe_minmax, classify_query, embed_query_audio, embed_query_text


# ---------------------------------------------------------------------------
# Module-level caches (populated lazily, keyed by video filename)
# ---------------------------------------------------------------------------
_subtitle_cache: Dict[str, Optional[pysubs2.SSAFile]] = {}
_duration_cache: Dict[str, float] = {}


def _clear_caches() -> None:
    """Clear runtime caches (useful between test runs)."""
    _subtitle_cache.clear()
    _duration_cache.clear()


# ---------------------------------------------------------------------------
# Runtime subtitle / duration helpers
# ---------------------------------------------------------------------------

def load_subs(
    video_name: str,
    video_root: Path,
    subtitle_root: Path,
) -> Optional[pysubs2.SSAFile]:
    """
    Return a cached pysubs2 subtitle file for *video_name*, or ``None``.

    Results are memoised — the same file is parsed only once per session.
    """
    if video_name not in _subtitle_cache:
        from videorag.dataset.preprocessing import find_subtitle
        p = find_subtitle(video_root / video_name, subtitle_root)
        _subtitle_cache[video_name] = pysubs2.load(str(p)) if p else None
    return _subtitle_cache[video_name]


def get_duration(video_name: str, video_root: Path) -> float:
    """
    Return the total duration (seconds) of *video_name*, with caching.
    Returns 0.0 if the file cannot be opened.
    """
    if video_name not in _duration_cache:
        cap = cv2.VideoCapture(str(video_root / video_name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        _duration_cache[video_name] = n / fps if fps > 0 else 0.0
    return _duration_cache[video_name]


def subs_in_range(
    subs: Optional[pysubs2.SSAFile],
    start: float,
    end: float,
) -> str:
    """Return all subtitle text overlapping ``[start, end]`` as a single string."""
    if not subs:
        return ""
    return " ".join(
        line.text.replace("\\N", " ").replace("\n", " ").strip()
        for line in subs
        if (
            line.end / 1000.0 >= start
            and line.start / 1000.0 <= end
            and line.text.replace("\\N", " ").strip()
        )
    )


# ---------------------------------------------------------------------------
# Subtitle boundary helpers
# ---------------------------------------------------------------------------

def snap_to_subtitle_boundaries(
    subs: Optional[pysubs2.SSAFile],
    span_start: float,
    span_end: float,
    tolerance: float = 1.5,
) -> Tuple[float, float]:
    """
    Extend span edges to align with nearby subtitle line boundaries.

    Avoids partial-line truncation that silently reduces IoU.
    This function only **extends** — it never shrinks the span.

    Args:
        subs:        Loaded subtitle file, or ``None``.
        span_start:  Current span start (seconds).
        span_end:    Current span end (seconds).
        tolerance:   Maximum distance (seconds) from the span edge within
                     which a subtitle boundary is snapped.

    Returns:
        ``(new_start, new_end)`` — equal to or wider than the input span.
    """
    if not subs:
        return span_start, span_end

    ns, ne = span_start, span_end
    for line in subs:
        ls = line.start / 1000.0
        le = line.end   / 1000.0
        if abs(ls - span_start) <= tolerance and le <= span_end + tolerance:
            ns = min(ns, ls)
        if abs(le - span_end) <= tolerance and ls >= span_start - tolerance:
            ne = max(ne, le)
    return float(ns), float(ne)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def smooth_scores(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a simple moving-average smoothing to score array *x*."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def _norm_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "of", "and", "is", "are",
    "was", "were", "her", "his", "their", "with", "from", "like", "when",
    "then", "that", "this", "for",
})


def _content_words(q: str):
    """Return meaningful words from *q* (len > 2, not stop words)."""
    words = _norm_text(q).split()
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


def kw_ratio(query: str, text: str) -> float:
    """Fraction of content words in *query* that appear in *text*."""
    q_words = _content_words(query)
    if not q_words:
        return 0.0
    text_n = _norm_text(text)
    hits = sum(1 for w in q_words if w in text_n)
    return hits / len(q_words)


def expand_using_subtitles(
    subs: Optional[pysubs2.SSAFile],
    query: str,
    span_start: float,
    span_end: float,
    max_extra: float = 4.0,
) -> Tuple[float, float]:
    """
    Extend the span to include nearby subtitle lines that contain
    content words from *query*.

    Args:
        subs:       Loaded subtitle file, or ``None``.
        query:      The search query.
        span_start: Current span start (seconds).
        span_end:   Current span end (seconds).
        max_extra:  Maximum extension distance (seconds) on each side.

    Returns:
        Potentially extended ``(span_start, span_end)``.
    """
    if not subs:
        return float(span_start), float(span_end)

    q_words = _content_words(query)
    best_start, best_end = float(span_start), float(span_end)

    for line in subs:
        ls = line.start / 1000.0
        le = line.end   / 1000.0
        txt = line.text.replace("\\N", " ").replace("\n", " ").strip()
        if not txt:
            continue
        near = (le >= span_start - max_extra) and (ls <= span_end + max_extra)
        if not near:
            continue
        text_n = _norm_text(txt)
        if any(w in text_n for w in q_words):
            best_start = min(best_start, ls)
            best_end   = max(best_end, le)

    return float(best_start), float(best_end)


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def refine(
    row: pd.Series,
    query: str,
    bundle: ModelBundle,
    settings: Settings,
    video_root: Path,
    subtitle_root: Path,
) -> Tuple[float, float, float]:
    """
    Perform fine-grained temporal refinement within the neighbourhood
    of a candidate scene returned by :func:`~videorag.retrieval.hybrid_search`.

    Algorithm
    ---------
    1. Expand search window by ±``expand`` seconds beyond the scene boundary.
    2. Scan with overlapping ``bin_size``-second bins at ``stride``-second
       intervals.
    3. Per bin: subtitle semantic similarity (SentenceTransformer cosine +
       keyword overlap) + multi-frame CLIP visual similarity.
    4. Frame aggregation:
       * ``action``  → **max-pool** per bin  (one peak frame matters)
       * other types → **mean-pool** per bin (stable appearance)
    5. Smooth the temporal score curve (moving average).
    6. Peak-expansion at 50 % of peak value → raw span.
    7. Enforce ``min_span`` / ``max_span`` constraints.
    8. Snap to subtitle boundaries (avoids partial-line IoU loss).
    9. Expand span to cover nearby subtitles containing query words.

    BUG FIX #6: VideoCapture opened **once** per call (not once per bin).
    BUG FIX #7: Peak-expand with 50 % threshold + 10 s ``min_span``
    (replaced the original hard 0.85 threshold that produced 2-3 s spans).

    Args:
        row:          A single row from the hybrid_search DataFrame.
        query:        The original search query string.
        bundle:       Loaded :class:`~videorag.models.embeddings.ModelBundle`.
        settings:     Project settings.
        video_root:   Root directory containing video files.
        subtitle_root: Root directory containing subtitle files.

    Returns:
        ``(span_start, span_end, confidence)`` — all in seconds,
        where *confidence* equals the peak smoothed fused score.
    """
    video    = row["video"]
    sc_start = float(row["start"])
    sc_end   = float(row["end"])
    dur      = get_duration(video, video_root)
    subs     = load_subs(video, video_root, subtitle_root)

    qtype, alpha_s, beta_i, gamma_a = classify_query(query, settings)
    frame_agg = "max" if qtype == "action" else "mean"

    ref = settings.refinement

    # ── Query features (computed once per call) ──
    q_emb = embed_query_text(query, bundle)
    q_aud = embed_query_audio(query, bundle) if settings.audio.enabled else None

    txt_inp = bundle.clip_processor(
        text=[query], return_tensors="pt", padding=True
    )
    with torch.no_grad():
        t_out = bundle.clip_model.text_model(
            input_ids=txt_inp["input_ids"].to(bundle.device),
            attention_mask=txt_inp["attention_mask"].to(bundle.device),
        )
        t_feat = bundle.clip_model.text_projection(t_out.pooler_output)
        t_feat = t_feat / torch.norm(t_feat, dim=-1, keepdim=True)
    t_feat_np = t_feat.detach().cpu().numpy()

    lo = max(0.0, sc_start - ref.expand)
    hi = min(float(dur), sc_end + ref.expand)

    # BUG FIX #6: Open VideoCapture once for the whole refinement call.
    cap = cv2.VideoCapture(str(video_root / video))
    fps = cap.get(cv2.CAP_PROP_FPS)

    bin_fracs = np.linspace(0.15, 0.85, ref.bin_frames).tolist()
    bins = []
    t = lo

    while t < hi:
        b_s = t
        b_e = min(t + ref.bin_size, hi)

        # ── Subtitle score ──
        sub_text = subs_in_range(subs, b_s, b_e)
        if sub_text.strip():
            t_emb = bundle.text_model.encode(
                [sub_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            s_sem   = float((q_emb @ t_emb.T)[0, 0])
            s_kw    = kw_ratio(query, sub_text)
            s_score = 0.75 * s_sem + 0.25 * s_kw
        else:
            s_score = 0.0

        # ── Multi-frame visual score ──
        i_score = 0.0
        if fps and fps > 0:
            sims = []
            for frac in bin_fracs:
                t_sample = b_s + frac * (b_e - b_s)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sample * fps))
                ret, frame = cap.read()
                if ret:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    v   = _clip_single_frame(pil, bundle)
                    if v is not None:
                        sim = float((v.reshape(1, -1) @ t_feat_np.T)[0, 0])
                        sims.append(max(0.0, sim))
            if sims:
                i_score = float(
                    np.max(sims) if frame_agg == "max" else np.mean(sims)
                )

        # ── Audio score (non-ASR cues via CLAP text↔audio similarity) ──
        a_score = 0.0
        if (
            q_aud is not None
            and bundle.audio_model is not None
            and bundle.audio_processor is not None
            and settings.audio.enabled
        ):
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{b_s:.3f}",
                "-to",
                f"{b_e:.3f}",
                "-i",
                str(video_root / video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(settings.audio.sample_rate),
                "-f",
                "f32le",
                "pipe:1",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, check=False)
                if proc.returncode == 0 and proc.stdout:
                    wav = np.frombuffer(proc.stdout, dtype=np.float32)
                    if wav.size > 0:
                        inp = bundle.audio_processor(
                            audios=[wav],
                            sampling_rate=settings.audio.sample_rate,
                            return_tensors="pt",
                        )
                        with torch.no_grad():
                            af = bundle.audio_model.get_audio_features(
                                input_features=inp["input_features"].to(bundle.device),
                            )
                            af = af / torch.norm(af, dim=-1, keepdim=True)
                        av = af.detach().cpu().numpy()
                        a_score = max(0.0, float((av @ q_aud.T)[0, 0]))
            except Exception:
                a_score = 0.0

        bins.append({"b_s": b_s, "b_e": b_e, "s_raw": s_score, "i_raw": i_score, "a_raw": a_score})
        t += ref.stride

    cap.release()

    if not bins:
        fallback = float(row.get("hybrid_score", 0.0))
        return sc_start, sc_end, fallback

    bdf = pd.DataFrame(bins).reset_index(drop=True)

    # ── Normalise within window then fuse ──
    bdf["s_norm"]    = _safe_minmax(bdf["s_raw"].values)
    bdf["i_norm"]    = _safe_minmax(bdf["i_raw"].values)
    bdf["a_norm"]    = _safe_minmax(bdf["a_raw"].values)
    if q_aud is None:
        total = alpha_s + beta_i
        alpha_adj = alpha_s / total if total > 1e-8 else 0.5
        beta_adj = beta_i / total if total > 1e-8 else 0.5
        bdf["score_raw"] = alpha_adj * bdf["s_norm"] + beta_adj * bdf["i_norm"]
    else:
        bdf["score_raw"] = (
            alpha_s * bdf["s_norm"]
            + beta_i * bdf["i_norm"]
            + gamma_a * bdf["a_norm"]
        )
    bdf["score"]     = smooth_scores(bdf["score_raw"].values, window=ref.smooth_window)

    # ── Peak expansion ──
    peak_idx = int(np.argmax(bdf["score"].values))
    peak_val = float(bdf.loc[peak_idx, "score"])
    threshold = max(0.25, 0.50 * peak_val)

    left = peak_idx
    while left > 0 and float(bdf.loc[left - 1, "score"]) >= threshold:
        left -= 1

    right = peak_idx
    while right < len(bdf) - 1 and float(bdf.loc[right + 1, "score"]) >= threshold:
        right += 1

    span_start = float(bdf.loc[left, "b_s"])
    span_end   = float(bdf.loc[right, "b_e"])

    # Wider buffer to catch near-miss boundaries
    span_start = max(lo, span_start - 3.0)
    span_end   = min(hi, span_end   + 3.0)

    # ── Enforce minimum span ──
    span_dur = span_end - span_start
    if span_dur < ref.min_span:
        pad = (ref.min_span - span_dur) / 2.0
        span_start = max(lo, span_start - pad)
        span_end   = min(hi, span_end   + pad)
        if span_end - span_start < ref.min_span:
            if span_start <= lo:
                span_end = min(hi, span_start + ref.min_span)
            else:
                span_start = max(lo, span_end - ref.min_span)

    # ── Cap maximum span ──
    if span_end - span_start > ref.max_span:
        ctr = 0.5 * (span_start + span_end)
        span_start = max(lo, ctr - ref.max_span / 2.0)
        span_end   = min(hi, ctr + ref.max_span / 2.0)

    # ── Subtitle boundary snapping ──
    span_start, span_end = snap_to_subtitle_boundaries(
        subs, span_start, span_end, tolerance=ref.snap_tolerance
    )

    # ── Keyword-based subtitle expansion ──
    span_start, span_end = expand_using_subtitles(
        subs, query, span_start, span_end, max_extra=ref.expand_max_extra
    )

    # Final clamp
    span_start = max(0.0, min(span_start, hi))
    span_end   = max(span_start, min(span_end, hi))

    # Re-cap after expansion
    if span_end - span_start > ref.max_span:
        ctr = 0.5 * (span_start + span_end)
        span_start = max(0.0, ctr - ref.max_span / 2.0)
        span_end   = min(float(dur), ctr + ref.max_span / 2.0)

    confidence = float(np.max(bdf["score"].values))
    return float(span_start), float(span_end), confidence

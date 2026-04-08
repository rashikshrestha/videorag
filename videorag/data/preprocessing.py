"""
videorag.preprocessing
~~~~~~~~~~~~~~~~~~~~~~
Scene detection, multi-frame extraction and subtitle matching.

Public API
----------
find_subtitle(video_path, subtitle_root) -> Path | None
detect_scenes(video_path, threshold)     -> list[tuple[float, float]]
extract_keyframes(video_path, start, end, out_dir, fractions) -> list[str]
get_subtitle_text(subs, start, end)      -> str
run_preprocessing(settings)              -> pd.DataFrame
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pysubs2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

from videorag.config import Settings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Subtitle matching
# ---------------------------------------------------------------------------

def find_subtitle(
    video_path: Path | str,
    subtitle_root: Path | str,
) -> Optional[Path]:
    """
    Locate the subtitle file for *video_path* inside *subtitle_root*.

    Strategy:
    1. Exact stem match  (``<subtitle_root>/<stem>.srt|ass|ssa``)
    2. Fuzzy match       (normalised names, substring containment)

    BUG FIX #9: original code failed when called with a bare filename
    because the implicit Path join produced wrong results.  This version
    accepts both a bare name and a full path and adds fuzzy fallback.

    Returns ``None`` when no subtitle file can be found.
    """
    stem = Path(video_path).stem
    subtitle_root = Path(subtitle_root)

    for ext in (".srt", ".ass", ".ssa"):
        candidate = subtitle_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Fuzzy: normalise both names and check substring containment.
    norm = stem.lower().replace("_", "").replace("-", "").replace(".", "")
    for f in subtitle_root.iterdir():
        if f.suffix.lower() in (".srt", ".ass", ".ssa"):
            fn = f.stem.lower().replace("_", "").replace("-", "").replace(".", "")
            if norm in fn or fn in norm:
                return f

    return None


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def detect_scenes(
    video_path: Path | str,
    threshold: int = 27,
) -> List[Tuple[float, float]]:
    """
    Use PySceneDetect's ``ContentDetector`` to find scene boundaries.

    Args:
        video_path: Path to the video file.
        threshold:  Sensitivity threshold for the content detector.
                    Lower values detect more (finer) scenes.

    Returns:
        List of ``(start_sec, end_sec)`` tuples, one per detected scene.
    """
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    return [
        (s[0].get_seconds(), s[1].get_seconds())
        for s in manager.get_scene_list()
    ]


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------

def extract_keyframes(
    video_path: Path | str,
    start: float,
    end: float,
    out_dir: Path | str,
    fractions: Optional[List[float]] = None,
) -> List[str]:
    """
    Extract *N* frames at evenly-spaced fractional positions within
    ``[start, end]`` and save them as JPEG files.

    Why fractions instead of a single midpoint?
    Scenes are heterogeneous — an action (opening fridge, hug) can
    occur at any point.  5 frames spread across the scene ensure at
    least one lands near the key visual event.

    Fractions avoid 0.0 and 1.0 because those timestamps often land on
    scene-cut black frames.

    Args:
        video_path: Source video file path.
        start:      Scene start time in seconds.
        end:        Scene end time in seconds.
        out_dir:    Directory to write frame JPEGs.
        fractions:  Fractional positions to sample (default: 5 fractions
                    from the settings-configured list).

    Returns:
        List of absolute paths (str) for each saved frame, in order.
        Frames that could not be decoded are silently omitted.
    """
    if fractions is None:
        fractions = [0.10, 0.30, 0.50, 0.70, 0.90]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    saved: List[str] = []

    if fps > 0:
        for i, frac in enumerate(fractions):
            t = start + frac * (end - start)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if ret:
                p = out_dir / f"f{i}.jpg"
                cv2.imwrite(str(p), frame)
                saved.append(str(p))

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Subtitle extraction helper
# ---------------------------------------------------------------------------

def get_subtitle_text(
    subs: Optional[pysubs2.SSAFile],
    start: float,
    end: float,
) -> str:
    """
    Return a single string of all subtitle lines that overlap ``[start, end]``.

    Args:
        subs:  Loaded pysubs2 subtitle file, or ``None``.
        start: Interval start in seconds.
        end:   Interval end in seconds.

    Returns:
        Space-joined subtitle text, empty string when ``subs`` is ``None``
        or no lines overlap the interval.
    """
    if subs is None:
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
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(settings: Settings) -> pd.DataFrame:
    """
    Discover all videos under ``settings.paths.video_root``, detect scenes,
    extract multi-frame keyframes, collect subtitle text and build a
    ``segments.csv`` representing every scene.

    Saved columns
    -------------
    video     : filename of the source video
    scene_id  : integer scene index within the video
    start     : scene start time (seconds)
    end       : scene end time (seconds)
    duration  : end − start (seconds)
    subtitle  : all subtitle lines overlapping the scene
    frames    : JSON list of absolute paths to the N extracted frame JPEGs

    Args:
        settings: Project :class:`~videorag.config.Settings`.

    Returns:
        DataFrame with one row per scene, also written to
        ``<output_root>/segments.csv``.

    Raises:
        FileNotFoundError: when no video files are found in ``video_root``.
    """
    video_root = settings.paths.video_root
    subtitle_root = settings.paths.subtitle_root
    output_root = settings.paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Discover videos ──
    video_files = sorted(
        f
        for f in video_root.iterdir()
        if f.suffix.lower() in (".mkv", ".mp4", ".avi")
        and "friends" in f.name.lower()
    )
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_root}")

    print(f"Found {len(video_files)} video(s):")
    for v in video_files:
        sub = find_subtitle(v, subtitle_root)
        print(f"  {v.name:50s}  subtitle={'YES' if sub else 'MISSING'}")

    # ── Build segments DataFrame ──
    rows = []
    for video_path in tqdm(video_files, desc="Preprocessing"):
        sub_path = find_subtitle(video_path, subtitle_root)
        subs = pysubs2.load(str(sub_path)) if sub_path else None
        scenes = detect_scenes(video_path, threshold=settings.preprocessing.scene_threshold)

        print(
            f"  {video_path.name}: {len(scenes)} scenes  "
            f"subtitle={'yes' if subs else 'NO'}"
        )

        for i, (start, end) in enumerate(scenes):
            frame_dir = output_root / "frames" / video_path.stem / str(i)
            frame_paths = extract_keyframes(
                video_path,
                start,
                end,
                frame_dir,
                fractions=settings.preprocessing.frame_fractions,
            )
            text = get_subtitle_text(subs, start, end)
            rows.append(
                {
                    "video": video_path.name,
                    "scene_id": i,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "subtitle": text,
                    "frames": json.dumps(frame_paths),
                }
            )

    df = pd.DataFrame(rows)
    segments_path = output_root / "segments.csv"
    df.to_csv(segments_path, index=False)

    frame_counts = df["frames"].apply(lambda x: len(json.loads(x)))
    print(f"\n✅ {len(df)} segments saved → {segments_path}")
    print(
        f"   Frames/segment: min={frame_counts.min()} "
        f"mean={frame_counts.mean():.1f} max={frame_counts.max()}"
    )
    print(f"   Total frames on disk: {frame_counts.sum()}")
    print(f"   With subtitles:       {(df['subtitle'].str.len() > 0).sum()}")

    return df

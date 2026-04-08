from __future__ import annotations

import contextlib
import json
import os
import subprocess
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


@contextlib.contextmanager
def _suppress_native_stderr():
    """
    Temporarily silence native stderr (C/C++ libs like FFmpeg via OpenCV).

    Useful for noisy, recoverable decoder warnings that otherwise flood logs.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
            os.close(devnull_fd)
        except Exception:
            pass


def _extract_embedded_subtitle(
    video_path: Path,
    cache_dir: Path,
) -> Optional[Path]:
    """
    Extract the first embedded subtitle stream to a sidecar subtitle file.

    Returns extracted subtitle path (srt/ass) on success, otherwise None.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_srt = cache_dir / f"{video_path.stem}.embedded.srt"
    if out_srt.exists() and out_srt.stat().st_size > 0:
        return out_srt

    cmd_srt = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:s:0",
        str(out_srt),
    ]
    try:
        proc = subprocess.run(cmd_srt, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_srt.exists() and out_srt.stat().st_size > 0:
            return out_srt
    except Exception:
        pass

    out_ass = cache_dir / f"{video_path.stem}.embedded.ass"
    if out_ass.exists() and out_ass.stat().st_size > 0:
        return out_ass

    cmd_ass = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:s:0",
        str(out_ass),
    ]
    try:
        proc = subprocess.run(cmd_ass, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_ass.exists() and out_ass.stat().st_size > 0:
            return out_ass
    except Exception:
        pass

    return None


def _extract_scene_audio(
    video_path: Path,
    start: float,
    end: float,
    out_wav: Path,
    sample_rate: int,
) -> Optional[str]:
    """
    Extract mono wav audio for [start, end] using ffmpeg.

    Returns output path as string on success, otherwise None.
    """
    if end <= start:
        return None

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 44:
            return str(out_wav)
    except Exception:
        pass
    return None


def find_subtitle(
    video_path: Path | str,
    subtitle_root: Path | str | None,
    try_embedded: bool = True,
    embedded_cache_dir: Path | str | None = None,
) -> Optional[Path]:
    """
    Locate subtitle for *video_path*.

    Strategy:
    1. Exact stem match in ``subtitle_root`` and video directory
       (``<stem>.srt|ass|ssa``)
    2. Fuzzy match in ``subtitle_root`` (normalised substring containment)
    3. Optional fallback: extract first embedded subtitle stream via ffmpeg
       into ``embedded_cache_dir``.

    Returns ``None`` when no subtitle file can be found.
    """
    video_path = Path(video_path)
    stem = video_path.stem
    subtitle_root_path = Path(subtitle_root) if subtitle_root is not None else None

    search_roots = [video_path.parent]
    if subtitle_root_path is not None:
        search_roots.insert(0, subtitle_root_path)

    for root in search_roots:
        for ext in (".srt", ".ass", ".ssa"):
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    # Fuzzy: normalise both names and check substring containment.
    norm = stem.lower().replace("_", "").replace("-", "").replace(".", "")
    if subtitle_root_path is not None and subtitle_root_path.exists():
        for f in subtitle_root_path.iterdir():
            if f.suffix.lower() in (".srt", ".ass", ".ssa"):
                fn = f.stem.lower().replace("_", "").replace("-", "").replace(".", "")
                if norm in fn or fn in norm:
                    return f

    if try_embedded:
        cache_dir = (
            Path(embedded_cache_dir)
            if embedded_cache_dir is not None
            else (video_path.parent / "_embedded_subs_cache")
        )
        return _extract_embedded_subtitle(video_path, cache_dir)

    return None


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
    with _suppress_native_stderr():
        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video)
    return [
        (s[0].get_seconds(), s[1].get_seconds())
        for s in manager.get_scene_list()
    ]


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

    saved: List[str] = []
    with _suppress_native_stderr():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

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
    audio_path: absolute path to extracted scene-level wav clip
    audio_events: JSON list of top-k sound events (filled at indexing step)
    audio_event_text: whitespace-joined event labels (filled at indexing step)

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
    audio_enabled = bool(getattr(settings, "audio", None) and settings.audio.enabled)
    audio_sample_rate = (
        int(settings.audio.sample_rate)
        if audio_enabled
        else 16000
    )
    audio_base_dir = (
        output_root / settings.audio.scene_audio_dir
        if audio_enabled
        else output_root / "audio" / "scenes"
    )

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
        sub = find_subtitle(
            v,
            subtitle_root,
            try_embedded=True,
            embedded_cache_dir=video_root / "_embedded_subs_cache",
        )
        print(f"  {v.name:50s}  subtitle={'YES' if sub else 'MISSING'}")

    # ── Build segments CSV incrementally (flush after each video) ──
    segments_path = output_root / "segments.csv"
    columns = [
        "video",
        "scene_id",
        "start",
        "end",
        "duration",
        "subtitle",
        "frames",
        "audio_path",
        "audio_events",
        "audio_event_text",
    ]
    wrote_header = False
    if segments_path.exists():
        segments_path.unlink()

    for video_path in tqdm(video_files, desc="Preprocessing"):
        sub_path = find_subtitle(
            video_path,
            subtitle_root,
            try_embedded=True,
            embedded_cache_dir=video_root / "_embedded_subs_cache",
        )
        subs = pysubs2.load(str(sub_path)) if sub_path else None
        scenes = detect_scenes(video_path, threshold=settings.preprocessing.scene_threshold)

        print(
            f"  {video_path.name}: {len(scenes)} scenes  "
            f"subtitle={'yes' if subs else 'NO'}"
        )

        video_rows = []
        for i, (start, end) in tqdm(
            enumerate(scenes),
            total=len(scenes),
            desc=f"Scenes [{video_path.stem}]",
            leave=False,
        ):
            frame_dir = output_root / "frames" / video_path.stem / str(i)
            frame_paths = extract_keyframes(
                video_path,
                start,
                end,
                frame_dir,
                fractions=settings.preprocessing.frame_fractions,
            )
            text = get_subtitle_text(subs, start, end)

            audio_path = ""
            if audio_enabled:
                wav_path = audio_base_dir / video_path.stem / f"{i}.wav"
                audio_path = _extract_scene_audio(
                    video_path=video_path,
                    start=start,
                    end=end,
                    out_wav=wav_path,
                    sample_rate=audio_sample_rate,
                ) or ""

            video_rows.append(
                {
                    "video": video_path.name,
                    "scene_id": i,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "subtitle": text,
                    "frames": json.dumps(frame_paths),
                    "audio_path": audio_path,
                    "audio_events": "[]",
                    "audio_event_text": "",
                }
            )

        if video_rows:
            video_df = pd.DataFrame(video_rows, columns=columns)
            video_df.to_csv(
                segments_path,
                mode="a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            print(f"    ↳ saved {len(video_rows)} scene rows to {segments_path}")

    if segments_path.exists():
        df = pd.read_csv(segments_path)
    else:
        df = pd.DataFrame(columns=columns)

    frame_counts = df["frames"].apply(lambda x: len(json.loads(x)))
    print(f"\n✅ {len(df)} segments saved → {segments_path}")
    print(
        f"   Frames/segment: min={frame_counts.min()} "
        f"mean={frame_counts.mean():.1f} max={frame_counts.max()}"
    )
    print(f"   Total frames on disk: {frame_counts.sum()}")
    print(f"   With subtitles:       {(df['subtitle'].str.len() > 0).sum()}")

    return df

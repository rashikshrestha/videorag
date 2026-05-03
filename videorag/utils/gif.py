"""
videorag.utils.gif
~~~~~~~~~~~~~~~~~~
Extract equally-spaced frames from a video clip and save as an animated GIF.

Public API
----------
save_result_gifs(results, video_root, out_dir, n_frames, duration_ms) -> list[Path]
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image


# Maximum width for GIF frames (height scaled proportionally).
_GIF_MAX_WIDTH = 480


def save_result_gifs(
    results: pd.DataFrame,
    video_root: Path,
    out_dir: Path,
    n_frames: int = 10,
    duration_ms: int = 200,
) -> list[Path]:
    """
    For each row in *results*, sample *n_frames* equally-spaced frames between
    ``refined_start`` and ``refined_end`` from the source video and write an
    animated GIF to *out_dir*.

    Args:
        results:     DataFrame returned by :func:`~videorag.pipeline.pipeline.run`.
        video_root:  Directory that contains the source video files.
        out_dir:     Destination directory (created if it does not exist).
        n_frames:    Number of frames to sample per clip (default: 10).
        duration_ms: Display time per frame in milliseconds (default: 200).

    Returns:
        Paths of the GIF files that were successfully written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(video_root)

    saved: list[Path] = []
    for i, (_, row) in enumerate(results.iterrows(), 1):
        video_path = video_root / row["video"]
        if not video_path.exists():
            print(f"[gif] Video not found, skipping result {i}: {video_path}")
            continue

        start = float(row["refined_start"])
        end   = float(row["refined_end"])
        if end <= start:
            print(f"[gif] Invalid span [{start:.2f}, {end:.2f}] for result {i}, skipping.")
            continue

        frames = _extract_frames(video_path, start, end, n_frames)
        if not frames:
            print(f"[gif] No frames extracted for result {i}, skipping.")
            continue

        gif_path = out_dir / f"result_{i:02d}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[gif] Saved result {i}: {gif_path}")
        saved.append(gif_path)

    return saved


def _extract_frames(
    video_path: Path,
    start: float,
    end: float,
    n: int,
) -> list[Image.Image]:
    """
    Open *video_path* with OpenCV and return *n* PIL Images at equally-spaced
    timestamps between *start* and *end* (seconds).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    timestamps = np.linspace(start, end, n)

    frames: list[Image.Image] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = _resize(img, _GIF_MAX_WIDTH)
        frames.append(img)

    cap.release()
    return frames


def _resize(img: Image.Image, max_width: int) -> Image.Image:
    """Downscale *img* so its width is at most *max_width*, preserving aspect ratio."""
    w, h = img.size
    if w <= max_width:
        return img
    new_w = max_width
    new_h = int(h * max_width / w)
    return img.resize((new_w, new_h), Image.LANCZOS)

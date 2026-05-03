"""
videorag.utils.gif
~~~~~~~~~~~~~~~~~~
Extract equally-spaced frames from a video clip and save as an animated GIF.

Public API
----------
save_grid_gif(results, video_root, out_dir, n_frames, duration_ms, n_cols) -> Path | None
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Cell size for each result inside the grid (width × height in pixels).
_CELL_W = 320
_CELL_H = 180

# Thin border drawn between cells (pixels).
_BORDER = 2
_BORDER_COLOR = (40, 40, 40)

# Label strip at the top of each cell.
_LABEL_H = 22
_LABEL_BG = (20, 20, 20)
_LABEL_FG = (220, 220, 220)


def save_grid_gif(
    results: pd.DataFrame,
    video_root: Path,
    out_dir: Path,
    n_frames: int = 10,
    duration_ms: int = 200,
    n_cols: int = 3,
) -> Path | None:
    """
    Sample *n_frames* equally-spaced frames from every result clip, then
    compose each frame-step into a grid and save the whole sequence as one
    animated GIF (``grid.gif``) inside *out_dir*.

    Grid layout: up to *n_cols* results per row; rows added as needed.
    Each cell is labelled with its 1-based rank.

    Args:
        results:     DataFrame returned by :func:`~videorag.pipeline.pipeline.run`.
        video_root:  Directory that contains the source video files.
        out_dir:     Destination directory (created if it does not exist).
        n_frames:    Frames to sample per clip (default: 10).
        duration_ms: Display time per GIF frame in milliseconds (default: 200).
        n_cols:      Maximum number of results per grid row (default: 3).

    Returns:
        Path of the saved GIF, or ``None`` if no frames could be extracted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(video_root)

    # --- collect per-result frame strips -----------------------------------
    # strips[i] is a list of n_frames PIL Images for result i+1.
    strips: list[list[Image.Image]] = []
    labels: list[str] = []

    for i, (_, row) in enumerate(results.iterrows(), 1):
        video_path = video_root / row["video"]
        if not video_path.exists():
            print(f"[gif] Video not found, skipping result {i}: {video_path}")
            strips.append([])
            labels.append(f"#{i} (missing)")
            continue

        start = float(row["refined_start"])
        end   = float(row["refined_end"])
        if end <= start:
            print(f"[gif] Invalid span for result {i}, skipping.")
            strips.append([])
            labels.append(f"#{i} (invalid)")
            continue

        frames = _extract_frames(video_path, start, end, n_frames)
        strips.append(frames)
        labels.append(f"#{i}")
        if frames:
            print(f"[gif] Extracted {len(frames)} frames for result {i}")
        else:
            print(f"[gif] No frames extracted for result {i}")

    n_results = len(strips)
    if n_results == 0 or all(len(s) == 0 for s in strips):
        print("[gif] Nothing to save.")
        return None

    # Pad every strip to n_frames with blank cells so indexing is uniform.
    blank_cell = Image.new("RGB", (_CELL_W, _CELL_H), (0, 0, 0))
    padded: list[list[Image.Image]] = []
    for s in strips:
        if len(s) == 0:
            padded.append([blank_cell] * n_frames)
        elif len(s) < n_frames:
            padded.append(s + [s[-1]] * (n_frames - len(s)))
        else:
            padded.append(s[:n_frames])

    # --- build grid frames -------------------------------------------------
    actual_cols = min(n_cols, n_results)
    n_rows = math.ceil(n_results / actual_cols)

    cell_h_total = _CELL_H + _LABEL_H  # label strip + video area
    grid_w = actual_cols * _CELL_W + (actual_cols + 1) * _BORDER
    grid_h = n_rows * cell_h_total + (n_rows + 1) * _BORDER

    grid_frames: list[Image.Image] = []
    for f_idx in range(n_frames):
        canvas = Image.new("RGB", (grid_w, grid_h), _BORDER_COLOR)
        for i, (strip, label) in enumerate(zip(padded, labels)):
            col = i % actual_cols
            row = i // actual_cols
            x = _BORDER + col * (_CELL_W + _BORDER)
            y = _BORDER + row * (cell_h_total + _BORDER)

            # Label strip
            lbl_img = _make_label(label, _CELL_W, _LABEL_H)
            canvas.paste(lbl_img, (x, y))

            # Video frame
            canvas.paste(strip[f_idx], (x, y + _LABEL_H))

        grid_frames.append(canvas)

    gif_path = out_dir / "grid.gif"
    grid_frames[0].save(
        gif_path,
        save_all=True,
        append_images=grid_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"[gif] Saved grid GIF ({n_results} results, {n_frames} frames): {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_frames(
    video_path: Path,
    start: float,
    end: float,
    n: int,
) -> list[Image.Image]:
    """
    Open *video_path* with OpenCV and return *n* PIL Images at equally-spaced
    timestamps between *start* and *end* (seconds), resized to ``_CELL_W × _CELL_H``.
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
        img = Image.fromarray(rgb).resize((_CELL_W, _CELL_H), Image.LANCZOS)
        frames.append(img)

    cap.release()
    return frames


def _make_label(text: str, width: int, height: int) -> Image.Image:
    """Render a small dark label bar with *text* centred."""
    img = Image.new("RGB", (width, height), _LABEL_BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=_LABEL_FG, font=font)
    return img

"""
videorag.utils.gif
~~~~~~~~~~~~~~~~~~
Extract equally-spaced frames from a video clip and save as an animated GIF.

Public API
----------
save_grid_gif(results, video_root, out_dir, query, n_frames, duration_ms, n_cols) -> Path | None
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Cell dimensions for each result (pixels).
_CELL_W = 320
_CELL_H = 180

# Thin separator drawn between cells.
_BORDER = 2
_BORDER_COLOR = (40, 40, 40)

# Height of the rank label strip above each video frame.
_LABEL_H = 22
_LABEL_BG = (20, 20, 20)
_LABEL_FG = (220, 220, 220)

# Height of the subtitle strip below each video frame.
_SUB_H = 52
_SUB_BG = (15, 15, 15)
_SUB_FG = (200, 200, 200)

# Height of the query header bar that spans the full grid width.
_QUERY_H = 32
_QUERY_BG = (10, 40, 80)
_QUERY_FG = (240, 240, 240)

# Approximate characters that fit in one subtitle line at the default font.
_SUB_WRAP_CHARS = 42


def save_grid_gif(
    results: pd.DataFrame,
    video_root: Path,
    out_dir: Path,
    query: str = "",
    n_frames: int = 10,
    duration_ms: int = 200,
    n_cols: int = 3,
) -> Path | None:
    """
    Sample *n_frames* equally-spaced frames from every result clip, compose
    each frame-step into a grid, and save the whole sequence as one animated
    GIF (``grid.gif``) inside *out_dir*.

    Layout (top → bottom):
      • Query header bar spanning the full grid width.
      • Grid of cells, ≤ *n_cols* per row; each cell contains:
        – rank label  (#1, #2, …)
        – video frame
        – subtitle strip

    Args:
        results:     DataFrame returned by :func:`~videorag.pipeline.pipeline.run`.
        video_root:  Directory that contains the source video files.
        out_dir:     Destination directory (created if it does not exist).
        query:       Query string shown in the header bar.
        n_frames:    Frames to sample per clip (default: 10).
        duration_ms: Display time per GIF frame in milliseconds (default: 200).
        n_cols:      Maximum number of results per grid row (default: 3).

    Returns:
        Path of the saved GIF, or ``None`` if no frames could be extracted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(video_root)

    # --- collect per-result frame strips and metadata ----------------------
    strips:    list[list[Image.Image]] = []
    labels:    list[str] = []
    subtitles: list[str] = []

    for i, (_, row) in enumerate(results.iterrows(), 1):
        subtitle_raw = str(row.get("subtitle", "")).replace("\n", " ").strip()
        subtitles.append(subtitle_raw)

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

    # Pad every strip to n_frames (repeat last frame or use blank).
    blank_cell = Image.new("RGB", (_CELL_W, _CELL_H), (0, 0, 0))
    padded: list[list[Image.Image]] = []
    for s in strips:
        if len(s) == 0:
            padded.append([blank_cell] * n_frames)
        elif len(s) < n_frames:
            padded.append(s + [s[-1]] * (n_frames - len(s)))
        else:
            padded.append(s[:n_frames])

    # Pre-render static per-cell images (label + subtitle) — same every frame.
    label_imgs:    list[Image.Image] = [_make_label(lbl, _CELL_W, _LABEL_H) for lbl in labels]
    subtitle_imgs: list[Image.Image] = [_make_subtitle(sub, _CELL_W, _SUB_H) for sub in subtitles]

    # --- build grid frames -------------------------------------------------
    actual_cols = min(n_cols, n_results)
    n_rows = math.ceil(n_results / actual_cols)

    cell_h_total = _LABEL_H + _CELL_H + _SUB_H
    grid_w = actual_cols * _CELL_W + (actual_cols + 1) * _BORDER
    grid_h = _QUERY_H + n_rows * cell_h_total + (n_rows + 1) * _BORDER

    query_header = _make_query_header(query, grid_w, _QUERY_H)

    grid_frames: list[Image.Image] = []
    for f_idx in range(n_frames):
        canvas = Image.new("RGB", (grid_w, grid_h), _BORDER_COLOR)
        canvas.paste(query_header, (0, 0))

        for i, (strip, lbl_img, sub_img) in enumerate(
            zip(padded, label_imgs, subtitle_imgs)
        ):
            col = i % actual_cols
            row = i // actual_cols
            x = _BORDER + col * (_CELL_W + _BORDER)
            y = _QUERY_H + _BORDER + row * (cell_h_total + _BORDER)

            canvas.paste(lbl_img,     (x, y))
            canvas.paste(strip[f_idx], (x, y + _LABEL_H))
            canvas.paste(sub_img,     (x, y + _LABEL_H + _CELL_H))

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
    """Return *n* PIL Images at equally-spaced timestamps, resized to cell size."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    frames: list[Image.Image] = []
    for ts in np.linspace(start, end, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb).resize((_CELL_W, _CELL_H), Image.LANCZOS))

    cap.release()
    return frames


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _make_query_header(query: str, width: int, height: int) -> Image.Image:
    """Full-width dark-blue bar with the query text centred."""
    img = Image.new("RGB", (width, height), _QUERY_BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(14)
    text = f"Query: {query}" if query else "Query: —"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=_QUERY_FG, font=font)
    return img


def _make_label(text: str, width: int, height: int) -> Image.Image:
    """Small dark rank bar (e.g. '#1') centred in the cell."""
    img = Image.new("RGB", (width, height), _LABEL_BG)
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=_LABEL_FG, font=font)
    return img


def _make_subtitle(text: str, width: int, height: int) -> Image.Image:
    """Dark subtitle strip with wrapped text left-aligned with a small margin."""
    img = Image.new("RGB", (width, height), _SUB_BG)
    if not text:
        return img
    draw = ImageDraw.Draw(img)
    font = _load_font(11)
    wrapped = textwrap.fill(text, width=_SUB_WRAP_CHARS, max_lines=3, placeholder="…")
    draw.text((6, 5), wrapped, fill=_SUB_FG, font=font)
    return img

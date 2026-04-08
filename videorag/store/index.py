"""
videorag.store.index
~~~~~~~~~~~~~~
FAISS index construction, persistence and loading.

Indices and raw embeddings are saved under
``<output_root>/indices/`` so they can be reused across runs without
re-encoding every segment.

Public API
----------
build_index(embeddings)                         -> faiss.IndexFlatIP
save_index(index, path)
load_index(path)                                -> faiss.IndexFlatIP
save_embeddings(emb, path)
load_embeddings(path)                           -> np.ndarray
build_or_load_indices(segments_df, settings, bundle)
    -> (text_index, image_index, text_emb, image_emb)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
import pandas as pd

from videorag.config import Settings
from videorag.models.embeddings import (
    ModelBundle,
    generate_image_embeddings,
    generate_text_embeddings,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an inner-product (cosine for unit vectors) FAISS flat index.

    Args:
        embeddings: Float32 array ``(N, D)`` with unit-norm rows.

    Returns:
        ``faiss.IndexFlatIP`` loaded with all *N* vectors.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path | str) -> None:
    """Persist a FAISS index to disk."""
    faiss.write_index(index, str(path))


def load_index(path: Path | str) -> faiss.IndexFlatIP:
    """Load a previously persisted FAISS index from disk."""
    return faiss.read_index(str(path))


def save_embeddings(emb: np.ndarray, path: Path | str) -> None:
    """Save a numpy embedding matrix to disk (.npy)."""
    np.save(str(path), emb)


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load a numpy embedding matrix from disk."""
    return np.load(str(path))


# ---------------------------------------------------------------------------
# High-level: build or load both indices
# ---------------------------------------------------------------------------

def build_or_load_indices(
    segments_df: pd.DataFrame,
    settings: Settings,
    bundle: ModelBundle,
) -> Tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, np.ndarray, np.ndarray]:
    """
    Return ``(text_index, image_index, text_emb, image_emb)``.

    If persisted indices exist under ``<output_root>/indices/`` *and* the
    ``segments.csv`` timestamp has not changed since they were built, the
    cached files are loaded directly — skipping potentially slow embedding.

    Otherwise both embedding matrices are computed from scratch, saved to
    disk and loaded back.

    Args:
        segments_df: DataFrame produced by :func:`~videorag.data.preprocessing.run_preprocessing`.
        settings:    Project :class:`~videorag.config.Settings`.
        bundle:      Loaded :class:`~videorag.models.embeddings.ModelBundle`.

    Returns:
        Tuple of ``(text_index, image_index, text_embeddings, image_embeddings)``.
    """
    index_dir = settings.paths.output_root / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)

    text_idx_path = index_dir / "text_index.faiss"
    img_idx_path  = index_dir / "image_index.faiss"
    text_emb_path = index_dir / "text_embeddings.npy"
    img_emb_path  = index_dir / "image_embeddings.npy"
    stamp_path    = index_dir / "segments_mtime.txt"

    seg_path = settings.paths.output_root / "segments.csv"
    current_mtime = str(seg_path.stat().st_mtime) if seg_path.exists() else ""

    cached_paths = [text_idx_path, img_idx_path, text_emb_path, img_emb_path, stamp_path]
    cache_valid = (
        all(p.exists() for p in cached_paths)
        and stamp_path.read_text().strip() == current_mtime
    )

    if cache_valid:
        print("Loading cached FAISS indices…")
        text_index = load_index(text_idx_path)
        image_index = load_index(img_idx_path)
        text_emb = load_embeddings(text_emb_path)
        image_emb = load_embeddings(img_emb_path)
        print(
            f"✅ Loaded  text={text_index.ntotal}×{text_emb.shape[1]}d  "
            f"image={image_index.ntotal}×{image_emb.shape[1]}d"
        )
        return text_index, image_index, text_emb, image_emb

    # ── Build from scratch ──
    print("Building FAISS indices from scratch…")

    texts = segments_df["embed_text"].tolist()
    text_emb = generate_text_embeddings(
        texts, bundle, batch_size=settings.models.text_embed_batch_size
    )

    image_emb = generate_image_embeddings(segments_df["frames"].tolist(), bundle)

    text_index  = build_index(text_emb)
    image_index = build_index(image_emb)

    # Persist
    save_index(text_index, text_idx_path)
    save_index(image_index, img_idx_path)
    save_embeddings(text_emb, text_emb_path)
    save_embeddings(image_emb, img_emb_path)
    stamp_path.write_text(current_mtime)

    print(
        f"✅ Indices built and saved → {index_dir}\n"
        f"   text={text_index.ntotal}×{text_emb.shape[1]}d  "
        f"image={image_index.ntotal}×{image_emb.shape[1]}d"
    )
    return text_index, image_index, text_emb, image_emb

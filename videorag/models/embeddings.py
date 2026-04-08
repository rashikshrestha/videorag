"""
videorag.embeddings
~~~~~~~~~~~~~~~~~~~
Model loading, frame embedding and bulk embedding generation.

Public API
----------
load_models(settings)                      -> ModelBundle
_clip_single_frame(pil_img, bundle)        -> np.ndarray | None
emb_img_multi(frames_json, bundle)         -> np.ndarray | None
generate_text_embeddings(texts, bundle)    -> np.ndarray  (N, 384) float32
generate_image_embeddings(frames_col, bundle) -> np.ndarray (N, 512) float32
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from videorag.config import Settings


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

class ModelBundle(NamedTuple):
    """Lightweight container for the three model objects used throughout."""
    text_model: SentenceTransformer
    clip_model: CLIPModel
    clip_processor: CLIPProcessor
    device: str


def load_models(settings: Settings) -> ModelBundle:
    """
    Instantiate and return all required models.

    Models are moved to the device specified in *settings* and set to eval
    mode.  Call this once and pass the returned :class:`ModelBundle` to all
    functions that need it.

    Args:
        settings: Project settings (used for model names and device).

    Returns:
        :class:`ModelBundle` with ``text_model``, ``clip_model``,
        ``clip_processor`` and resolved ``device`` string.
    """
    device = settings.models.device

    text_model = SentenceTransformer(settings.models.text_model, device=device)

    clip_model = CLIPModel.from_pretrained(settings.models.clip_model)
    clip_model = clip_model.to(device).eval()  # type: ignore[attr-defined]

    clip_processor = CLIPProcessor.from_pretrained(settings.models.clip_model)

    print(f"✅ Models loaded  [device={device}]")
    return ModelBundle(
        text_model=text_model,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
    )


# ---------------------------------------------------------------------------
# Single-frame CLIP embedding
# ---------------------------------------------------------------------------

def _clip_single_frame(
    pil_img: Image.Image,
    bundle: ModelBundle,
) -> Optional[np.ndarray]:
    """
    Project one PIL image through the CLIP visual encoder.

    Returns a normalised (512,) float32 array, or ``None`` on failure.

    Args:
        pil_img: PIL Image in RGB mode.
        bundle:  Loaded :class:`ModelBundle`.
    """
    try:
        inp = bundle.clip_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            out = bundle.clip_model.vision_model(
                pixel_values=inp["pixel_values"].to(bundle.device)
            )
            feat = bundle.clip_model.visual_projection(out.pooler_output)
            feat = feat / torch.norm(feat, dim=-1, keepdim=True)
        return feat.cpu().numpy()[0].astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-frame CLIP embedding (scene level)
# ---------------------------------------------------------------------------

def emb_img_multi(
    frames_json: str,
    bundle: ModelBundle,
) -> Optional[np.ndarray]:
    """
    Embed all frames listed in *frames_json*, mean-pool into one vector.

    Why mean-pool for the index?
    The FAISS index is used for scene-level retrieval — mean-pooling gives
    the overall visual fingerprint of the full scene rather than a single
    potentially unrepresentative snapshot.

    (Per-frame max-pool is used in :func:`videorag.refinement.refine` for
    fine-grained action grounding.)

    Args:
        frames_json: JSON-encoded list of absolute frame file paths.
        bundle:      Loaded :class:`ModelBundle`.

    Returns:
        Normalised (512,) float32 vector, or ``None`` when no valid
        frames could be loaded or embedded.
    """
    try:
        paths = json.loads(frames_json)
    except Exception:
        return None

    vecs = []
    for p in paths:
        if p and Path(p).exists():
            try:
                img = Image.open(p).convert("RGB")
                v = _clip_single_frame(img, bundle)
                if v is not None:
                    vecs.append(v)
            except Exception:
                pass

    if not vecs:
        return None

    agg = np.mean(np.stack(vecs), axis=0)
    norm = np.linalg.norm(agg)
    return (agg / norm).astype(np.float32) if norm > 1e-8 else None


# ---------------------------------------------------------------------------
# Bulk embedding helpers
# ---------------------------------------------------------------------------

def generate_text_embeddings(
    texts: List[str],
    bundle: ModelBundle,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Batch-encode a list of strings via SentenceTransformer.

    BUG FIX #2: embed *pure* subtitle text — the original notebook prepended
    an episode/scene prefix that caused all vectors to cluster together,
    rendering retrieval nearly random.

    Args:
        texts:      List of strings to embed (one per segment).
        bundle:     Loaded :class:`ModelBundle`.
        batch_size: Encoding batch size.

    Returns:
        Float32 array of shape ``(len(texts), 384)`` with unit-norm rows.
    """
    print("Encoding text embeddings…")
    emb = bundle.text_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    result = emb.astype(np.float32)
    print(f"✅ Text embeddings: {result.shape}")
    return result


def generate_image_embeddings(
    frames_col: List[str],
    bundle: ModelBundle,
) -> np.ndarray:
    """
    Embed all scenes using :func:`emb_img_multi`, filling failed
    scenes with zero vectors (which will score near zero for any query).

    Args:
        frames_col: List of JSON-encoded frame path lists (one per segment).
        bundle:     Loaded :class:`ModelBundle`.

    Returns:
        Float32 array of shape ``(N, 512)`` with unit-norm rows
        (zero rows for missing/failed frames).
    """
    # Probe dimension
    img_dim = 512
    for frames_json in frames_col:
        v = emb_img_multi(frames_json, bundle)
        if v is not None:
            img_dim = v.shape[0]
            break
    else:
        raise RuntimeError(
            "Could not embed any frames — did preprocessing complete successfully?"
        )

    vecs = []
    failed = 0
    for frames_json in tqdm(frames_col, desc="Embedding image frames"):
        v = emb_img_multi(frames_json, bundle)
        if v is None:
            v = np.zeros(img_dim, dtype=np.float32)
            failed += 1
        vecs.append(v)

    result = np.stack(vecs).astype(np.float32)

    # Re-normalise (zero rows stay zero)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result /= norms

    print(f"✅ Image embeddings: {result.shape}  (failed: {failed}/{len(frames_col)})")
    return result

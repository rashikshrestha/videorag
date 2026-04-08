"""
videorag.config
~~~~~~~~~~~~~~~
Load and validate project settings from config.yaml.

All other modules import `Settings` via:
    from videorag.config import load_settings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PathSettings:
    video_root: Path
    subtitle_root: Optional[Path]
    output_root: Path


@dataclass
class PreprocessingSettings:
    n_frames: int
    frame_fractions: List[float]
    scene_threshold: int


@dataclass
class ModelSettings:
    text_model: str
    clip_model: str
    device: str          # resolved at load time; "cuda" or "cpu"
    text_embed_batch_size: int


@dataclass
class WeightSet:
    text: float
    image: float
    audio: float = 0.0


@dataclass
class RetrievalSettings:
    top_k: int
    merge_gap: float
    weights: dict              # {"action": WeightSet, "dialogue": WeightSet, "mixed": WeightSet}
    character_boost: float


@dataclass
class AudioSettings:
    enabled: bool
    sample_rate: int
    scene_audio_dir: str
    event_labels: List[str]
    top_k_events: int
    model: str


@dataclass
class RefinementSettings:
    expand: float
    bin_size: float
    stride: float
    min_span: float
    max_span: float
    bin_frames: int
    smooth_window: int
    snap_tolerance: float
    expand_max_extra: float


@dataclass
class PipelineSettings:
    calibrate_floor: float
    calibrate_ceiling: float
    hybrid_weight: float
    grounding_weight: float
    keyword_weight: float


@dataclass
class Settings:
    paths: PathSettings
    preprocessing: PreprocessingSettings
    models: ModelSettings
    audio: AudioSettings
    retrieval: RetrievalSettings
    refinement: RefinementSettings
    pipeline: PipelineSettings
    characters: List[str]


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """
    Load settings from a YAML file and return a validated ``Settings`` object.

    Args:
        config_path: Path to ``config.yaml``. Relative paths are resolved
                     with respect to the current working directory.

    Returns:
        Fully populated :class:`Settings` instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Copy config.yaml to the project root and update the paths."
        )

    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    paths_raw = raw.get("paths", {})
    preprocessing_raw = raw.get("preprocessing", {})
    models_raw = raw.get("models", {})
    retrieval_raw = raw.get("retrieval", {})
    refinement_raw = raw.get("refinement", {})
    pipeline_raw = raw.get("pipeline", {})

    # ── Resolve device ──
    device_cfg = models_raw.get("device", "auto")
    if device_cfg == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = device_cfg

    # ── Retrieval weights ──
    default_weights = {
        "action": {"text": 0.30, "image": 0.50, "audio": 0.20},
        "dialogue": {"text": 0.70, "image": 0.20, "audio": 0.10},
        "mixed": {"text": 0.45, "image": 0.35, "audio": 0.20},
    }
    weights_raw = retrieval_raw.get("weights", default_weights)
    weights = {
        qt: WeightSet(text=float(v["text"]), image=float(v["image"]), audio=float(v.get("audio", 0.0)))
        for qt, v in weights_raw.items()
    }

    # Ensure all query types exist and weights are normalised.
    for qt in ("action", "dialogue", "mixed"):
        ws = weights.get(qt, WeightSet(text=0.5, image=0.5, audio=0.0))
        total = float(ws.text + ws.image + ws.audio)
        if total <= 1e-8:
            weights[qt] = WeightSet(text=0.5, image=0.5, audio=0.0)
        else:
            weights[qt] = WeightSet(
                text=float(ws.text / total),
                image=float(ws.image / total),
                audio=float(ws.audio / total),
            )

    audio_raw = raw.get("audio", {})
    default_event_labels = [
        "gunshot",
        "glass breaking",
        "scream",
        "explosion",
        "applause",
        "laughter",
        "door slam",
        "car horn",
        "footsteps",
        "sirens",
        "phone ringing",
        "dog barking",
    ]

    return Settings(
        paths=PathSettings(
            video_root=Path(paths_raw.get("video_root", "data/videos")),
            subtitle_root=(
                Path(paths_raw.get("subtitle_root"))
                if paths_raw.get("subtitle_root") is not None
                else None
            ),
            output_root=Path(paths_raw.get("output_root", "data/output")),
        ),
        preprocessing=PreprocessingSettings(
            n_frames=int(preprocessing_raw.get("n_frames", 5)),
            frame_fractions=[float(x) for x in preprocessing_raw.get("frame_fractions", [0.10, 0.30, 0.50, 0.70, 0.90])],
            scene_threshold=int(preprocessing_raw.get("scene_threshold", 27)),
        ),
        models=ModelSettings(
            text_model=models_raw.get("text_model", "sentence-transformers/all-MiniLM-L6-v2"),
            clip_model=models_raw.get("clip_model", "openai/clip-vit-base-patch32"),
            device=device,
            text_embed_batch_size=int(models_raw.get("text_embed_batch_size", 256)),
        ),
        audio=AudioSettings(
            enabled=bool(audio_raw.get("enabled", False)),
            sample_rate=int(audio_raw.get("sample_rate", 16000)),
            scene_audio_dir=str(audio_raw.get("scene_audio_dir", "audio/scenes")),
            event_labels=[str(x) for x in audio_raw.get("event_labels", default_event_labels)],
            top_k_events=int(audio_raw.get("top_k_events", 5)),
            model=str(audio_raw.get("model", "laion/clap-htsat-unfused")),
        ),
        retrieval=RetrievalSettings(
            top_k=int(retrieval_raw.get("top_k", 10)),
            merge_gap=float(retrieval_raw.get("merge_gap", 20.0)),
            weights=weights,
            character_boost=float(retrieval_raw.get("character_boost", 0.3)),
        ),
        refinement=RefinementSettings(
            expand=float(refinement_raw.get("expand", 15.0)),
            bin_size=float(refinement_raw.get("bin_size", 2.0)),
            stride=float(refinement_raw.get("stride", 0.5)),
            min_span=float(refinement_raw.get("min_span", 10.0)),
            max_span=float(refinement_raw.get("max_span", 35.0)),
            bin_frames=int(refinement_raw.get("bin_frames", 3)),
            smooth_window=int(refinement_raw.get("smooth_window", 5)),
            snap_tolerance=float(refinement_raw.get("snap_tolerance", 3.0)),
            expand_max_extra=float(refinement_raw.get("expand_max_extra", 8.0)),
        ),
        pipeline=PipelineSettings(
            calibrate_floor=float(pipeline_raw.get("calibrate_floor", 0.10)),
            calibrate_ceiling=float(pipeline_raw.get("calibrate_ceiling", 0.70)),
            hybrid_weight=float(pipeline_raw.get("hybrid_weight", 0.50)),
            grounding_weight=float(pipeline_raw.get("grounding_weight", 0.40)),
            keyword_weight=float(pipeline_raw.get("keyword_weight", 0.10)),
        ),
        characters=[str(c) for c in raw.get("characters", [])],
    )

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
from typing import List

import yaml


@dataclass
class PathSettings:
    video_root: Path
    subtitle_root: Path
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
        raw = yaml.safe_load(fh)

    # ── Resolve device ──
    import torch
    device_cfg = raw.get("models", {}).get("device", "auto")
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    # ── Retrieval weights ──
    weights_raw = raw.get("retrieval", {}).get("weights", {})
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
            video_root=Path(raw["paths"]["video_root"]),
            subtitle_root=Path(raw["paths"]["subtitle_root"]),
            output_root=Path(raw["paths"]["output_root"]),
        ),
        preprocessing=PreprocessingSettings(
            n_frames=raw["preprocessing"]["n_frames"],
            frame_fractions=raw["preprocessing"]["frame_fractions"],
            scene_threshold=raw["preprocessing"]["scene_threshold"],
        ),
        models=ModelSettings(
            text_model=raw["models"]["text_model"],
            clip_model=raw["models"]["clip_model"],
            device=device,
            text_embed_batch_size=raw["models"].get("text_embed_batch_size", 256),
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
            top_k=raw["retrieval"]["top_k"],
            merge_gap=float(raw["retrieval"]["merge_gap"]),
            weights=weights,
            character_boost=float(raw["retrieval"]["character_boost"]),
        ),
        refinement=RefinementSettings(
            expand=float(raw["refinement"]["expand"]),
            bin_size=float(raw["refinement"]["bin_size"]),
            stride=float(raw["refinement"]["stride"]),
            min_span=float(raw["refinement"]["min_span"]),
            max_span=float(raw["refinement"]["max_span"]),
            bin_frames=int(raw["refinement"]["bin_frames"]),
            smooth_window=int(raw["refinement"]["smooth_window"]),
            snap_tolerance=float(raw["refinement"]["snap_tolerance"]),
            expand_max_extra=float(raw["refinement"]["expand_max_extra"]),
        ),
        pipeline=PipelineSettings(
            calibrate_floor=float(raw["pipeline"]["calibrate_floor"]),
            calibrate_ceiling=float(raw["pipeline"]["calibrate_ceiling"]),
            hybrid_weight=float(raw["pipeline"]["hybrid_weight"]),
            grounding_weight=float(raw["pipeline"]["grounding_weight"]),
            keyword_weight=float(raw["pipeline"]["keyword_weight"]),
        ),
        characters=[str(c) for c in raw.get("characters", [])],
    )

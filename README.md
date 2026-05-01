# 🎞️ VideoRAG: Multimodal RAG Retrieval System for Video Grounding
Given a natural-language query, the system retrieves the most relevant video scenes using multimodal information: audio, video(image sequence) and subtitles.


## Overview

VideoRAG combines:
- **Scene Detection**: Automatic detection of scene boundaries using content analysis
- **Multi-frame Keyframe Extraction**: Sampling 5 frames per scene to maximize visual coverage
- **Text Embedding**: Scene subtitles embedded with SentenceTransformer (384-dim)
- **Multimodal Embedding**: Scene keyframes embedded with CLIP (512-dim shared space)
- **Dual-Index FAISS Retrieval**: Hybrid search across text and image modalities
- **Temporal Refinement**: Fine-grained grounding using subtitle alignment, keyword expansion, and multi-frame visual scoring

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (for audio extraction and embedded subtitle support)
- CUDA-enabled GPU (optional, for faster embedding generation)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd videorag
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   python -c "from videorag.api import build_context; print('✅ OK')"
   ```

## Preprocessing Pipeline

```bash
python scripts/run_pipeline.py preprocess --config config/preprocessing.yaml
```
Initial step to preprocess raw video files and prepare them for embedding and retrieval.

- Scene Detection: Uses ContentDetector with tunable threshold (default 27) to identify scene boundaries
- Keyframe Extraction: Samples 5 frames per scene at positions 10%, 30%, 50%, 70%, 90% to ensure visual coverage
- Subtitle Collection: Gathers all overlapping subtitle lines within each scene into single text field
- Audio Extraction (optional): Extracts WAV clips per scene at 16 kHz if enabled
- Metadata Assembly: Writes one CSV row per scene with 10 columns (video, scene_id, start, end, duration, subtitle, frames, audio_path, audio_events, audio_event_text)

#### Output Columns of segments.csv

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `video` | str | `"episode_1.mkv"` | Source video filename |
| `scene_id` | int | `0`, `1`, `2` | Sequential scene index within the video |
| `start` | float | `12.345` | Scene start time (seconds) |
| `end` | float | `67.890` | Scene end time (seconds) |
| `duration` | float | `55.545` | Duration of scene (end - start) |
| `subtitle` | str | `"Ross and Rachel argue about the list"` | All subtitle text in the scene |
| `frames` | str (JSON) | `"["/path/to/f0.jpg", ..., "/path/to/f4.jpg"]"` | List of 5 extracted frame paths |
| `audio_path` | str | `"/output/audio/scenes/episode_1/0.wav"` | Path to extracted audio (empty if audio disabled) |
| `audio_events` | str (JSON) | `"[]"` | Top-k sound event labels _(filled during indexing)_ |
| `audio_event_text` | str | `""` | Space-joined audio event descriptions _(filled during indexing)_ |


## Build FAISS Indices
```bash
python scripts/run_pipeline.py index --config config/pipeline.yaml
```
Output: Dual FAISS indices (text and image) saved to `<output_root>/indices/`.

Use `--force` to rebuild even if cached indices already exist:
```bash
python scripts/run_pipeline.py index --config config/pipeline.yaml --force
```

## Ground a Query

Run a natural-language query against the indexed video corpus:

```bash
python scripts/run_pipeline.py query \
    --config config/pipeline.yaml \
    --text "Rachel looses her Ring" \
    --top-k 5 \
    --show-top 5 \
    --merge-gap 20.0
```

### Query Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | _(required)_ | Natural-language query string |
| `--top-k` | `10` | Number of candidate scenes to retrieve before refinement |
| `--show-top` | `3` | Number of results to display |
| `--merge-gap` | `20.0` | Max gap in seconds between spans to merge into one result |

### Example Output

```
QUERY: Ross and Rachel argue
============================

[1] Friends.S01E19.The.One.Where.the.Monkey.Gets.Away.mkv  scene=311  type=dialogue
    SCENE:    17:42.85 → 17:53.86
    GROUNDED: 17:36.01 → 18:00.74  (span=24.7s)
    CONF:     100.00%  (ret=1.169 gnd=0.948 kw=0.030)
    SUB: I mean, you're off in Rachel Land... totally oblivious to people's feelings...
```

Each result includes:
- **SCENE**: raw scene boundary from preprocessing
- **GROUNDED**: refined temporal span after subtitle alignment and visual scoring
- **CONF**: confidence score broken down into retrieval (`ret`), grounding (`gnd`), and keyword (`kw`) components

## Evaluate on Gold Queries
```bash
python scripts/run_pipeline.py evaluate --config config/pipeline.yaml
```
Output: Metrics (Top-1 accuracy, IoU, recall@IoU≥0.5) on 17 built-in queries. Results saved to `<output_root>/results/`.

## Run the Full Pipeline
```bash
python scripts/run_pipeline.py run-all --config config/pipeline.yaml
```
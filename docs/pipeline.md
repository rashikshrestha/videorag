# VideoRAG Pipeline: Preprocessing, Indexing, and Querying

This document describes the theoretical design of the preprocessing, indexing, and query stages in VideoRAG. The first two stages transform raw video files into a set of searchable, multi-modal vector indices; the query stage uses those indices to locate the most relevant temporal span in response to a natural-language question.

---

## Overview

The pipeline processes video content at the **scene level**. Each detected scene becomes an atomic unit of retrieval, bundling three modalities — visual frames, dialogue text, and audio — into a single row that is later embedded and indexed. The overall flow is:

```
Raw Videos
    → Scene Detection
    → Per-Scene Extraction (frames, subtitles, audio)
    → Embedding Generation (text, visual, audio)
    → FAISS Index Construction
```

---

## Stage 1: Preprocessing

### Scene Detection

The first step partitions each video into semantically coherent scenes using a content-based change detector. The detector computes frame-to-frame differences in colour histogram and motion statistics; when the difference exceeds a configurable threshold, a scene boundary is recorded. A lower threshold produces finer-grained scenes; a higher threshold merges visually similar stretches into a single scene.

Each scene is represented by its start and end timestamps (in seconds).

### Keyframe Extraction

Rather than representing a scene by a single frame, the pipeline samples multiple frames spread across the scene's duration. Frame positions are expressed as fractions of the scene length, deliberately avoiding the very beginning and end of a scene where black or transition frames are common.

The number of sampled frames is controlled by a single parameter (`n_frames`), from which evenly-spaced fractional positions are derived automatically. Sampling multiple frames per scene is important because scenes often contain heterogeneous visual content — a single frame would miss events that occur mid-scene.

Extracted frames are stored as JPEG images on disk, organised by video and scene.

### Subtitle Collection

The pipeline supports three subtitle sources, tried in order:

1. **External subtitle files** matched to the video by filename.
2. **Fuzzy filename matching** when the names do not align exactly.
3. **Embedded subtitles** extracted directly from the video container as a fallback.

Once a subtitle track is located, all lines whose timestamps overlap the scene's `[start, end]` interval are collected and joined into a single string representing the dialogue for that scene. Scenes with no overlapping dialogue receive an empty subtitle field.

### Audio Extraction

When audio processing is enabled, FFmpeg extracts a mono WAV clip for each scene, resampled to a fixed sample rate (16 kHz by default). These clips are stored on disk and referenced in the scene metadata. They serve two purposes: generating audio embeddings for the index, and providing audio content for fine-grained temporal refinement at query time.

### Output: The Segments Table

At the end of preprocessing, all scene-level data is consolidated into a tabular structure (`segments.csv`) with one row per scene. Each row holds the source video name, scene identifier, start and end timestamps, duration, subtitle text, paths to the extracted frame images, and the path to the audio clip. This table is the single source of truth consumed by all downstream stages.

---

## Stage 2: Embedding Generation

Each scene is projected into three separate vector spaces — one per modality. All resulting vectors are L2-normalised to unit length, enabling cosine similarity to be computed via simple inner products.

### Text Embeddings

The subtitle text for each scene is encoded by a **SentenceTransformer** model. SentenceTransformer produces a fixed-dimensional dense vector that captures semantic meaning, allowing scenes with dialogue similar in meaning (not just lexically) to a query to score highly. Scenes with no dialogue are represented by a generic placeholder embedding rather than a zero vector.

Embedding dimension: **384**.

### Visual Embeddings

Each extracted frame is independently encoded by the vision encoder of a **CLIP** model. CLIP maps images into a shared image-text embedding space, which means a textual query about a visual concept (e.g., "two people hugging") can be directly compared against frame embeddings.

Because multiple frames are extracted per scene, the per-frame embeddings are **mean-pooled** into a single scene-level vector and re-normalised. Mean pooling provides a stable aggregate representation of the scene's visual content across its duration.

Embedding dimension: **512**.

### Audio Embeddings

Each scene's WAV clip is encoded by a **CLAP** (Contrastive Language-Audio Pretraining) model. CLAP is the audio analogue of CLIP: it projects both audio waveforms and free-form text descriptions into a shared embedding space. This means a textual query about a sound event (e.g., "crowd laughing") can be matched against audio embeddings.

**Audio event labelling** is also performed at this stage. A fixed vocabulary of sound event labels (e.g., laughter, applause, phone ringing) is encoded via the CLAP text encoder. For each scene, cosine similarities between the scene's audio embedding and all event-label embeddings are computed, and the top-k most similar labels are recorded back into the segments table. These labels become part of the scene's metadata and are used to enrich retrieval.

Embedding dimension: **512**.

---

## Stage 3: Indexing

### FAISS Flat Indices

Three separate FAISS indices are built — one per modality:

- **Text index**: indexes the SentenceTransformer scene embeddings (384-dim).
- **Image index**: indexes the mean-pooled CLIP visual scene embeddings (512-dim).
- **Audio index**: indexes the CLAP audio scene embeddings (512-dim).

Each index uses `IndexFlatIP` (flat inner-product search), which performs exact exhaustive nearest-neighbour search. Since all vectors are unit-normalised, inner product is equivalent to cosine similarity. Flat indices are chosen over approximate indices because the dataset size (hundreds to low thousands of scenes) does not require approximation, and exact search avoids recall loss.

The row ordering within each index corresponds directly to the row ordering in the segments table, so a retrieved index position maps unambiguously back to a specific scene.

### Embedding Matrices

In addition to the FAISS indices, the raw embedding matrices are persisted as NumPy arrays. These allow downstream components (such as the temporal refinement stage) to perform direct vector arithmetic on scene embeddings without re-encoding.

### Index Manifest and Caching

To avoid redundant recomputation, an index manifest records the modification time of the segments table and the audio configuration at the time the indices were built. On subsequent runs, if neither has changed, the pre-built indices and embeddings are loaded from disk directly. If the segments table has been updated (e.g., new videos were preprocessed) or the audio configuration has changed, the indices are rebuilt from scratch.

---

## Design Principles

**Scene as retrieval unit.** Scenes detected by content change are semantically coherent and form a natural granularity for retrieval. They are short enough to be specific but long enough to contain meaningful visual and dialogue content.

**Multi-frame visual aggregation.** Representing a scene by several frames reduces the risk of missing key visual events that occur mid-scene. Mean pooling over frames produces a stable aggregate while keeping index size constant regardless of scene length.

**Separate indices per modality.** Maintaining independent FAISS indices for text, visual, and audio enables query-time flexibility: the contribution of each modality can be weighted differently depending on the nature of the query (action-oriented, dialogue-oriented, or mixed), without any coupling at the index level.

**Shared embedding spaces.** The use of CLIP and CLAP ensures that text queries can be directly compared to visual and audio embeddings respectively, without requiring image-to-text transcription or audio-to-text transcription as intermediary steps.

**Unit normalisation.** Normalising all embeddings to unit length before indexing ensures that inner-product search is equivalent to cosine similarity, providing consistent, magnitude-independent similarity scoring across all modalities.

---

## Stage 4: Querying

A query is a natural-language string describing a moment in the video (e.g., "Ross and Rachel argue about the list" or "Monica opens the door"). The query stage maps this string to a ranked list of refined temporal spans, each with an associated confidence score.

### Query Classification

Before any search is performed, the query is classified into one of three types — **action**, **dialogue**, or **mixed** — by matching its words against two fixed vocabularies: one for action-oriented terms (physical events, movement, gestures) and one for dialogue-oriented terms (speech acts, conversations, verbal exchanges). Whichever vocabulary produces more matches determines the query type; ties default to mixed.

The classification drives the fusion weights used throughout retrieval. Action queries emphasise visual similarity; dialogue queries emphasise text similarity; mixed queries balance both. Audio weight applies uniformly across types when the audio index is available.

### Query Encoding

The query string is simultaneously encoded by all three model encoders used during indexing:

- The **SentenceTransformer** encodes the query into the same 384-dim semantic text space used for subtitle embeddings.
- The **CLIP text encoder** encodes the query into the 512-dim visual-language space shared with the frame embeddings.
- The **CLAP text encoder** encodes the query into the 512-dim audio-language space shared with the audio embeddings.

This produces three query vectors, each compatible with one of the FAISS indices, enabling cross-modal retrieval without any modality-specific query reformulation.

### Hybrid Search

All three indices are searched exhaustively against their respective query vectors, producing a raw cosine similarity score for every scene in the dataset. Scores from different modalities are not directly comparable in magnitude, so each modality's score vector is independently **min-max normalised** to the range [0, 1] across all scenes before fusion. This prevents any single modality from dominating due to scale differences.

The normalised scores are combined by a weighted sum:

```
fused_score = α × text_score + β × image_score + γ × audio_score
```

where α, β, γ are the query-type-specific weights. When the audio index is unavailable, the text and image weights are renormalised to sum to 1 so the scale of the fused score remains consistent.

### Character Boosting

If the query mentions character names present in the subtitle vocabulary, scenes whose subtitle text contains those same names receive an additive boost to their fused score. The boost is proportional to the fraction of queried characters that appear in the scene, rewarding scenes that directly involve the relevant characters. This acts as a lightweight lexical signal layered on top of the semantic retrieval.

### Candidate Selection

After fusion and character boosting, scenes are ranked by their fused score and the top-k candidates are retained. These candidates are coarse-grained: each one corresponds to a full detected scene, which may span tens of seconds. The next stage narrows each candidate to a precise sub-span.

---

## Stage 5: Temporal Refinement

Temporal refinement takes each retrieved scene candidate and locates the precise sub-interval within (and slightly around) it that best matches the query. This converts scene-level retrieval into fine-grained temporal grounding.

### Expansion Window

The candidate scene's boundaries are extended by a fixed margin in both directions to account for relevant content that may have been split across adjacent scenes by the scene detector. This produces an expanded search window within which bin scoring operates.

### Sliding Bin Scoring

The expanded window is divided into short, overlapping temporal bins (default: 2-second bins, sliding at 0.5-second stride). Each bin is scored independently across all three modalities:

**Subtitle score.** Subtitle lines overlapping the bin are collected and jointly encoded by the SentenceTransformer. The bin score is a weighted combination of semantic similarity (cosine similarity between the bin's subtitle embedding and the query's text embedding) and keyword overlap (the fraction of meaningful query words that appear verbatim in the subtitle). Semantic similarity captures paraphrased dialogue; keyword overlap provides a precise lexical anchor.

**Visual score.** A small number of frames are sampled from within the bin and individually encoded by the CLIP vision encoder. Each frame's cosine similarity with the CLIP query embedding is computed. For action queries, the maximum similarity across frames is taken — one peak frame capturing the key event is sufficient. For other query types, the mean is taken — stable appearance across the bin is preferred.

**Audio score.** If audio is enabled, a short audio clip is extracted for the bin and encoded by the CLAP audio encoder. Its cosine similarity with the CLAP query embedding provides an audio-domain relevance signal.

The three per-bin scores are normalised within the window and fused with the same query-type weights used during hybrid search.

### Score Smoothing and Peak Expansion

The sequence of fused bin scores forms a temporal relevance curve over the expanded window. A moving-average filter is applied to this curve to suppress noise from individual outlier bins. The smoothed peak — the bin with the highest score — identifies the most relevant moment.

The algorithm then expands outward from the peak in both directions, continuing as long as the score remains above a threshold relative to the peak value. This produces a raw span that captures the full extent of the relevant moment rather than just the single best bin.

### Span Constraints and Boundary Snapping

The raw span is adjusted to satisfy minimum and maximum duration constraints, padding or re-centring as needed. The edges are then **snapped** to nearby subtitle line boundaries: if a subtitle line starts or ends within a small tolerance of the span edge, the span is extended to align with it. This prevents the span from cutting a line of dialogue in half, which would reduce the intelligibility of retrieved content.

Finally, the span is further extended to include any subtitle lines within a short look-ahead window that contain meaningful query keywords. This keyword-guided expansion ensures that relevant lines just outside the initial span are incorporated.

---

## Stage 6: Score Fusion and Result Assembly

### Final Score Fusion

Each candidate produces two independent confidence signals: the hybrid search score (a retrieval-stage measure of how well the scene matched the query across all modalities) and the refinement confidence (the smoothed peak score from the bin scoring curve). A small lexical keyword bonus is also computed from the overlap between query terms and the final span's subtitle content. These three signals are combined by a fixed weighted sum into a single raw score.

### Confidence Calibration

The raw score is mapped to a calibrated confidence value in [0, 1] by linear rescaling between a configured floor and ceiling. Scores below the floor map to 0; scores above the ceiling map to 1; scores in between are linearly interpolated. Calibration ensures that the confidence value reflects a consistent notion of retrieval quality across different queries and datasets, rather than being sensitive to the absolute scale of the raw scores.

### Span Merging

When multiple candidates from the same video have refined spans that are close together (gap within a configured threshold), they are merged into a single span covering their union. The merged span inherits the metadata and score of the highest-scoring constituent. This avoids returning near-duplicate results for queries whose relevant content spans more than one originally detected scene.

---

## End-to-End Design Principles

**Coarse-to-fine retrieval.** Scene-level hybrid search provides a computationally cheap shortlist; bin-level temporal refinement then performs expensive per-frame encoding only on a small window around each candidate. This two-stage design keeps query latency tractable.

**Query-adaptive modality weighting.** Rather than treating all queries identically, the pipeline classifies each query and adjusts the balance between text, visual, and audio signals accordingly. A query about a physical action is answered primarily by visual evidence; a query about a conversation is answered primarily by subtitle evidence.

**Lexical signals as complements.** Pure embedding-based retrieval can miss exact name or keyword matches due to the compression inherent in dense vectors. Character boosting and keyword-based span expansion inject direct lexical signals at the points in the pipeline where they are most useful, without replacing semantic search.

**Temporal coherence over raw ranking.** The goal is not just to find the right scene but to return a span that a viewer could watch and understand. Subtitle snapping, keyword expansion, and span merging all serve this goal by ensuring the returned interval is aligned with natural dialogue and narrative boundaries.

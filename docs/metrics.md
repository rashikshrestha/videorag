# Evaluation Metrics

## Summary

The following metrics are used to evaluate temporal video grounding — the task of retrieving the correct time segment(s) from a video given a natural-language query.

| Metric | What it measures | Higher/Lower better |
|--------|-----------------|---------------------|
| **Mean IoU @ 1** | Temporal overlap between the top-ranked candidate and the nearest ground-truth segment | Higher |
| **Mean IoU @ K** | Best temporal overlap among the top-K candidates vs any ground-truth segment | Higher |
| **Recall @ 1 / IoU ≥ θ** | Fraction of queries where the top-ranked candidate overlaps a GT segment above threshold θ | Higher |
| **Recall @ K / IoU ≥ θ** | Fraction of queries where any top-K candidate overlaps a GT segment above threshold θ | Higher |
| **Precision @ K / IoU ≥ θ** | Fraction of top-K candidates (averaged over queries) that overlap a GT segment above threshold θ | Higher |
| **Mean Center Displacement** | Average distance (seconds) between the predicted segment's center and the nearest GT center | Lower |

Thresholds used: **IoU ≥ 0.3** (loose), **IoU ≥ 0.5** (standard), **IoU ≥ 0.7** (tight).

Each metric is reported in two flavours:
- **@ 1** — based on the single top-scored candidate only (strict: tests ranking quality)
- **@ K** — based on the top-K candidates (lenient: tests whether a correct answer exists in the shortlist)

If **R@K** is high but **R@1** is low, the model retrieves relevant segments but ranks them poorly.
If **Precision@K** is low despite high **R@K**, most of the shortlist is noise and only one candidate is relevant.
**Center Displacement** catches cases where IoU is low not because the wrong scene was found, but because the segment boundaries are slightly off.

---

## IoU (Intersection over Union)

IoU measures **temporal overlap** between a predicted segment and a ground-truth segment:

```
IoU = overlap duration / total covered duration
```

**Example:** predicted `[10, 30]`, ground truth `[15, 35]`
- Intersection = `[15, 30]` = 15s
- Union = `[10, 35]` = 25s
- IoU = 15/25 = **0.6**

IoU answers: *"How well does the predicted time window align with the ground truth window?"*

> IoU is **not** the same as Precision. Precision asks "of the things I retrieved, what fraction are correct?" — it is about retrieving the right items. IoU is about how accurately a retrieved item is localized in time.

---

## R@1 — Recall at 1

Uses only the **top-scored candidate**. Computes IoU against all GT segments and keeps the best.

**Example** with 4 GT segments:

```
Candidate 1  vs  GT1=0.2,  GT2=0.6,  GT3=0.1,  GT4=0.0  →  iou_at_1 = 0.6
```

R@1 answers: *"Does your single best guess overlap a ground-truth segment?"*

---

## R@K — Recall at K

Uses the **top-K candidates**. For each candidate, computes IoU against all GT segments and keeps the best. Then takes the max across all K candidates.

**Example** with 4 GT segments and K=5:

```
Candidate 1  vs  GT1=0.2,  GT2=0.6,  GT3=0.1,  GT4=0.0  →  best = 0.6
Candidate 2  vs  GT1=0.0,  GT2=0.1,  GT3=0.8,  GT4=0.3  →  best = 0.8
Candidate 3  vs  GT1=0.4,  GT2=0.0,  GT3=0.2,  GT4=0.1  →  best = 0.4
Candidate 4  vs  GT1=0.0,  GT2=0.0,  GT3=0.1,  GT4=0.9  →  best = 0.9
Candidate 5  vs  GT1=0.1,  GT2=0.2,  GT3=0.0,  GT4=0.0  →  best = 0.2

iou_at_k = max(0.6, 0.8, 0.4, 0.9, 0.2) = 0.9
```

R@K answers: *"Does any of your top-K candidates overlap a ground-truth segment?"*

---

## Comparison

| Metric | Candidates used | Strictness |
|--------|----------------|------------|
| R@1    | Only #1        | Strict — model must rank the right segment first |
| R@K    | Top K          | Lenient — segment just needs to appear in top K |

If R@K is high but R@1 is low, the model retrieves relevant segments but ranks them poorly.

---

## IoU Thresholds

Recall is reported at three thresholds:

| Threshold | Meaning |
|-----------|---------|
| IoU ≥ 0.3 | Loose overlap |
| IoU ≥ 0.5 | Moderate overlap (common standard) |
| IoU ≥ 0.7 | Tight overlap |

A query "hits" a threshold if its best IoU meets or exceeds it. The reported value is the **fraction of queries** that hit the threshold.

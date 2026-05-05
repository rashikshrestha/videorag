#!/usr/bin/env python
"""
scripts/combine_results.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads all eval JSON files in data/vis/ whose names start with
'single_concept', 'two_concepts', or 'three_concepts', groups them
by concept type, and for each group computes per-query:

  average — mean of each metric across all alpha/beta configs
  best    — best value per metric (max for IoU/precision, min for displacement)

Aggregates across queries are computed for both average and best.
Outputs saved to data/combined_vis/<concept_type>_combined.json.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

VIS_DIR  = Path("data/vis")
OUT_DIR  = Path("data/combined_vis")

PREFIXES = ["single_concept", "two_concepts", "three_concepts"]

FILENAME_RE = re.compile(
    r"^(single_concept|two_concepts|three_concepts)_(\d+\.\d+)_(\d+\.\d+)_(\d+\.\d+)\.json$"
)

HIGHER_BETTER   = ["iou_at_1", "iou_at_k"]
LOWER_BETTER    = ["disp_at_1", "disp_at_k"]
PREC_THRESHOLDS = ["0.3", "0.5", "0.7"]


def load_files_by_prefix() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(VIS_DIR.glob("*.json")):
        m = FILENAME_RE.match(path.name)
        if m:
            groups[m.group(1)].append(path)
    return groups


def collect_per_query(files: list[Path]) -> dict[str, list[dict]]:
    query_data: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        data = json.loads(path.read_text())
        for row in data["queries"]:
            query_data[row["query"]].append(row)
    return query_data


def per_query_stats(rows: list[dict]) -> tuple[dict, dict]:
    avg: dict  = {}
    best: dict = {}

    for key in HIGHER_BETTER:
        vals = [r[key] for r in rows]
        avg[key]  = round(sum(vals) / len(vals), 4)
        best[key] = round(max(vals), 4)

    for key in LOWER_BETTER:
        vals = [r[key] for r in rows]
        avg[key]  = round(sum(vals) / len(vals), 2)
        best[key] = round(min(vals), 2)

    avg_prec: dict  = {}
    best_prec: dict = {}
    for thr in PREC_THRESHOLDS:
        vals = [r["prec_at_k"][thr] for r in rows]
        avg_prec[thr]  = round(sum(vals) / len(vals), 4)
        best_prec[thr] = round(max(vals), 4)
    avg["prec_at_k"]  = avg_prec
    best["prec_at_k"] = best_prec

    return avg, best


def aggregate_queries(queries: list[dict], mode: str) -> dict:
    n   = len(queries)
    agg: dict = {}

    for key in HIGHER_BETTER:
        vals = [q[mode][key] for q in queries]
        agg[f"mean_{key}"] = round(sum(vals) / n, 4)

    for key in LOWER_BETTER:
        vals = [q[mode][key] for q in queries]
        agg[f"mean_{key}"] = round(sum(vals) / n, 2)

    for thr in PREC_THRESHOLDS:
        vals = [q[mode]["prec_at_k"][thr] for q in queries]
        thr_key = thr.replace(".", "")
        agg[f"mean_prec_at_k_iou{thr_key}"] = round(sum(vals) / n, 4)

    return agg


def process_group(prefix: str, files: list[Path]) -> None:
    query_data = collect_per_query(files)

    result_queries = []
    for query, rows in sorted(query_data.items()):
        avg, best = per_query_stats(rows)
        result_queries.append({
            "query":   query,
            "n_files": len(rows),
            "average": avg,
            "best":    best,
        })

    output = {
        "concept_type": prefix,
        "n_files":      len(files),
        "n_queries":    len(result_queries),
        "aggregate": {
            "average": aggregate_queries(result_queries, "average"),
            "best":    aggregate_queries(result_queries, "best"),
        },
        "queries": result_queries,
    }

    out_path = OUT_DIR / f"{prefix}_combined.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved → {out_path}")

    agg = output["aggregate"]
    print(f"  Aggregate of averages:")
    for k, v in agg["average"].items():
        print(f"    {k:<35} {v}")
    print(f"  Aggregate of bests:")
    for k, v in agg["best"].items():
        print(f"    {k:<35} {v}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = load_files_by_prefix()
    if not groups:
        raise SystemExit(f"No matching JSON files found in {VIS_DIR}")

    for prefix in PREFIXES:
        files = groups.get(prefix, [])
        if not files:
            print(f"\n[SKIP] No files for {prefix}")
            continue
        print(f"\n{'='*55}")
        print(f"  {prefix}  ({len(files)} files)")
        print(f"{'='*55}")
        process_group(prefix, files)


if __name__ == "__main__":
    main()

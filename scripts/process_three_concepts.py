import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import yaml

try:
    from rapidfuzz import fuzz
    def similarity(a: str, b: str) -> float:
        return fuzz.ratio(a.lower(), b.lower())
except ImportError:
    from difflib import SequenceMatcher
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100


SIMILARITY_THRESHOLD = 88  # percent


def fuzzy_cluster(concepts: list[str]) -> dict[str, str]:
    """Map each concept to a canonical (first-seen) representative."""
    clusters: list[tuple[str, list[str]]] = []
    mapping: dict[str, str] = {}

    for concept in concepts:
        for canon, members in clusters:
            if similarity(concept, canon) >= SIMILARITY_THRESHOLD:
                mapping[concept] = canon
                members.append(concept)
                break
        else:
            clusters.append((concept, [concept]))
            mapping[concept] = concept

    return mapping


def main():
    input_path = Path("data/queries.csv")
    output_path = Path("data/queries_three_concepts.yaml")

    all_rows: list[tuple[str, str, str, list[str]]] = []
    concept_order: list[str] = []
    seen_concepts: set[str] = set()

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode = row["episode"].strip()
            start = row["start"].strip()
            end = row["end"].strip()
            concepts = [c.strip() for c in row["concepts"].split(";") if c.strip()]
            all_rows.append((episode, start, end, concepts))
            for concept in concepts:
                if concept not in seen_concepts:
                    seen_concepts.add(concept)
                    concept_order.append(concept)

    canonical_map = fuzzy_cluster(concept_order)

    # For each row, generate all 3-concept combinations and group by sorted canonical triple
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    total_pairs = 0

    for episode, start, end, concepts in all_rows:
        canon_concepts = [canonical_map[c] for c in concepts]
        for c1, c2, c3 in combinations(canon_concepts, 3):
            triple = tuple(sorted([c1, c2, c3]))
            grouped[triple].append({"episode": int(episode), "start": start, "end": end})
            total_pairs += 1

    records = [
        {"query": f"{q[0]}; {q[1]}; {q[2]}", "timestamps": ts}
        for q, ts in grouped.items()
    ]

    with open(output_path, "w") as f:
        yaml.dump(records, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Input rows : {len(all_rows)}")
    print(f"Pairs      : {total_pairs}")
    print(f"Queries    : {len(records)}")
    print(f"Output     : {output_path}")

    # Report merges
    clusters: dict[str, list[str]] = {}
    for raw, canon in canonical_map.items():
        clusters.setdefault(canon, []).append(raw)

    merged = {k: v for k, v in clusters.items() if len(v) > 1}
    if merged:
        print(f"\nFuzzy-merged {len(merged)} concept group(s):")
        for canon, members in merged.items():
            variants = [m for m in members if m != canon]
            print(f"  '{canon}' <- {variants}")
    else:
        print("\nNo concepts were fuzzy-merged (all names distinct enough).")


if __name__ == "__main__":
    main()

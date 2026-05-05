import csv
from pathlib import Path

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
    output_path = Path("data/queries_processed.csv")

    flat_rows: list[tuple[str, str, str, str]] = []
    concept_order: list[str] = []
    seen_concepts: set[str] = set()

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode = row["episode"].strip()
            start = row["start"].strip()
            end = row["end"].strip()
            concepts = [c.strip() for c in row["concepts"].split(";") if c.strip()]
            for concept in concepts:
                flat_rows.append((episode, start, end, concept))
                if concept not in seen_concepts:
                    seen_concepts.add(concept)
                    concept_order.append(concept)

    canonical_map = fuzzy_cluster(concept_order)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "episode", "start", "end"])
        for episode, start, end, concept in flat_rows:
            writer.writerow([canonical_map[concept], episode, start, end])

    print(f"Input rows : {len(flat_rows)}")
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

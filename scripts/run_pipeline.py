#!/usr/bin/env python
"""
scripts/run_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~
Command-line interface for the VideoRAG multimodal grounding pipeline.

Sub-commands
------------
preprocess   Detect scenes, extract keyframes, build segments.csv
index        Build (or rebuild) FAISS indices from segments.csv
query        Ground a single natural-language query
evaluate     Run evaluation on the built-in gold query set
run-all      Run the complete pipeline end-to-end

Usage examples
--------------
# Full pipeline
python scripts/run_pipeline.py run-all --config config.yaml

# Individual steps
python scripts/run_pipeline.py preprocess --config config.yaml
python scripts/run_pipeline.py index      --config config.yaml
python scripts/run_pipeline.py query      --config config.yaml \\
    --text "Ross and Rachel argue about the list"
python scripts/run_pipeline.py evaluate   --config config.yaml

# With custom options
python scripts/run_pipeline.py query \\
    --config config.yaml \\
    --text "Chandler Ross Joey watching ice hockey" \\
    --top-k 10 \\
    --merge-gap 20.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_arg(parser: argparse.ArgumentParser) -> None:
    """Add --config argument to *parser*."""
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml  (default: %(default)s)",
    )


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run scene detection + keyframe extraction → segments.csv."""
    from videorag.config import load_settings
    from videorag.dataset.preprocessing import run_preprocessing

    settings = load_settings(args.config)
    run_preprocessing(settings)


def cmd_index(args: argparse.Namespace) -> None:
    """Build (or force-rebuild) FAISS indices from segments.csv."""
    import pandas as pd
    from videorag.config import load_settings
    from videorag.models.embeddings import load_models
    from videorag.store.index import build_or_load_indices

    settings = load_settings(args.config)
    seg_path = settings.paths.output_root / "segments.csv"

    if not seg_path.exists():
        sys.exit(
            f"ERROR: segments.csv not found at {seg_path}\n"
            "       Run `preprocess` first."
        )

    df = pd.read_csv(seg_path)
    df["subtitle"] = df["subtitle"].fillna("").astype(str)
    df["frames"]   = df["frames"].fillna("[]").astype(str)
    df["embed_text"] = df["subtitle"].apply(
        lambda x: x.strip() if x.strip() else "silent scene no dialogue"
    )

    if args.force:
        # Remove cached indices so they are rebuilt unconditionally
        index_dir = settings.paths.output_root / "indices"
        stamp = index_dir / "segments_mtime.txt"
        if stamp.exists():
            stamp.write_text("")  # invalidate cache
        print("Forcing index rebuild…")

    bundle = load_models(settings)
    build_or_load_indices(df, settings, bundle)


def cmd_query(args: argparse.Namespace) -> None:
    """Ground a single query and print results."""
    from videorag.api import build_context
    from videorag.pipeline.pipeline import run

    ctx = build_context(args.config)
    run(
        args.text,
        ctx,
        top_k=args.top_k,
        show_top=args.show_top,
        merge_gap=args.merge_gap,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate against the built-in gold query set."""
    from videorag.api import build_context
    from videorag.evaluation.evaluation import GOLD_QUERIES, evaluate_grounding

    ctx     = build_context(args.config)
    save_dir = str(ctx.settings.paths.output_root / "results")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    evaluate_grounding(
        GOLD_QUERIES,
        ctx,
        top_k=args.top_k,
        merge_gap=args.merge_gap,
        save_dir=save_dir,
    )


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the complete pipeline: preprocess → index → evaluate."""
    from videorag.api import build_context
    from videorag.config import load_settings
    from videorag.evaluation.evaluation import GOLD_QUERIES, evaluate_grounding
    from videorag.dataset.preprocessing import run_preprocessing

    settings = load_settings(args.config)

    # Step 1: Preprocessing
    print("\n" + "=" * 60)
    print("STEP 1 — PREPROCESSING")
    print("=" * 60)
    run_preprocessing(settings)

    # Step 2–3: Load models + build indices (via build_context)
    print("\n" + "=" * 60)
    print("STEP 2 — LOADING MODELS & BUILDING INDICES")
    print("=" * 60)
    ctx = build_context(args.config)

    # Step 4: Sample queries
    if args.queries:
        print("\n" + "=" * 60)
        print("STEP 3 — QUERY GROUNDING")
        print("=" * 60)
        from videorag.pipeline.pipeline import run
        for q in args.queries:
            run(q, ctx, top_k=args.top_k, merge_gap=args.merge_gap)
            print()

    # Step 5: Evaluation
    print("\n" + "=" * 60)
    print("STEP 4 — EVALUATION")
    print("=" * 60)
    save_dir = str(settings.paths.output_root / "results")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    evaluate_grounding(
        GOLD_QUERIES,
        ctx,
        top_k=args.top_k,
        merge_gap=args.merge_gap,
        save_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videorag",
        description="Multimodal RAG pipeline for video temporal grounding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── preprocess ──────────────────────────────────────────────────────
    p_pre = sub.add_parser(
        "preprocess",
        help="Detect scenes and extract keyframes → segments.csv",
    )
    _config_arg(p_pre)

    # ── index ────────────────────────────────────────────────────────────
    p_idx = sub.add_parser(
        "index",
        help="Build FAISS indices from segments.csv",
    )
    _config_arg(p_idx)
    p_idx.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even when cached indices exist",
    )

    # ── query ────────────────────────────────────────────────────────────
    p_qry = sub.add_parser(
        "query",
        help="Ground a natural-language query",
    )
    _config_arg(p_qry)
    p_qry.add_argument(
        "--text",
        required=True,
        metavar="QUERY",
        help="Query string, e.g. 'Ross and Rachel argue about the list'",
    )
    p_qry.add_argument("--top-k",    type=int,   default=10,   help="Retrieval candidates  (default: 10)")
    p_qry.add_argument("--show-top", type=int,   default=3,    help="Results to display    (default: 3)")
    p_qry.add_argument("--merge-gap", type=float, default=20.0, help="Span merge gap (s)   (default: 20.0)")

    # ── evaluate ─────────────────────────────────────────────────────────
    p_ev = sub.add_parser(
        "evaluate",
        help="Evaluate on the built-in gold query set",
    )
    _config_arg(p_ev)
    p_ev.add_argument("--top-k",    type=int,   default=10,   help="Retrieval candidates  (default: 10)")
    p_ev.add_argument("--merge-gap", type=float, default=20.0, help="Span merge gap (s)   (default: 20.0)")

    # ── run-all ──────────────────────────────────────────────────────────
    p_all = sub.add_parser(
        "run-all",
        help="Run complete pipeline: preprocess → index → query → evaluate",
    )
    _config_arg(p_all)
    p_all.add_argument(
        "--queries",
        nargs="*",
        metavar="QUERY",
        default=[
            "friends discussing kissing in coffee shop",
            "Chandler Ross Joey watching ice hockey",
        ],
        help="Sample queries to run after indexing",
    )
    p_all.add_argument("--top-k",    type=int,   default=10,   help="Retrieval candidates  (default: 10)")
    p_all.add_argument("--merge-gap", type=float, default=20.0, help="Span merge gap (s)   (default: 20.0)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "preprocess": cmd_preprocess,
        "index":      cmd_index,
        "query":      cmd_query,
        "evaluate":   cmd_evaluate,
        "run-all":    cmd_run_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

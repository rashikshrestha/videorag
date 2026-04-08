#!/usr/bin/env python
"""
Run only the VideoRAG preprocessing step.

Usage:
    python scripts/run_preprocessing.py --config config.yaml
"""
from __future__ import annotations

import argparse

from videorag.config import load_settings
from videorag.dataset.preprocessing import run_preprocessing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_preprocessing",
        description="Run VideoRAG preprocessing only (scenes, frames, subtitles, audio clips).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: %(default)s)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    run_preprocessing(settings)


if __name__ == "__main__":
    main()

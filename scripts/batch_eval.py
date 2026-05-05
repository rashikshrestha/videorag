#!/usr/bin/env python
"""
scripts/batch_eval.py
~~~~~~~~~~~~~~~~~~~~~
Run eval_results.py for every JSON file found in data/out/.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("data/out")


def main() -> None:
    json_files = sorted(OUT_DIR.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {OUT_DIR}")

    print(f"Found {len(json_files)} file(s) in {OUT_DIR}\n")

    failed = 0
    for path in json_files:
        print(f"Evaluating: {path}")
        result = subprocess.run([sys.executable, "scripts/eval_results.py", "--results", str(path)])
        if result.returncode != 0:
            failed += 1
            print(f"[FAILED] {path} (exit code {result.returncode})")

    print(f"\nDone. {len(json_files) - failed}/{len(json_files)} evaluations succeeded.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

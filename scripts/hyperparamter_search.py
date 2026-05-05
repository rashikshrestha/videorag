#!/usr/bin/env python
"""
scripts/hyperparamter_search.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Grid search over alpha/beta weights and yaml concept files.

Fixed: --config, --top-k, --merge-gap, --no-refine, --gamma
Variable:
  --yaml   : single_concept, two_concepts, three_concepts
  --alpha  : 0.0 to 1.0 in 0.1 steps
  --beta   : 1 - alpha
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

YAML_FILES = [
    "data/single_concept.yaml",
    "data/two_concepts.yaml",
    "data/three_concepts.yaml",
]

FIXED_ARGS = [
    "--config", "config/pipeline.yaml",
    "--top-k", "5",
    "--merge-gap", "20.0",
    "--no-refine",
    "--gamma", "0",
]


def main() -> None:
    total = 0
    failed = 0

    for yaml_file in YAML_FILES:
        for i in range(11):  # 0 to 10
            alpha = round(i / 10, 1)
            beta = round(1.0 - alpha, 1)

            cmd = [
                sys.executable, "scripts/run_pipeline.py", "query-multi",
                *FIXED_ARGS,
                "--yaml", yaml_file,
                "--alpha", str(alpha),
                "--beta", str(beta),
            ]

            label = f"yaml={Path(yaml_file).stem}  alpha={alpha:.1f}  beta={beta:.1f}"
            print(f"\n{'='*60}")
            print(f"Running: {label}")
            print(f"CMD: {' '.join(cmd)}")
            print('='*60)

            result = subprocess.run(cmd)
            total += 1
            if result.returncode != 0:
                failed += 1
                print(f"[FAILED] {label} (exit code {result.returncode})")

    print(f"\nDone. {total - failed}/{total} runs succeeded.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

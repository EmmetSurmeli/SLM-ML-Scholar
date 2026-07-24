#!/usr/bin/env python3
"""Evaluate the deterministic cited extractive baseline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.grounded_answer_evaluation import (  # noqa: E402
    evaluate_fixture_method,
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic grounded extractive baseline."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "extractive_answer_evaluation",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf"),
        default="bm25",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    summary = evaluate_fixture_method(
        method="extractive",
        output_directory=args.output_directory,
        retrieval_method=args.retriever,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

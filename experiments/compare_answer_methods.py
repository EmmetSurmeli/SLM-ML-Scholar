#!/usr/bin/env python3
"""Compare grounded answer methods under one deterministic fixture."""

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
from localml_scholar._version import __version__  # noqa: E402
from localml_scholar.answering import (  # noqa: E402
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extractive and explicit-checkpoint grounded methods."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "answer_method_comparison",
    )
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.checkpoint is None:
        print(
            "No checkpoint supplied; the controlled four-method comparison "
            "requires --checkpoint and does not fabricate generative results.",
            file=sys.stderr,
        )
        return 2
    destination = args.output_directory
    answerer = GroundedGenerativeAnswerer.from_checkpoint(
        args.checkpoint,
        config=GroundedGenerationConfig(
            maximum_new_tokens=args.maximum_new_tokens,
            greedy=True,
        ),
    )
    runs = []
    for method in (
        "top_passage",
        "extractive",
        "generative",
        "generative_with_extractive_fallback",
    ):
        runs.append(
            evaluate_fixture_method(
                method=method,
                output_directory=destination / method,
                generative_answerer=(
                    answerer if method.startswith("generative") else None
                ),
            )
        )
    summary = {
        "milestone": 9,
        "package_version": __version__,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": answerer.checkpoint_sha256,
        "control": (
            "All methods use the same authored corpus, index construction, "
            "BM25 configuration, questions, and deterministic ordering."
        ),
        "runs": runs,
    }
    path = destination / "comparison_summary.json"
    atomic_write_text(
        path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

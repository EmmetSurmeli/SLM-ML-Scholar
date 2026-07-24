#!/usr/bin/env python3
"""Evaluate an explicitly supplied local checkpoint on grounded generation."""

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
from localml_scholar.answering import (  # noqa: E402
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
)
from localml_scholar.serialization import atomic_write_text  # noqa: E402


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded generation without hiding invalid output."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "grounded_generation_evaluation",
    )
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument(
        "--method",
        choices=("generative", "generative_with_extractive_fallback"),
        default="generative",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.checkpoint is None:
        print(
            "No checkpoint supplied. Pass --checkpoint with an explicit local "
            "model-only .npz containing its tokenizer.",
            file=sys.stderr,
        )
        return 2
    answerer = GroundedGenerativeAnswerer.from_checkpoint(
        args.checkpoint,
        config=GroundedGenerationConfig(
            maximum_new_tokens=args.maximum_new_tokens,
            greedy=True,
        ),
    )
    summary = evaluate_fixture_method(
        method=args.method,
        output_directory=args.output_directory,
        generative_answerer=answerer,
    )
    summary["model"] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": answerer.checkpoint_sha256,
        "configuration": answerer.model.config.to_dict(),
        "parameter_count": answerer.model.parameter_count,
        "tokenizer_type": answerer.tokenizer.tokenizer_type,
        "context_length": answerer.model.config.maximum_context_length,
        "generation": answerer.config.to_dict(),
    }
    atomic_write_text(
        summary["artifacts"]["summary"],
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
